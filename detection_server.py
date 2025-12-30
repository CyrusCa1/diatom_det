import os
# Note: YOLO_VERBOSE will be set dynamically based on dev_log_mode in setup()

import matplotlib
matplotlib.use('Agg')
import json
import argparse
import sys
import winreg
import numpy as np
import imageio.v2 as imageio
import shutil
import time
import warnings
import torch
import base64
import io
import tempfile
import threading
import binascii
import uuid
import logging
import glob # 新增: 用于查找残留文件
from tqdm import tqdm
from flask import Flask, request, jsonify
from waitress import serve
from ultralytics import YOLO
from high_scale.classification import Classification as HighScaleClassification
from PIL import Image

warnings.filterwarnings("ignore")

# Image format whitelist (will be overridden by config)
VALID_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

# Globals for models and config
DEVICE = None
LOW_SCALE_MODEL = None
HIGH_SCALE_CLASSIFIER = None
DEFAULT_CONFIG = None
CLASS_MAPPING = None
PREDICT_CONFIG = None
SERVER_CONFIG = {} # 优化: 默认为空字典避免NoneType
FONT_PATH = None
high_scale_lock = threading.Lock()
low_scale_lock = threading.Lock()

app = Flask(__name__)

def get_device():
    """Detects available device (CUDA, MPS, or CPU)."""
    print("Checking available device...")
    if torch.cuda.is_available():
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

def get_base_path():
    """Gets the base path of the script or frozen executable."""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))

def get_fonts(font_path_cfg):
    """Use specified font or find a suitable default font on Windows."""
    from PIL import ImageFont, ImageDraw, Image as PILImage
    
    def check_valid_fonts(font_path):
        """Check whether the font path does support Chinese. Return true when PIL can use the font to display Chinese correctly."""
        if not os.path.exists(font_path):
            return False
        
        try:
            # Try to load the font with PIL
            font = ImageFont.truetype(font_path, size=20)
            
            # Test with a common Chinese character
            test_char = "中"
            test_img = PILImage.new('RGB', (50, 50), color='white')
            draw = ImageDraw.Draw(test_img)
            draw.text((10, 10), test_char, font=font, fill='black')
            
            # If we got here without exception, the font supports Chinese
            return True
        except Exception:
            # Font doesn't support Chinese or can't be loaded
            return False
    
    def get_windows_fonts():
        """Find a suitable default font on Windows that supports Chinese."""
        fonts_path = os.path.join(os.environ['WINDIR'], 'Fonts')
        try:
            reg_key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts")
            fonts = {}
            for i in range(winreg.QueryInfoKey(reg_key)[1]):
                font_name, font_file, _ = winreg.EnumValue(reg_key, i)
                fonts[font_name] = os.path.join(fonts_path, font_file)
            winreg.CloseKey(reg_key)
            
            # Preferred Chinese fonts (in priority order)
            preferred_fonts = ["Microsoft YaHei", "SimHei", "SimSun", "Arial", "Segoe UI"]
            
            # First pass: try preferred fonts with validation
            for font_name in preferred_fonts:
                for key, path in fonts.items():
                    if font_name in key and os.path.exists(path):
                        # Validate this font supports Chinese
                        if check_valid_fonts(path):
                            return path
            
            # Second pass (fallback): try any font that supports Chinese
            for path in fonts.values():
                if os.path.exists(path) and check_valid_fonts(path):
                    return path
        except Exception as e:
            print(f"Error while fetching fonts: {e}")
        return None

    # Priority 1: Use configured font if valid
    if font_path_cfg is not None and check_valid_fonts(font_path_cfg):
        return font_path_cfg
    else:
        # Priority 2: Find system font that supports Chinese
        return get_windows_fonts()


def calculate_optimal_batch_size(requested_batch):
    """
    Calculates the safe batch size based on available VRAM.
    Assumption: ~300MB VRAM per image (640sz, FP16) + Overhead.
    """
    if DEVICE != 'cuda':
        return 1 # CPU or MPS, safe default

    try:
        # Get free memory of the first GPU in bytes
        free_mem, total_mem = torch.cuda.mem_get_info(0)
        
        # Reserve 2GB for system/overhead
        safe_mem = max(0, free_mem - (2 * 1024**3))
        
        # Estimate per image cost (conservative estimate: 300MB)
        per_image_cost = 300 * 1024 * 1024 
        
        estimated_max_batch = int(safe_mem / per_image_cost)
        estimated_max_batch = max(1, estimated_max_batch) # At least 1
        
        final_batch = min(requested_batch, estimated_max_batch)
        # print(f"[Batch Logic] Req: {requested_batch}, FreeVRAM: {free_mem/1024**3:.2f}GB, EstMax: {estimated_max_batch} -> Final: {final_batch}")
        return final_batch
    except Exception as e:
        # print(f"[Batch Logic] Error estimating VRAM: {e}. Fallback to {requested_batch}")
        return requested_batch

def filter_valid_images(folder_path):
    """
    Filters valid images using os.scandir for better performance on large directories.
    Returns: bool (True if at least one valid image found)
    """
    try:
        with os.scandir(folder_path) as it:
            for entry in it:
                if entry.is_file():
                    ext = os.path.splitext(entry.name)[1].lower()
                    if ext in VALID_IMAGE_EXTENSIONS:
                        return True # Found at least one, return True quickly
    except Exception as e:
        print(f"Error scanning folder: {e}")
    return False

def filter_and_delete_invalid_content(folder_path):
    """
    Deletes all files and subdirectories that are not in the valid image extensions whitelist.
    Only used when SERVER_CONFIG['filt_and_delete_invalid_content'] is True.
    """
    try:
        with os.scandir(folder_path) as it:
            for entry in it:
                if entry.is_dir():
                    # Delete subdirectories
                    try:
                        shutil.rmtree(entry.path)
                    except Exception as e:
                        print(f"Warning: Could not delete directory {entry.path}: {e}")
                elif entry.is_file():
                    ext = os.path.splitext(entry.name)[1].lower()
                    if ext not in VALID_IMAGE_EXTENSIONS:
                        # Delete non-whitelisted files
                        try:
                            os.remove(entry.path)
                        except Exception as e:
                            print(f"Warning: Could not delete file {entry.path}: {e}")
    except Exception as e:
        print(f"Error filtering folder: {e}")

def cleanup_stale_temp_files():
    """
    Cleans up stale temporary files/directories left over from previous crashed sessions.
    Runs once at server startup if configured.
    """
    if not SERVER_CONFIG.get('check_and_delete_residual_tmp', True):
        return

    print("Checking for stale temporary files...")
    temp_dir = tempfile.gettempdir()
    
    # Define patterns to look for based on our naming conventions
    patterns = [
        "yolo_input_*",      # Base64 input temp dirs
        "diatom_jpg_*",      # High-scale conversion temp dirs
        "yolo_cache_*.jsonl" # JSONL buffer files
    ]
    
    count = 0
    start_t = time.time()
    for pattern in patterns:
        full_pattern = os.path.join(temp_dir, pattern)
        for target in glob.glob(full_pattern):
            try:
                # Basic check: verify it's ours and not currently in use (hard to detect lock, but relying on unique UUIDs)
                # Since we generate new UUIDs/timestamps on every request, old named files are safe to delete.
                if os.path.isdir(target):
                    shutil.rmtree(target, ignore_errors=True)
                else:
                    os.remove(target)
                count += 1
            except Exception as e:
                print(f"Warning: Failed to clean stale file {target}: {e}")
    
    if count > 0:
        print(f"Cleaned up {count} stale temporary items (Time: {time.time() - start_t:.3f}s)")
    else:
        print("No stale temporary files found.")

def convert_to_jpg(image_path, output_dir):
    """
    Converts images to JPG format (for high-scale model only).
    Returns: (output_dir, filename_mapping)
        - output_dir: path to the directory with converted images
        - filename_mapping: dict mapping converted filename (jpg) -> original filename
    """
    os.makedirs(output_dir, exist_ok=True)
    filename_mapping = {}  # Maps converted filename -> original filename

    def process_image(img_path, dest_dir, original_name):
        img_name = os.path.basename(os.path.splitext(img_path)[0])
        jpg_filename = img_name + ".jpg"
        jpg_image_path = os.path.join(dest_dir, jpg_filename)
        
        # Store mapping
        filename_mapping[jpg_filename] = original_name
        
        if os.path.abspath(img_path) == os.path.abspath(jpg_image_path):
            return
        try:
            img = imageio.imread(img_path)
            if img.dtype == np.uint16:
                img = (img / 256).astype(np.uint8)
            elif img.dtype in [np.float32, np.float64]:
                img = (255 * img).astype(np.uint8)
            if img.ndim == 3 and img.shape[2] == 4:
                img = img[:, :, :3]
            imageio.imwrite(jpg_image_path, img)
        except Exception as e:
            print(f"Error converting {img_path} to JPG: {e}", file=sys.stderr)

    if os.path.isfile(image_path):
        original_name = os.path.basename(image_path)
        process_image(image_path, output_dir, original_name)
    else:
        # Use scandir for performance
        with os.scandir(image_path) as it:
            for entry in it:
                if entry.is_file():
                    ext = os.path.splitext(entry.name)[1].lower()
                    if ext in VALID_IMAGE_EXTENSIONS:
                        if ext in ['.jpg', '.jpeg']:
                            shutil.copy(entry.path, output_dir)
                            # For jpg files, mapping is identity
                            filename_mapping[entry.name] = entry.name
                        else:
                            process_image(entry.path, output_dir, entry.name)
    return output_dir, filename_mapping

def process_low_scale(source_path, output_require, class_list, conf, iou, batch_size, imgsz, half, rect, max_det, augment, is_base64_input=False):
    """Handles low-scale object detection with streaming."""
    # 当use_high_fidelity_model为true时, 使用专门的高精度但是类别数量更少的专用模型对困难类别进行专门的检测, 测试人员会保证启动服务器时, config.json以及模型权重都是配套的高精度模型的配置. 这是测试人员的一个临时需求, 在正式项目中不应该被使用, 这点由模型调用者保证.
    use_high_fidelity_model = SERVER_CONFIG.get('use_high_fidelity_model', False)
    high_fidelity_model_classes = SERVER_CONFIG.get('high_fidelity_model_classes', ['沟链藻', '圆筛藻', '舟形藻', '小环藻', '双眉藻', '异极藻', '菱形藻'])
    if use_high_fidelity_model:
        class_list = high_fidelity_model_classes
    save_folder = output_require.get('save_folder')
    no_local_save = output_require.get('no_local_save', False)
    no_response_save = output_require.get('no_response_save', False)
    return_images = output_require.get('return_images', False)
    
    json_results = []
    rendered_images = []
    
    # 策略优化：如果是文件夹输入（潜在的大规模数据），无论 no_local_save 为何值，
    # 都强烈建议使用 JSONL 缓存，以防止内存溢出。
    # 只有 Base64 输入（通常量小）才直接存内存。
    use_jsonl_cache = not is_base64_input
    jsonl_cache_path = None
    jsonl_file_handle = None

    if use_jsonl_cache:
        # 使用 mkstemp 保证文件名唯一，防止并发冲突
        fd, jsonl_cache_path = tempfile.mkstemp(prefix=f"yolo_cache_{uuid.uuid4().hex}_", suffix=".jsonl")
        os.close(fd) # 关闭底层文件句柄，使用 open 上下文管理
        jsonl_file_handle = open(jsonl_cache_path, 'w', encoding='utf-8')

    batch_buffer = []
    BATCH_SIZE = 1000  # Write to disk every 1000 results
    
    # Progress bar setup (only in non-dev mode)
    pbar = None
    dev_log_mode = SERVER_CONFIG.get('dev_log_mode', False)
    
    # Count total images for progress bar
    total_images = None
    if not dev_log_mode:
        try:
            total_images = 0
            with os.scandir(source_path) as it:
                for entry in it:
                    if entry.is_file():
                        ext = os.path.splitext(entry.name)[1].lower()
                        if ext in VALID_IMAGE_EXTENSIONS:
                            total_images += 1
        except Exception:
            total_images = None
    
    try:
        with low_scale_lock:
            # YOLOv8 原生支持文件夹路径，会自动递归或过滤，速度最快
            # Suppress YOLO output in non-dev mode
            verbose = dev_log_mode
            
            results = LOW_SCALE_MODEL.predict(
                source=source_path, 
                save_conf=True, 
                iou=iou, 
                conf=conf, 
                batch=batch_size,
                imgsz=imgsz,
                half=half,
                rect=rect,
                max_det=max_det,
                augment=augment,
                device=DEVICE,
                stream=True,  # Enable streaming for memory efficiency
                verbose=verbose  # Control YOLO logging
            )
            
            # Initialize progress bar for non-dev mode
            if not dev_log_mode:
                pbar = tqdm(desc="Processing images", unit="img", total=total_images, dynamic_ncols=True)
        
            for result in results:
                json_data = {}
                filename_without_ext = os.path.splitext(os.path.basename(result.path))[0]
                # 在predict之后, plot之前修改result中的类别映射
                for i in range(len(result.names)):
                    result.names[i] = class_list[i]
                
                # Update progress bar
                if pbar is not None:
                    pbar.update(1)
                
                # Save rendered image if requested
                if return_images:
                    if not no_local_save:
                        result_img_filename = f"{filename_without_ext}_result.jpg"
                        # Use plot() to render with custom font and class names
                        try:
                            # plot() will use the model's updated names automatically
                            rendered_result = result.plot(font=FONT_PATH)
                            
                            # Handle both old (numpy array) and new (PIL Image) return types
                            if isinstance(rendered_result, Image.Image):
                                # New version: already a PIL Image
                                img_pil = rendered_result
                            else:
                                # Old version: numpy array, need to convert BGR to RGB
                                img_pil = Image.fromarray(rendered_result[..., ::-1])
                            
                            # Save using PIL for better Chinese path support
                            img_pil.save(os.path.join(save_folder, result_img_filename))
                        except Exception as e:
                            # Fallback: try direct save
                            try:
                                result.save(filename=os.path.join(save_folder, result_img_filename))
                            except Exception:
                                pass
                    
                    if no_local_save:
                        # Encode to base64 for response
                        # Warning: Doing this for 120k images will create a MASSIVE response.
                        # Usually clients shouldn't ask for return_images=True on huge batches.
                        try:
                            # plot() will use the model's updated names automatically
                            rendered_result = result.plot(font=FONT_PATH)
                            
                            # Handle both old (numpy array) and new (PIL Image) return types
                            if isinstance(rendered_result, Image.Image):
                                # New version: already a PIL Image
                                img_pil = rendered_result
                            else:
                                # Old version: numpy array, need to convert BGR to RGB
                                img_pil = Image.fromarray(rendered_result[..., ::-1])
                            
                            buffered = io.BytesIO()
                            img_pil.save(buffered, format="JPEG")
                            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                            rendered_images.append({
                                "image_name": os.path.basename(result.path),
                                "image_base64": f"data:image/jpeg;base64,{img_base64}",
                                "format": "jpg"
                            })
                        except Exception as e:
                            # Skip this image if rendering fails
                            pass
                
                # Build JSON result
                json_data['source'] = os.path.basename(result.path)
                json_data['resultpic'] = f"{filename_without_ext}_result.jpg" if return_images else ""
                json_data['diatoms'] = []
                
                for i in range(len(result.obb.cls)):
                    x, y, w, h, angle = [elem.item() for elem in result.obb.xywhr[i]]
                    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = [(x.item(), y.item()) for (x, y) in result.obb.xyxyxyxy[i]]
                    json_data['diatoms'].append({
                        'type': class_list[result.obb.cls[i].int().item()],
                        'confidence': result.obb.conf[i].item(),
                        'location': {'x': x, 'y': y, 'w': w, 'h': h, 'angle': angle, 
                                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 
                                    'x3': x3, 'y3': y3, 'x4': x4, 'y4': y4}
                    })
                
                # Handle result storage
                if not use_jsonl_cache:
                    # Small batch (Base64), just keep in memory
                    if not no_response_save:
                        json_results.append(json_data)
                    
                    # Local save for Base64 (individual files)
                    if not no_local_save:
                         json_path = os.path.join(save_folder, f"{filename_without_ext}_result.json")
                         with open(json_path, "w", encoding='utf-8') as f:
                            json.dump(json_data, f, indent=4, ensure_ascii=False)
                else:
                    # Large batch (Folder), buffer to JSONL
                    batch_buffer.append(json_data)
                    
                    # Save individual JSON if configured
                    if SERVER_CONFIG.get('save_every_json_result', False) and not no_local_save:
                        json_path = os.path.join(save_folder, f"{filename_without_ext}_result.json")
                        with open(json_path, "w", encoding='utf-8') as f:
                            json.dump(json_data, f, indent=4, ensure_ascii=False)
                    
                    if len(batch_buffer) >= BATCH_SIZE:
                        for item in batch_buffer:
                            jsonl_file_handle.write(json.dumps(item, ensure_ascii=False) + '\n')
                        batch_buffer.clear()
        
        # Flush remaining batch
        if batch_buffer and jsonl_file_handle:
            for item in batch_buffer:
                jsonl_file_handle.write(json.dumps(item, ensure_ascii=False) + '\n')
            batch_buffer.clear()

        # Close handle before reading back
        if jsonl_file_handle:
            jsonl_file_handle.close()
        
        # Post-Processing for Folder Input
        if use_jsonl_cache and os.path.exists(jsonl_cache_path):
            
            # 1. If local save is required, convert JSONL to standard JSON
            if not no_local_save:
                final_json_path = os.path.join(save_folder, "detection_results.json")
                with open(final_json_path, 'w', encoding='utf-8') as f_out:
                    f_out.write('[\n')
                    with open(jsonl_cache_path, 'r', encoding='utf-8') as f_in:
                        first_line = True
                        for line in f_in:
                            if not first_line: f_out.write(',\n')
                            f_out.write(line.strip())
                            first_line = False
                    f_out.write('\n]')

            # 2. If response is required, load into memory (RISKY for large datasets)
            if not no_response_save:
                # Warning: Loading 120k results into RAM might OOM.
                # But if client asked for it, we try.
                with open(jsonl_cache_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        json_results.append(json.loads(line.strip()))
    
    except Exception as e:
        print(f"Error in low scale process: {e}")
        raise e
    finally:
        # Close progress bar
        if pbar is not None:
            pbar.close()
        
        # Cleanup
        if jsonl_file_handle and not jsonl_file_handle.closed:
            jsonl_file_handle.close()
        if jsonl_cache_path and os.path.exists(jsonl_cache_path):
            try:
                os.remove(jsonl_cache_path)
            except:
                pass

    return json_results, rendered_images

def process_high_scale(source_path, output_require, class_mapping, filename_mapping=None):
    """
    Handles high-scale image classification.
    
    Args:
        source_path: Path to the directory with images (converted to jpg)
        output_require: Output configuration
        class_mapping: Class mapping dictionary
        filename_mapping: Optional dict mapping converted filename -> original filename
    """
    classifier = HIGH_SCALE_CLASSIFIER
    # Update class mapping if a custom one is provided in the request
    if class_mapping != classifier.class_mapping:
        classifier.class_mapping = class_mapping

    json_results = []
    save_folder = output_require.get('save_folder')
    no_local_save = output_require.get('no_local_save', False)
    dev_log_mode = SERVER_CONFIG.get('dev_log_mode', False)
    
    # Count total images for progress bar
    total_images = None
    if not dev_log_mode:
        try:
            total_images = sum(1 for entry in os.scandir(source_path) if entry.is_file())
        except Exception:
            total_images = None
    
    pbar = None
    if not dev_log_mode:
        pbar = tqdm(desc="Classifying images", unit="img", total=total_images, dynamic_ncols=True)

    # Use scandir for performance
    try:
        with os.scandir(source_path) as it:
            for entry in it:
                if entry.is_file():
                    try:
                        image = Image.open(entry.path)
                        with high_scale_lock:
                            class_name, probability = classifier.detect_image(image)
                        
                        # Use original filename if mapping exists, otherwise use current name
                        original_filename = filename_mapping.get(entry.name, entry.name) if filename_mapping else entry.name
                        
                        detection = {
                            "source": original_filename,
                            "type": class_name,
                            "confidence": float(probability),
                        }
                        json_results.append(detection)
                        
                        if pbar is not None:
                            pbar.update(1)
                    except Exception as e:
                        # Skip non-images or bad files
                        if pbar is not None:
                            pbar.update(1)
    except Exception as e:
        print(f"Error scanning high scale folder: {e}")
    finally:
        if pbar is not None:
            pbar.close()

    if not no_local_save and json_results:
        json_path = os.path.join(save_folder, f"{os.path.basename(os.path.normpath(source_path))}_result.json")
        with open(json_path, "w", encoding='utf-8') as f:
            json.dump(json_results, f, indent=4, ensure_ascii=False)
            
    return json_results, [] 

def setup():
    """Initializes models, configurations, and device."""
    global DEVICE, LOW_SCALE_MODEL, HIGH_SCALE_CLASSIFIER, DEFAULT_CONFIG, CLASS_MAPPING, PREDICT_CONFIG, SERVER_CONFIG, FONT_PATH, VALID_IMAGE_EXTENSIONS
    
    base_path = get_base_path()
    DEVICE = get_device()
    print(f"Using device: {DEVICE}")

    # Config loading
    config_path = os.path.join(base_path, 'logs', 'config.json')
    print(f"Loading config from: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        DEFAULT_CONFIG = json.load(f)
    CLASS_MAPPING = DEFAULT_CONFIG["CLASS_MAPPING"]
    PREDICT_CONFIG = DEFAULT_CONFIG["predict_config"]
    SERVER_CONFIG = DEFAULT_CONFIG.get("server_config", {
        "filt_and_delete_invalid_content": True,
        "check_and_delete_residual_tmp": True,
        "valid_img_ext": [".jpg", ".png", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"],
        "dev_log_mode": False,
        "save_every_json_result": False,
        "default_font_path": r"logs/SimHei.ttf"  # if this path are none or unloadable, we will search the registry table to get a default font
    })
    
    # Update valid image extensions from config
    VALID_IMAGE_EXTENSIONS = set(ext.lower() for ext in SERVER_CONFIG.get('valid_img_ext', VALID_IMAGE_EXTENSIONS))
    
    # Set YOLO verbosity based on dev_log_mode
    dev_log_mode = SERVER_CONFIG.get('dev_log_mode', False)
    if dev_log_mode:
        # Enable YOLO verbose output in dev mode
        os.environ['YOLO_VERBOSE'] = 'True'
    else:
        # Suppress YOLO output in production mode
        os.environ['YOLO_VERBOSE'] = 'False'
    
    print(f"Server config loaded: dev_log_mode={SERVER_CONFIG.get('dev_log_mode')}, "
          f"filt_and_delete_invalid_content={SERVER_CONFIG.get('filt_and_delete_invalid_content')}, "
          f"check_and_delete_residual_tmp={SERVER_CONFIG.get('check_and_delete_residual_tmp')}, "
          f"save_every_json_result={SERVER_CONFIG.get('save_every_json_result')}, "
          f"default_font_path={SERVER_CONFIG.get('default_font_path')}")

    # Low-scale model loading
    low_scale_model_path = os.path.join(base_path, 'logs', 'low_scale.pt')
    print(f"Loading low-scale model from: {low_scale_model_path}")
    LOW_SCALE_MODEL = YOLO(low_scale_model_path)
    LOW_SCALE_MODEL.to(DEVICE)
    print(f"Low-scale model loaded. Model device: {next(LOW_SCALE_MODEL.parameters()).device}")

    # High-scale model loading
    high_scale_model_path = os.path.join(base_path, 'logs', 'high_scale.pth')
    print(f"Loading high-scale model from: {high_scale_model_path}")
    HIGH_SCALE_CLASSIFIER = HighScaleClassification(model_path=high_scale_model_path, class_mapping=CLASS_MAPPING)
    print("High-scale model loaded.")

    # Font loading
    font_path_cfg = SERVER_CONFIG.get('default_font_path')
    if font_path_cfg:
        # Convert relative path to absolute path
        if not os.path.isabs(font_path_cfg):
            font_path_cfg = os.path.join(base_path, font_path_cfg)
    
    FONT_PATH = get_fonts(font_path_cfg)
    if FONT_PATH:
        print(f"Using font: {FONT_PATH}")
    else:
        print("No suitable font found. Result images will use default font.")

    # Perform startup cleanup if enabled
    cleanup_stale_temp_files()

@app.route('/detection', methods=['POST'])
def detection_endpoint():
    start_time = time.time()
    
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "code": 400, "message": "Invalid JSON"}), 400

    request_id = data.get("request_id")
    
    # Log request in non-dev mode
    if not SERVER_CONFIG.get('dev_log_mode', False):
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Received request: {request_id}")
    
    task_type = data.get("task_type")
    scale = data.get("scale")
    input_data = data.get("input")
    output_require = data.get("output_require")
    config_override = data.get("config", {})

    # --- Validation ---
    if not all([request_id, task_type, scale, input_data, output_require]):
        error_msg = "Missing required fields"
        if not SERVER_CONFIG.get('dev_log_mode', False):
            print(f"  ❌ Validation failed: {error_msg}")
        return jsonify({"request_id": request_id, "status": "error", "code": 400, "message": error_msg}), 400
    if task_type != "detection":
        error_msg = f"Unsupported task_type: {task_type}"
        if not SERVER_CONFIG.get('dev_log_mode', False):
            print(f"  ❌ Validation failed: {error_msg}")
        return jsonify({"request_id": request_id, "status": "error", "code": 400, "message": error_msg}), 400
    if scale not in ["low", "high"]:
        error_msg = f"Invalid scale: {scale}"
        if not SERVER_CONFIG.get('dev_log_mode', False):
            print(f"  ❌ Validation failed: {error_msg}")
        return jsonify({"request_id": request_id, "status": "error", "code": 400, "message": error_msg}), 400
    
    no_local_save = output_require.get('no_local_save', False)
    no_response_save = output_require.get('no_response_save', False)
    return_images = output_require.get('return_images', False)
    save_folder = output_require.get('save_folder')
    
    # Validate output_require logic
    if no_local_save and no_response_save:
        error_msg = "Both no_local_save and no_response_save cannot be true. Results must be saved somewhere."
        if not SERVER_CONFIG.get('dev_log_mode', False):
            print(f"  ❌ Validation failed: {error_msg}")
        return jsonify({"request_id": request_id, "status": "error", "code": 400, "message": error_msg}), 400
    
    if not no_local_save:
        # save_folder is required when saving locally
        if not save_folder:
            error_msg = "save_folder is required when no_local_save is false"
            if not SERVER_CONFIG.get('dev_log_mode', False):
                print(f"  ❌ Validation failed: {error_msg}")
            return jsonify({"request_id": request_id, "status": "error", "code": 400, "message": error_msg}), 400
        
        # save_folder must exist
        if not os.path.isdir(save_folder):
            error_msg = f"save_folder does not exist: {save_folder}"
            if not SERVER_CONFIG.get('dev_log_mode', False):
                print(f"  ❌ Validation failed: {error_msg}")
            return jsonify({"request_id": request_id, "status": "error", "code": 400, "message": error_msg}), 400
        
        # save_folder must be writable
        if not os.access(save_folder, os.W_OK):
            error_msg = f"save_folder is not writable: {save_folder}"
            if not SERVER_CONFIG.get('dev_log_mode', False):
                print(f"  ❌ Validation failed: {error_msg}")
            return jsonify({"request_id": request_id, "status": "error", "code": 400, "message": error_msg}), 400

    # --- Config ---
    conf = config_override.get("predict_config", {}).get("conf", PREDICT_CONFIG['conf'])
    iou = config_override.get("predict_config", {}).get("iou", PREDICT_CONFIG['iou'])
    imgsz = config_override.get("predict_config", {}).get("imgsz", PREDICT_CONFIG['imgsz'])
    half = config_override.get("predict_config", {}).get("half", PREDICT_CONFIG['half'])
    rect = config_override.get("predict_config", {}).get("rect", PREDICT_CONFIG['rect'])
    augment = config_override.get("predict_config", {}).get("augment", PREDICT_CONFIG['augment'])
    max_det = config_override.get("predict_config", {}).get("max_det", PREDICT_CONFIG['max_det'])
    requested_batch = config_override.get("predict_config", {}).get("batch", PREDICT_CONFIG.get('batch', 1))
    batch_size = calculate_optimal_batch_size(requested_batch)
    class_mapping = config_override.get("CLASS_MAPPING", CLASS_MAPPING)
    
    # --- Input Processing ---
    temp_input_dir = None
    temp_jpg_dir = None
    input_source = None # Holds either path string or list of paths
    is_base64_input = False
    
    try:
        input_type = input_data.get("input_type")
        
        if input_type == "folder":
            input_path = input_data.get("folder_path")
            if not input_path or not os.path.isdir(input_path):
                error_msg = f"Input folder not found: {input_path}"
                if not SERVER_CONFIG.get('dev_log_mode', False):
                    print(f"  ❌ Validation failed: {error_msg}")
                return jsonify({"request_id": request_id, "status": "error", "code": 400, "message": error_msg}), 400
            
            # Filter and delete invalid content if configured
            if SERVER_CONFIG.get('filt_and_delete_invalid_content', False):
                filter_and_delete_invalid_content(input_path)
            
            # Fast check if folder has any valid images (avoids loading model for empty folders)
            if not filter_valid_images(input_path):
                error_msg = "No valid images found in folder."
                if not SERVER_CONFIG.get('dev_log_mode', False):
                    print(f"  ❌ Validation failed: {error_msg}")
                return jsonify({"request_id": request_id, "status": "error", "code": 400, 
                               "message": error_msg}), 400
            
            input_source = input_path
                
        elif input_type == "imgbase64_list":
            is_base64_input = True
            img_list = input_data.get("imgbase64_list", [])
            if not img_list:
                error_msg = "imgbase64_list is empty"
                if not SERVER_CONFIG.get('dev_log_mode', False):
                    print(f"  ❌ Validation failed: {error_msg}")
                return jsonify({"request_id": request_id, "status": "error", "code": 400, "message": error_msg}), 400
            
            temp_input_dir = tempfile.mkdtemp(prefix='yolo_input_')
            valid_count = 0
            invalid_images = []
            filtered_images = []
            
            for img_item in img_list:
                img_name = img_item.get('image_name', 'unknown')
                try:
                    b64_str = img_item['image_base64']
                    # Robust base64 handling
                    if ',' in b64_str:
                        _, b64_data = b64_str.split(',', 1)
                    else:
                        b64_data = b64_str
                        
                    img_bytes = base64.b64decode(b64_data)
                    img_path = os.path.join(temp_input_dir, img_name)
                    
                    # Check file extension against whitelist
                    ext = os.path.splitext(img_name)[1].lower()
                    if ext not in VALID_IMAGE_EXTENSIONS:
                        filtered_images.append(img_name)
                        if SERVER_CONFIG.get('dev_log_mode', False):
                            print(f"Filtered out non-whitelisted image: {img_name} (extension: {ext})")
                        continue
                    
                    with open(img_path, 'wb') as f:
                        f.write(img_bytes)
                    
                    # Validate the image can be opened
                    try:
                        with Image.open(img_path) as img:
                            img.verify()
                        valid_count += 1
                    except Exception as e:
                        # Remove invalid image file
                        os.remove(img_path)
                        invalid_images.append(img_name)
                        if SERVER_CONFIG.get('dev_log_mode', False):
                            print(f"Removed invalid/corrupted image: {img_name} ({str(e)})")
                        
                except Exception as e:
                    invalid_images.append(img_name)
                    if SERVER_CONFIG.get('dev_log_mode', False):
                        print(f"Skipping malformed base64 for {img_name}: {e}")
            
            # Build detailed error message
            if valid_count == 0:
                error_parts = [f"No valid images in imgbase64_list. Total: {len(img_list)}"]
                if filtered_images:
                    error_parts.append(f"Filtered (non-whitelisted): {len(filtered_images)}")
                if invalid_images:
                    error_parts.append(f"Invalid/corrupted: {len(invalid_images)}")
                error_msg = ". ".join(error_parts)
                
                if not SERVER_CONFIG.get('dev_log_mode', False):
                    print(f"  ❌ Validation failed: {error_msg}")
                
                return jsonify({"request_id": request_id, "status": "error", "code": 400, 
                               "message": error_msg}), 400
            
            input_source = temp_input_dir
        else:
            error_msg = f"Invalid input_type: {input_type}"
            if not SERVER_CONFIG.get('dev_log_mode', False):
                print(f"  ❌ Validation failed: {error_msg}")
            return jsonify({"request_id": request_id, "status": "error", "code": 400, "message": error_msg}), 400

        # --- Detection ---
        json_results, rendered_images = [], []
        
        if scale == 'low':
            # Direct YOLO processing for folder (fastest)
            class_list = {value: key for key, value in class_mapping.items()}
            json_results, rendered_images = process_low_scale(input_source, output_require, class_list, conf, iou, batch_size, imgsz, half, rect, max_det, augment, is_base64_input)
            
        elif scale == 'high':
            # High-scale needs standard conversion still
            temp_jpg_dir = tempfile.mkdtemp(prefix='diatom_jpg_')
            jpg_path, filename_mapping = convert_to_jpg(input_source, temp_jpg_dir)
            json_results, rendered_images = process_high_scale(jpg_path, output_require, class_mapping, filename_mapping)

        # --- Response ---
        end_time = time.time()
        
        msg = "detection success"
        if no_response_save:
            msg += ". Results saved to disk (JSON/JSONL) but omitted from response."
            
        response = {
            "request_id": request_id,
            "status": "success",
            "code": 200,
            "message": msg,
            "result": {
                "task_time_ms": int((end_time - start_time) * 1000),
                "json_results": json_results,
                "rendered_images": rendered_images
            }
        }
        return jsonify(response), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"request_id": request_id, "status": "error", "code": 500, "message": str(e)}), 500
    finally:
        # --- Cleanup ---
        if temp_input_dir and os.path.exists(temp_input_dir):
            shutil.rmtree(temp_input_dir, ignore_errors=True)
        if temp_jpg_dir and os.path.exists(temp_jpg_dir):
            shutil.rmtree(temp_jpg_dir, ignore_errors=True)

@app.route('/shutdown', methods=['POST'])
def shutdown():
    """Shuts down the server."""
    if request.remote_addr != '127.0.0.1':
        return jsonify({"status": "error", "message": "Shutdown only allowed from localhost."}), 403
    
    def delayed_shutdown():
        time.sleep(1)
        os._exit(0)

    threading.Thread(target=delayed_shutdown).start()
    return jsonify({"status": "success", "message": "Server is shutting down..."})

# --- Logging Setup ---
class Logger(object):
    def __init__(self, filename="default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

def setup_logging():
    """Redirects stdout and stderr to a log file and the console."""
    base_path = get_base_path()
    log_file_path = os.path.join(base_path, "detection_server.log")
    
    try:
        if os.path.exists(log_file_path) and os.path.getsize(log_file_path) > 5 * 1024 * 1024: # 5 MB
            os.rename(log_file_path, log_file_path + ".old")
    except Exception as e:
        print(f"Could not rotate log file: {e}", file=sys.__stderr__)

    sys.stdout = Logger(log_file_path, sys.stdout)
    sys.stderr = Logger(log_file_path, sys.stderr)

if __name__ == "__main__":
    setup_logging()
    print(f"\n--- Server Initializing @ {time.strftime('%Y-%m-%d %H:%M:%S')} ---")

    parser = argparse.ArgumentParser(description="YOLO Detection Server")
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on.')
    args = parser.parse_args()
    
    setup()
    
    # Configure logging based on dev_log_mode
    if not SERVER_CONFIG.get('dev_log_mode', False):
        # Suppress waitress warnings in production mode
        waitress_logger = logging.getLogger('waitress')
        waitress_logger.setLevel(logging.ERROR)
        
        # Suppress ultralytics warnings (already set in env, but belt and suspenders)
        # os.environ['YOLO_VERBOSE'] = 'False' 
    
    print(f"Handing off to Waitress server. Listening on http://0.0.0.0:{args.port}")
    serve(app, host='0.0.0.0', port=args.port, threads=6)