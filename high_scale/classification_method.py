import os
import json
import argparse
from PIL import Image
from high_scale.classification import Classification
from tqdm import tqdm
import sys


def process_image(image_path, classfication, dir_save_path):
    try:
        image = Image.open(image_path)

        # 获取处理后的图像和识别结果
        class_name, probability = classfication.detect_image(image)

        # 构建 JSON 结构
        detections = {
            "source": os.path.basename(image_path),
            "type": class_name,
            "confidence": probability,
        }
        detections["confidence"] = float(detections["confidence"])
        print(detections)
        return detections

    except Exception as e:
        print(f"Error: 处理图像 {os.path.basename(image_path)} 时出错: {e}", file=sys.stderr)
        sys.exit(3)

def main():
    # 配置参数解析器
    parser = argparse.ArgumentParser(description="处理图像并保存结果。")
    parser.add_argument('input_path', type=str, help='输入的图像文件路径或文件夹路径。')
    parser.add_argument('output_dir', type=str, help='保存处理后图像和JSON结果的输出文件夹路径。')

    args = parser.parse_args()

    input_path = args.input_path  # 获取输入路径
    dir_save_path = args.output_dir  # 获取输出文件夹路径

    temp_files = []  # 用于跟踪临时文件的列表

    try:
        # 确保输出目录存在，否则创建它
        if not os.path.exists(dir_save_path):
            os.makedirs(dir_save_path)

        classfication = Classification()

        # 判断输入路径是文件还是文件夹
        if os.path.isfile(input_path):
            # 处理单个文件
            detections = process_image(input_path, classfication, dir_save_path)

            # 保存单个文件的检测结果到JSON文件
            json_filename = f"{os.path.splitext(os.path.basename(input_path))[0]}_result.json"
            json_path = os.path.join(dir_save_path, json_filename)

            try:
                with open(json_path, "w") as f:
                    json.dump(detections, f, indent=4)
            except Exception as e:
                print(f"Error: 保存JSON文件时出错: {e}", file=sys.stderr)
                sys.exit(4)

        elif os.path.isdir(input_path):
            # 处理文件夹中的所有图像
            img_names = os.listdir(input_path)
            if not img_names:
                print(f"Error: 输入目录 {input_path} 中没有图像文件。", file=sys.stderr)
                sys.exit(2)

            all_detections = []

            for img_name in tqdm(img_names):
                if img_name.lower().endswith(
                        ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                    image_path = os.path.join(input_path, img_name)
                    detections = process_image(image_path, classfication, dir_save_path)
                    all_detections.append(detections)

            # 获取输入文件夹名作为JSON文件名的一部分
            input_folder_name = os.path.basename(os.path.normpath(input_path))
            json_filename = f"{input_folder_name}_results.json"
            json_path = os.path.join(dir_save_path, json_filename)

            # 保存所有图像的检测结果到一个JSON文件
            try:
                with open(json_path, "w") as f:
                    json.dump(all_detections, f, indent=4)
            except Exception as e:
                print(f"Error: 保存JSON文件时出错: {e}", file=sys.stderr)
                sys.exit(4)

        else:
            print(f"Error: 输入路径 {input_path} 既不是有效的文件也不是有效的文件夹。", file=sys.stderr)
            sys.exit(1)

        sys.exit(0)

    finally:
        # 删除临时文件
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)


if __name__ == "__main__":
    main()