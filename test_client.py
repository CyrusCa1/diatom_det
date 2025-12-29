import requests
import base64
import os
import json
import time
import sys
import msvcrt
import concurrent.futures

# --- Configuration ---
SERVER_URL = 'http://127.0.0.1:5000/detection'
# IMAGE_DIR = 'E:\\AI\\diatom_det\\exclude\\data'
IMAGE_DIR = 'E:\\AI\\diatom_det\\exclude\\images'
LARGE_DIR = 'E:\\AI\\datasets\\diatom\\datasets\\test' 
OUTPUT_DIR_LOCAL_SAVE = 'E:\\AI\\diatom_det\\exclude\\test_output'

def image_to_base64(path):
    """Encodes an image file to a base64 string with a data URI prefix."""
    try:
        ext = path.split('.')[-1].lower()
        # if ext not in ['png', 'jpg', 'jpeg']:
        #     return None
        
        prefix = f'data:image/{ext};base64,'
        with open(path, 'rb') as f:
            encoded_string = base64.b64encode(f.read()).decode('utf-8')
        return prefix + encoded_string
    except Exception:
        return None

def test_base64_input():
    print("--- Running Test 1: Low-Scale Base64 Input ---")
    img_base64_list = []
    
    # 获取一张有效图片用于测试
    # valid_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.png'))]
    valid_files = [f for f in os.listdir(IMAGE_DIR)]
    for filename in valid_files: # 只取前3张
        filepath = os.path.join(IMAGE_DIR, filename)
        b64_string = image_to_base64(filepath)
        if b64_string:
            img_base64_list.append({
                "image_name": filename,
                "image_base64": b64_string
            })

    if not img_base64_list:
        print("No JPG or PNG images found in IMAGE_DIR.")
        return

    payload = {
        "request_id": f"req_py_test_base64_{int(time.time())}",
        "task_type": "detection",
        "scale": "low",
        "input": {
            "input_type": "imgbase64_list",
            "imgbase64_list": img_base64_list
        },
        "output_require": {
            "no_local_save": True,
            "return_images": True
        },
        "config": {"predict_config": {"conf": 0.4}}
    }

    try:
        response = requests.post(SERVER_URL, json=payload)
        response_data = response.json()
        print(f"  Server message: {response_data.get('message')}")
        response.raise_for_status()
        with open('response_base64.json', 'w', encoding='utf-8') as f:
            json.dump(response_data, f, indent=4, ensure_ascii=False)
        print(f"Request successful. Response saved to response_base64.json")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

def test_folder_input():
    print("\n--- Running Test 2: Low-Scale Folder Input & Local Save ---")
    if not os.path.exists(OUTPUT_DIR_LOCAL_SAVE):
        os.makedirs(OUTPUT_DIR_LOCAL_SAVE)

    payload = {
        "request_id": f"req_py_test_folder_{int(time.time())}",
        "task_type": "detection",
        "scale": "low",
        "input": {
            "input_type": "folder",
            "folder_path": IMAGE_DIR
        },
        "output_require": {
            "no_local_save": False,
            "save_folder": OUTPUT_DIR_LOCAL_SAVE,
            "return_images": True 
        },
        "config": {}
    }

    try:
        response = requests.post(SERVER_URL, json=payload)
        response_data = response.json()
        print(f"  Server message: {response_data.get('message')}")
        response.raise_for_status()
        with open('response_folder.json', 'w', encoding='utf-8') as f:
            json.dump(response_data, f, indent=4, ensure_ascii=False)
        print(f"Request successful. Response saved to response_folder.json")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

def test_high_scale_input():
    print("\n--- Running Test 3: High-Scale Classification (Folder Input) ---")
    payload = {
        "request_id": f"req_py_test_high_folder_{int(time.time())}",
        "task_type": "detection",
        "scale": "high",
        "input": {
            "input_type": "folder",
            "folder_path": IMAGE_DIR
        },
        "output_require": {
            "no_local_save": True,
            "return_images": False 
        },
        "config": {}
    }

    try:
        response = requests.post(SERVER_URL, json=payload)
        response_data = response.json()
        print(f"  Server message: {response_data.get('message')}")
        response.raise_for_status()
        with open('response_high_scale.json', 'w', encoding='utf-8') as f:
            json.dump(response_data, f, indent=4, ensure_ascii=False)
        print(f"Request successful. Response saved to response_high_scale.json")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

def test_high_scale_b64_input():
    print("\n--- Running Test 4: High-Scale Base64 Input ---")
    img_base64_list = []
    valid_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.png'))]
    for filename in valid_files[:3]:
        filepath = os.path.join(IMAGE_DIR, filename)
        b64_string = image_to_base64(filepath)
        if b64_string:
            img_base64_list.append({"image_name": filename, "image_base64": b64_string})

    if not img_base64_list:
        print("No images found.")
        return

    payload = {
        "request_id": f"req_py_test_high_b64_{int(time.time())}",
        "task_type": "detection",
        "scale": "high",
        "input": {
            "input_type": "imgbase64_list",
            "imgbase64_list": img_base64_list
        },
        "output_require": {
            "no_local_save": True,
            "return_images": False
        },
        "config": {}
    }

    try:
        response = requests.post(SERVER_URL, json=payload)
        response_data = response.json()
        print(f"  Server message: {response_data.get('message')}")
        response.raise_for_status()
        with open('response_high_scale_base64.json', 'w', encoding='utf-8') as f:
            json.dump(response_data, f, indent=4, ensure_ascii=False)
        print(f"Request successful.")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

def test_invalid_requests():
    print("\n--- Running Test 5: Invalid Requests ---")
    # ... (保持原有的 invalid requests 测试代码不变)
    test_cases = [
        {"name": "Missing 'scale'", "payload": {"request_id": "r1", "task_type": "detection", "input": {"input_type": "folder", "folder_path": IMAGE_DIR}, "output_require": {"no_local_save": True}}},
        {"name": "Invalid 'scale'", "payload": {"request_id": "r2", "task_type": "detection", "scale": "medium", "input": {"input_type": "folder", "folder_path": IMAGE_DIR}, "output_require": {"no_local_save": True}}},
        {"name": "Missing 'folder_path'", "payload": {"request_id": "r3", "task_type": "detection", "scale": "low", "input": {"input_type": "folder"}, "output_require": {"no_local_save": True}}},
        {"name": "Missing 'save_folder'", "payload": {"request_id": "r4", "task_type": "detection", "scale": "low", "input": {"input_type": "folder", "folder_path": IMAGE_DIR}, "output_require": {"no_local_save": False}}},
        {"name": "Empty 'imgbase64_list'", "payload": {"request_id": "r5", "task_type": "detection", "scale": "low", "input": {"input_type": "imgbase64_list", "imgbase64_list": []}, "output_require": {"no_local_save": True}}},
        {"name": "Malformed base64", "payload": {"request_id": "r6", "task_type": "detection", "scale": "low", "input": {"input_type": "imgbase64_list", "imgbase64_list": [{"image_name": "bad.jpg", "image_base64": "data:image/jpeg;base64,bad"}]}, "output_require": {"no_local_save": True}}},
        {"name": "File as Folder", "payload": {"request_id": "r7", "task_type": "detection", "scale": "low", "input": {"input_type": "folder", "folder_path": "dummy.txt"}, "output_require": {"no_local_save": True}}}
    ]

    with open("dummy.txt", "w") as f: f.write("x")

    passed = 0
    for i, case in enumerate(test_cases):
        print(f"  Sub-test {i+1}: {case['name']}...", end=" ")
        time.sleep(1)
        try:
            r = requests.post(SERVER_URL, json=case['payload'])
            if r.status_code >= 400:
                print(f"PASS ({r.status_code})")
                passed += 1
            else:
                print(f"FAIL ({r.status_code})")
        except Exception as e:
            print(f"ERROR: {e}")
    os.remove("dummy.txt")
    print(f"Invalid requests result: {passed}/{len(test_cases)}")

def test_large_folder_input():
    """
    Tests the server with a large folder input (using Block Processing).
    This simulates the batch processing mode.
    """
    print(f"\n--- Running Test 6: Large Folder Input (Block Processing) ---")
    print(f"  Target: {LARGE_DIR}")
    
    if not os.path.isdir(LARGE_DIR):
        print(f"  [Skip] LARGE_DIR not found: {LARGE_DIR}")
        return

    # 这里我们使用 no_local_save=False 但 no_response_save=True
    # 模拟“处理完保存在服务器本地，不要把巨大的结果JSON传回来”的场景
    large_output_dir = os.path.join(OUTPUT_DIR_LOCAL_SAVE, "large_batch_results")
    if not os.path.exists(large_output_dir):
        os.makedirs(large_output_dir)

    payload = {
        "request_id": f"req_py_test_large_{int(time.time())}",
        "task_type": "detection",
        "scale": "low",
        "input": {
            "input_type": "folder",
            "folder_path": LARGE_DIR
        },
        "output_require": {
            "no_local_save": False,
            "save_folder": large_output_dir,
            "return_images": False,  # 大批量通常不生成渲染图以节省时间
            "no_response_save": True # 关键参数：防止响应体爆炸
        },
        "config": {}
    }
    
    start_t = time.time()
    try:
        print("  Sending request, waiting for processing (this may take time)...")
        response = requests.post(SERVER_URL, json=payload)
        elapsed = time.time() - start_t
        
        response_data = response.json()
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Status Code: {response.status_code}")
        print(f"  Server message: {response_data.get('message')}")
        
        result_content = response_data.get('result', {})
        json_res_len = len(result_content.get('json_results', []))
        print(f"  Returned json_results count: {json_res_len} (Expected 0 if no_response_save=True)")
        
        if response.status_code == 200:
             print("  Test Passed: Large batch processed successfully.")
        else:
             print("  Test Failed.")

    except requests.exceptions.RequestException as e:
        print(f"  Request failed: {e}")

def test_batch_performance_comparison():
    """
    对比测试：使用默认 Batch (1) 与 大 Batch (8) 处理同一个大文件夹的耗时差异。
    注意：这需要服务端代码支持从 config 中读取 batch 参数并传给 model.predict。
    """
    # 创建所需的输出目录
    perf_b1_dir = os.path.join(OUTPUT_DIR_LOCAL_SAVE, "perf_b1")
    perf_b8_dir = os.path.join(OUTPUT_DIR_LOCAL_SAVE, "perf_b8")
    os.makedirs(perf_b1_dir, exist_ok=True)
    os.makedirs(perf_b8_dir, exist_ok=True)

    
    print(f"\n--- Running Test 8: Batch Size Performance Comparison ---")
    print(f"  Target: {LARGE_DIR}")
    
    if not os.path.isdir(LARGE_DIR):
        print(f"  [Skip] LARGE_DIR not found.")
        return

    # 定义两个 Payload，唯一的区别是 config 中的 batch 大小
    payload_batch_1 = {
        "request_id": f"req_perf_b1_{int(time.time())}",
        "task_type": "detection",
        "scale": "low",
        "input": {"input_type": "folder", "folder_path": LARGE_DIR},
        "output_require": {
            "no_local_save": False, # 正常保存结果
            "save_folder": os.path.join(OUTPUT_DIR_LOCAL_SAVE, "perf_b1"),
            "return_images": False, # 关闭图片返回以排除网络传输干扰，只测计算性能
            "no_response_save": True # 必须开启，防止大JSON卡顿
        },
        "config": {
            "predict_config": {
                "conf": 0.25, 
                "batch": 1  # 显式指定 Batch = 1
            }
        }
    }

    payload_batch_8 = {
        "request_id": f"req_perf_b8_{int(time.time())}",
        "task_type": "detection",
        "scale": "low",
        "input": {"input_type": "folder", "folder_path": LARGE_DIR},
        "output_require": {
            "no_local_save": False,
            "save_folder": os.path.join(OUTPUT_DIR_LOCAL_SAVE, "perf_b8"),
            "return_images": False,
            "no_response_save": True
        },
        "config": {
            "predict_config": {
                "conf": 0.25, 
                "batch": 8  # 显式指定 Batch = 8 (建议根据显存调整，如 4, 8, 16)
            }
        }
    }

    # 1. 测试 Batch = 1
    print("  [1/2] Testing Batch = 1 ...")
    t1_start = time.time()
    try:
        r1 = requests.post(SERVER_URL, json=payload_batch_1)
        r1.raise_for_status()
        t1_cost = time.time() - t1_start
        print(f"    -> Time cost: {t1_cost:.2f}s")
    except Exception as e:
        print(f"    -> Failed: {e}")
        return

    # 2. 测试 Batch = 8
    print("  [2/2] Testing Batch = 8 ...")
    t2_start = time.time()
    try:
        r2 = requests.post(SERVER_URL, json=payload_batch_8)
        r2.raise_for_status()
        t2_cost = time.time() - t2_start
        print(f"    -> Time cost: {t2_cost:.2f}s")
    except Exception as e:
        print(f"    -> Failed: {e}")
        return

    # 3. 结果分析
    print("-" * 40)
    print(f"  Batch 1 Total Time: {t1_cost:.2f}s")
    print(f"  Batch 8 Total Time: {t2_cost:.2f}s")
    
    if t2_cost < t1_cost:
        speedup = (t1_cost - t2_cost) / t1_cost * 100
        print(f"  Result: Batch 8 is {speedup:.1f}% faster!")
    else:
        print(f"  Result: Batch 8 did not improve speed (Diff: {t2_cost - t1_cost:.2f}s).")
        print("  Possible reasons: Server ignores 'batch' param, IO bottleneck, or small dataset.")

def single_request_task(idx, img_payload):
    """Helper for concurrent testing"""
    # 每个请求使用独立的ID
    payload = img_payload.copy()
    payload['request_id'] = f"req_concurrent_{idx}_{int(time.time())}"
    try:
        r = requests.post(SERVER_URL, json=payload, timeout=30)
        return r.status_code
    except Exception as e:
        return str(e)

def test_concurrent_requests():
    """
    Tests server concurrency by sending multiple base64 requests simultaneously.
    """
    print("\n--- Running Test 7: Concurrent Requests (Stress Test) ---")
    
    # 准备一个轻量级的 Payload (单张图片)
    valid_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.png'))]
    if not valid_files:
        print("  [Skip] No images for payload generation.")
        return
        
    b64_str = image_to_base64(os.path.join(IMAGE_DIR, valid_files[0]))
    base_payload = {
        "task_type": "detection",
        "scale": "low",
        "input": {
            "input_type": "imgbase64_list",
            "imgbase64_list": [{"image_name": "stress_test.jpg", "image_base64": b64_str}]
        },
        "output_require": {"no_local_save": True, "return_images": False},
        "config": {}
    }

    CONCURRENT_COUNT = 20
    print(f"  Launching {CONCURRENT_COUNT} simultaneous requests...")
    
    start_t = time.time()
    success_cnt = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(single_request_task, i, base_payload) for i in range(CONCURRENT_COUNT)]
        
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res == 200:
                success_cnt += 1
            else:
                print(f"    Request failed with: {res}")
                
    elapsed = time.time() - start_t
    print(f"  Finished in {elapsed:.2f}s. Success rate: {success_cnt}/{CONCURRENT_COUNT}")

if __name__ == '__main__':
    print(f"测试程序即将启动, 请确认配置:")
    print(f"  SERVER_URL: {SERVER_URL}")
    print(f"  IMAGE_DIR: {IMAGE_DIR}")
    print(f"  LARGE_DIR: {LARGE_DIR}")
    print("按任意键开始，'q' 退出")
    
    key = msvcrt.getch().decode('utf-8', errors='ignore')
    if key.lower() == 'q': exit(0)
    
    while True:
        test_base64_input()
        time.sleep(1)
        test_folder_input()
        time.sleep(1)
        test_high_scale_input()
        time.sleep(1)
        test_high_scale_b64_input()
        time.sleep(1)
        test_invalid_requests()
        time.sleep(1)
        test_large_folder_input()   # 新增
        time.sleep(1)
        test_batch_performance_comparison()
        time.sleep(1)
        test_concurrent_requests()  # 新增
        
        print("\n按 'q' 退出并关闭服务器，其他键重试...")
        key = msvcrt.getch().decode('utf-8', errors='ignore')
        if key.lower() == 'q':
            requests.post(f"{SERVER_URL.rsplit('/', 1)[0]}/shutdown")
            break