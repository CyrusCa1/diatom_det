# 硅藻检测服务器 (Diatom Detection Server)

一个基于 YOLOv8 和深度学习的硅藻图像检测与分类服务器，提供 HTTP API 接口，支持低倍目标检测（OBB）和高倍图像分类。

## 项目简介

本项目是一个 Flask 构建的 C/S 架构检测服务器，专门用于硅藻显微图像的自动化检测与分类。服务器支持：

- **低倍模型**：基于 YOLOv8-OBB 的旋转边界框目标检测
- **高倍模型**：基于深度学习的图像分类
- **灵活的输入方式**：支持文件夹批量处理和 Base64 编码的实时预览
- **高性能优化**：支持批处理、流式推理、JSONL 缓存等优化策略
- **完善的配置系统**：通过 JSON 配置文件灵活控制服务器行为

## 主要特性

- ✅ 支持多种图像格式（JPG, PNG, TIF, BMP, WEBP 等）
- ✅ 旋转边界框（OBB）目标检测
- ✅ 批量处理大规模图像数据集
- ✅ Base64 编码支持实时预览
- ✅ 自定义类别映射和检测参数
- ✅ 中文字体支持的可视化结果
- ✅ 自动清理临时文件
- ✅ 详细的日志记录
- ✅ 优雅的进度条显示

## 环境要求

### Python 版本
- Python 3.10.14

### 主要依赖

```
torch==2.7.0+cu128
torchvision==0.22.0+cu128
ultralytics==8.2.75
flask==3.1.2
waitress==3.0.2
opencv-python==4.10.0.84
pillow==10.4.0
numpy==1.26.4
scikit-learn==1.5.1
tqdm==4.66.5
imageio==2.35.1
```

完整的环境依赖请参考 `yolov8_env_backup.txt`。

### 硬件要求

- **推荐**：NVIDIA GPU（支持 CUDA 12.1+）
- **最低**：CPU（性能会显著降低）
- **内存**：建议 8GB+ RAM
- **显存**：建议 4GB+ VRAM（使用 GPU 时）

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/yourusername/diatom-detection-server.git
cd diatom-detection-server
```

### 2. 安装依赖

使用 Conda 创建环境（推荐）：

```bash
conda create -n yolov8 python=3.10.14
conda activate yolov8
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install ultralytics==8.2.75 flask waitress opencv-python pillow numpy scikit-learn tqdm imageio
```

或使用 pip：

```bash
pip install -r requirements.txt
```

### 3. 准备模型文件

**注意：本项目不提供预训练权重文件。** 你需要自行训练或获取以下模型文件，并放置在 `logs/` 目录下：

- `logs/low_scale.pt` - YOLOv8-OBB 低倍检测模型
- `logs/high_scale.pth` - 高倍分类模型
- `logs/config.json` - 配置文件（已提供模板）
- `logs/SimHei.ttf` - 中文字体文件（可选，用于可视化）

### 4. 配置服务器

编辑 `logs/config.json` 文件，设置类别映射和服务器参数：

```json
{
    "CLASS_MAPPING": {
        "类别1": 0,
        "类别2": 1,
        ...
    },
    "predict_config": {
        "iou": 0.3,
        "conf": 0.21,
        "batch": 1
    },
    "server_config": {
        "filt_and_delete_invalid_content": true,
        "check_and_delete_residual_tmp": true,
        "valid_img_ext": [".jpg", ".png", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"],
        "dev_log_mode": false,
        "save_every_json_result": false,
        "default_font_path": "logs/SimHei.ttf"
    }
}
```

### 5. 运行服务器

```bash
python detection_server.py --port 5000
```

服务器将在 `http://127.0.0.1:5000` 启动。

### 6. 测试服务器

使用提供的测试客户端：

```bash
python test_client.py
```

## 使用文档

详细的 API 文档和使用说明请参考 **[wiki.md](wiki.md)**，包括：

- 完整的 API 参考
- 请求/响应格式说明
- `json_results` 字段详解（低倍/高倍模型）
- 使用示例
- 配置参数说明

## 打包为可执行文件

使用 PyInstaller 将服务器打包为独立的可执行文件：

```bash
pyinstaller --name detection_server --onedir --add-data "logs;logs" --add-binary "D:\develop\anaconda\envs\yolov8\Library\bin;." --noupx detection_server.py
```

**注意事项：**
- 请根据你的实际环境修改 `--add-binary` 路径
- 打包后的可执行文件位于 `dist/detection_server/` 目录
- 确保 `logs/` 目录及其内容（模型文件、配置文件）与可执行文件在同一目录

## 项目结构

```
diatom-detection-server/
├── detection_server.py          # 主服务器程序
├── test_client.py                # 测试客户端
├── wiki.md                       # 详细使用文档
├── README.md                     # 项目说明
├── yolov8_env_backup.txt        # 完整环境依赖
├── logs/                         # 模型和配置目录
│   ├── config.json              # 服务器配置文件
│   ├── low_scale.pt             # 低倍检测模型（需自行提供）
│   ├── high_scale.pth           # 高倍分类模型（需自行提供）
│   └── SimHei.ttf               # 中文字体文件（可选）
├── high_scale/                   # 高倍分类模型代码
│   ├── classification.py
│   ├── classification_method.py
│   └── ...
└── runs/                         # YOLO 运行输出目录（自动生成）
```

## API 端点

### 检测端点

- **URL**: `POST /detection`
- **功能**: 执行图像检测或分类
- **请求格式**: JSON
- **响应格式**: JSON

### 关闭端点

- **URL**: `POST /shutdown`
- **功能**: 安全关闭服务器
- **限制**: 仅接受来自 `127.0.0.1` 的请求

详细的 API 文档请参考 [wiki.md](wiki.md)。

## 配置说明

### 服务器配置 (`server_config`)

- `filt_and_delete_invalid_content`: 是否过滤并删除非白名单文件（提升性能）
- `check_and_delete_residual_tmp`: 启动时清理残留临时文件
- `valid_img_ext`: 支持的图像格式白名单
- `dev_log_mode`: 开发者日志模式（建议生产环境设为 `false`）
- `save_every_json_result`: 是否保存每张图片的单独 JSON 结果
- `default_font_path`: 可视化使用的字体路径

### 预测配置 (`predict_config`)

- `conf`: 置信度阈值（0.0-1.0）
- `iou`: IoU 阈值（0.0-1.0）
- `batch`: 批处理大小（低倍模型支持）

## 性能优化

本项目针对大规模图像处理进行了多项优化：

1. **流式推理**：使用 YOLO 的 `stream=True` 模式，降低内存占用
2. **JSONL 缓存**：大批量数据使用 JSONL 格式缓存，避免内存溢出
3. **批处理支持**：低倍模型支持 batch 参数，提升 GPU 利用率
4. **智能过滤**：直接删除非法文件而非过滤，避免 DataLoader 初始化开销
5. **进度条显示**：非开发模式下显示美观的 tqdm 进度条

## 注意事项

⚠️ **重要提示**：

1. **模型文件不包含在本仓库中**，需要自行训练或获取
2. **`exclude/` 目录内容不公开**，包含开发过程中的测试文件和旧版本
3. 启用 `filt_and_delete_invalid_content` 时，服务器会**删除**非白名单文件，请确保数据安全
4. 大规模输入时，建议设置 `no_response_save=true` 和 `return_images=false`
5. 高倍模型建议使用 `no_local_save=true`，通过响应体获取结果

## 日志记录

服务器会自动在可执行文件目录创建 `detection_server.log` 日志文件：

- 记录所有启动信息、请求、错误和异常
- 日志文件大小上限 5MB，超过后自动备份为 `.old`
- 后台运行时，日志文件是排查问题的唯一途径

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

如有问题或建议，请通过 GitHub Issues 联系。

---

**注意**：本项目用于学术研究和技术交流，不提供商业支持和预训练模型。
