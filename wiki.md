# 硅藻检测服务器文档

## 1. 概述

本文档为前端开发人员提供与硅藻检测服务器交互的说明。该服务器提供了一个简单的HTTP API，用于对图像进行目标检测或分类。

该服务器采用Flask构建，以客户端-服务器（C/S）模式运行。前端（例如Qt应用程序）向此服务器发送检测请求，并以结构化的JSON格式接收结果。

## 2. 开始使用

### 运行服务器

服务器是一个Python脚本，需要在后台运行以接收请求。它应该在主应用程序需要时启动。

**运行命令:**

```bash
# 程序将打包为可执行文件, 运行命令如下:
# 端口可根据需要更改。
detECT_SERVER.EXE --port 5000
```

启动后，服务器将加载必要的模型，并开始监听POST请求。端点地址为：

`http://127.0.0.1:5000/detection`

其中, 5000为服务器的默认端口, 前端程序应自行判断客户机上的可用端口, 并通过`--port`参数指定端口号启动服务器, 防止端口冲突。

### 服务器初始化参数

服务器在启动时, 会读取 `detection_server.exe`文件同级目录下的 `logs`文件夹, 该文件夹下有一个名为 `config.json`的文件, 该文件包含服务器的配置信息, 默认配置文件如下:

```json
{
    "CLASS_MAPPING": {...},
    "predict_config": {
        "iou": 0.3,
        "conf": 0.21,
        "batch": 1,
        "imgsz": 1024,
        "half": true,
        "rect": true,
        "augment": false,
        "max_det": 300
    },
    "server_config": {
        "filt_and_delete_invalid_content": true,
        "check_and_delete_residual_tmp": true,
        "valid_img_ext": [".jpg", ".png", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"],
        "dev_log_mode": false,
        "save_every_json_result": false,
        "default_font_path": "logs/SimHei.ttf",
        "use_high_fidelity_model": false,
        "high_fidelity_model_classes": ["沟链藻", "圆筛藻", "舟形藻", "小环藻", "双眉藻", "异极藻", "菱形藻"]
    }
}
```

其中, 每个字段含义如下:

- `CLASS_MAPPING`: 模型所使用的类别映射, **此参数允许用户通过请求体覆盖.** 。
- `predict_config`: 模型预测的配置参数, **此参数允许用户通过请求体覆盖.** 当请求体中没有此字段, 则使用此文件中定义的默认值。
- `server_config`: 服务器的配置参数。
  - `high_scale_model_type`: 高倍分类模型的骨干网络类型，可选值：`mobilenetv2`、`resnet18`、`resnet34`、`resnet50`、`resnet101`、`resnet152`、`vgg11`、`vgg13`、`vgg16`、`vgg11_bn`、`vgg13_bn`、`vgg16_bn`、`vit_b_16`、`swin_transformer_tiny`、`swin_transformer_small`、`swin_transformer_base`。默认为 `resnet50`。**注意：高倍模型会自动使用与低倍模型相同的设备（GPU/CPU）。**
  - `high_scale_input_shape`: 高倍模型所使用Resize大小, 默认为[512, 512].
  - `check_and_delete_residual_tmp`: 是否在服务器每次启动时, 检查是否有残留的缓存文件或文件夹, 并删除这些文件或文件夹。
  - `filt_and_delete_invalid_content`: 当请求方式是文件夹时, 是否对所请求的文件夹按照 `valid_img_ext`白名单进行过滤, 并删除白名单外的文件, 服务器使用直接删除而非过滤非法内容的方式, 是为了极致地加快检测速度, 避免dataloader初始化输入列表带来的大量时间开销。**所以注意: 强烈建议开启此功能, 避免模型对文件夹中非图片文件进行检测, 并且服务器将会删除非白名单的文件/目录, 请务将无关的重要文件保存在请求检测的目录中, 正常请求过程中也不应该请求非法内容。总之当开启此功能时, 前端程序需要为数据安全负责。**
  - `valid_img_ext`: 白名单, 即允许处理的图像文件扩展名。
  - `dev_log_mode`: 是否启用开发者日志模式, 关闭后服务器将使用更加美观的控制台输出方式, 建议设为 `false`。
  - `save_every_json_result`: 是否保存每个请求的JSON结果, 建议设为为 `false`, 因为模型的检测速度甚至与磁盘io速度相当, 如果大规模输入场景下开启此功能将严重拖慢服务器运行速度。
  - `default_font_path`: 默认的字体文件路径, 默认为 `logs/SimHei.ttf`。用户可以自行使用其他的字体文件路径, 服务器会自行检查该文件路径是否正确且是否支持中文显示, 如果不支持, 服务器会自动查找本机字体作为默认字体。
  - `use_high_fidelity_model`: 这是一个临时的测试用逻辑, 当此字段为true时, 低倍模型将使用高精度模型的类别进行映射, 但是检测时使用的权重文件必须由于为高精度模型, 作为临时的行为, 服务器不会进行模型权重文件是否为正确的高精度模型的检查, 需要测试人员自行保证。
  - `high_fidelity_model_classes`: 高精度模型所使用的类别映射。

注意: 上述配置文件仅在服务器初始化时加载, 如果需要更改配置文件, 请修改该文件并重新启动服务器。

### 日志记录

为了便于调试和追踪服务器状态，服务器在启动时会自动在其可执行文件（`.exe`）所在的目录创建一个名为 `detection_server.log` 的日志文件。

- **所有** 启动信息、模型加载状态、接收到的请求、以及任何错误或异常都会被记录到这个文件中。
- 当服务器以无窗口模式在后台运行时，如果遇到任何启动失败或运行错误，此日志文件是定位问题的**唯一途径**。
- 日志文件大小上限为5MB，超过后会自动备份为 `detection_server.log.old`，以防无限增大。

## 3. API 参考

### 3.1. 检测端点 (`/detection`)

#### 请求体

所有请求都必须是包含JSON正文的 `POST` 请求。其结构如下：

```json
{
  "request_id": "req_20241104_123456",
  "task_type": "detection",
  "scale": "low",
  "input": { ... },
  "output_require": { ... },
  "config": { ... }
}
```

**字段详解:**

- `request_id` (字符串, **必需**): 由客户端为每个请求生成的唯一ID。这对于跟踪和匹配响应至关重要，尤其是在异步工作流中。
- `task_type` (字符串, **必需**): 任务类型。当前仅支持 `"detection"`。
- `scale` (字符串, **必需**): 要使用的模型。

  - `"low"`: 用于低倍率OBB（旋转框）目标检测。
  - `"high"`: 用于高倍率图像分类。
- `input` (对象, **必需**): 定义要处理的图像。

  - `input_type` (字符串): 可选值如下：
    - `"imgbase64_list"`: 提供一个Base64编码的图像列表。适用于少量图像或“在线预览”功能。**注意：** base64编码后图片将比原始图片大小增加30%左右, 为避免高内存占用和请求大小限制，请勿对大量图像使用此方法, 另外, 新版服务器程序已支持在白名单格式内的所有图像使用base64编码方式的输入。
    - `"folder"`: 提供一个本地文件夹路径。服务器将处理此文件夹中的所有图像。这是进行批量处理的推荐方法。
  - `folder_path` (字符串): 当 `input_type` 为 `"folder"` 时必需。必须是包含图像的目录的绝对路径。
  - `imgbase64_list` (数组): 当 `input_type` 为 `"imgbase64_list"` 时必需。一个对象数组，每个对象包含：
    - `image_name` (字符串): 图像文件的原始名称。
    - `image_base64` (字符串): Base64编码的图像字符串，包含数据URI前缀 (例如, `data:image/png;base64,xxxx...`)。
- `output_require` (对象, **必需**): 指定如何交付结果。

  - `no_local_save` (布尔值):
    - `false` (默认): 服务器将结果JSON和（可选的）结果图像保存到 `save_folder`。响应体将包含检测数据，但不包含渲染后的图像数据。
    - `true`: 服务器**不会**在本地保存任何文件。所有结果，包括渲染后的图像（如果请求），都将作为Base64字符串在响应体中返回。这是“预览”模式的理想选择。另外, **调用高倍模型(`scale`为 `high`)时, 请将此设置为 `true`, 因为高倍模型的检测结果只有一个文件, 并且可以会在响应体中返回给客户端, 完全没有必要保存文件**。如果调用高倍模型同时又将 `no_local_save`设置为 `false`, 服务器将把json结果文件保存为一个服务端生成的缓存文件名, 这将不方便使用, 所以再次强调调用高倍模型时, 请通过响应体获取检测结果。
  - `no_response_save` (布尔值):
    - `false` (默认): 服务器将通过响应体的 `json_results`字段返回检测结果。
    - `true`: 服务器不再将检测结果通过响应体的 `json_results`字段返回, 在某些大规模输入场景时, 前端程序的请求请确保使用 `no_response_save`为 `true`, 否则会由于图片过多导致响应体过大而出错。请确保 `no_local_save`与 `no_local_save`不同时为 `true`时, 否则服务器将抛出错误
  - `save_folder` (字符串): 当 `no_local_save` 为 `false` 时必需。用于保存结果文件的目录的绝对路径。
  - `return_images` (布尔值):
    - `true`: 服务器生成带有检测框的结果图像, 服务器会始终返回jpg格式图像。
    - `false`: 服务器不生成结果图像（相当于旧脚本的 `--no-jpgs` 标志）。
- `config` (对象, 可选): 用于覆盖服务器的默认检测参数。如果为空 (`{}`), 服务器将使用其默认的 `config.json`。

  - 示例: `{"predict_config": {"conf": 0.5, "iou": 0.6, "batch": 1}}` 将该请求的置信度阈值设置为0.5，IoU阈值设置为0.6, batch size设置为1。**注意新版服务器程序支持对低倍模型使用batch参数, 可以加快某些大规模输入场景的检测速度。**

#### 响应体

服务器以一个JSON对象作为响应，其中包含检测的状态和结果。

```json
{
  "request_id": "req_20241104_123456",
  "status": "success",
  "code": 200,
  "message": "detection success",
  "result": { ... }
}
```

**字段详解:**

- `request_id` (字符串): 来自原始请求的唯一ID，允许客户端匹配响应。
- `status` (字符串): 请求的结果。`"success"` 或 `"error"`。
- `code` (整数): 类似HTTP的状态码 (例如, `200` 表示成功, `400` 表示错误请求, `500` 表示服务器错误)。
- `message` (字符串): 人类可读的状态消息。在出错时，这里会包含错误原因。
- `result` (对象): 包含实际的检测结果。
  - `task_time_ms` (整数): 请求的总处理时间（毫秒）。
  - `json_results` (数组): 一个JSON对象列表，每个对象代表一张图像的检测数据。**该字段格式为了兼容旧项目而保留，低倍和高倍模型的格式不同，详见下文。**
  - `rendered_images` (数组): **仅当** `output_require.no_local_save` 为 `true` 且 `output_require.return_images` 为 `true` 时，此数组才会被填充。它包含带有以下内容的对象：
    - `image_name` (字符串): 原始图像名称。
    - `image_base64` (字符串): 绘制了检测结果的Base64编码结果图像。
    - `format` (字符串): Base64字符串的图像格式 (始终为 `"jpg"`)。

#### `json_results` 字段详解

`json_results` 是一个数组，包含每张图像的检测结果。该字段的格式严格遵守旧版项目的规定，以保持向后兼容性。**低倍模型和高倍模型的格式不同。**

##### 低倍模型 (`scale: "low"`) 的 `json_results` 格式

低倍模型用于目标检测（OBB - Oriented Bounding Box，旋转边界框），每个图像的检测结果包含以下字段：

```json
{
  "source": "image001.png",
  "resultpic": "image001_result.jpg",
  "diatoms": [
    {
      "type": "圆筛藻-海链藻(齐)",
      "confidence": 0.8523,
      "location": {
        "x": 512.5,
        "y": 384.2,
        "w": 120.3,
        "h": 85.6,
        "angle": 45.2,
        "x1": 450.2,
        "y1": 320.5,
        "x2": 574.8,
        "y2": 320.5,
        "x3": 574.8,
        "y3": 447.9,
        "x4": 450.2,
        "y4": 447.9
      }
    }
  ]
}
```

**字段说明：**

- `source` (字符串): 原始图像的文件名（包含扩展名）。
- `resultpic` (字符串): 可视化结果图像的文件名。如果 `return_images` 为 `false`，此字段为空字符串 `""`。
- `diatoms` (数组): 检测到的所有硅藻对象列表。每个对象包含：
  - `type` (字符串): 硅藻的类别名称（使用 `CLASS_MAPPING` 中定义的名称）。
  - `confidence` (浮点数): 检测置信度，范围 0.0 到 1.0。
  - `location` (对象): 旋转边界框的位置信息，包含以下字段：
    - `x` (浮点数): 边界框中心点的 X 坐标（像素）。
    - `y` (浮点数): 边界框中心点的 Y 坐标（像素）。
    - `w` (浮点数): 边界框的宽度（像素）。
    - `h` (浮点数): 边界框的高度（像素）。
    - `angle` (浮点数): 边界框的旋转角度（度数）。
    - `x1`, `y1` (浮点数): 旋转边界框第一个顶点的坐标。
    - `x2`, `y2` (浮点数): 旋转边界框第二个顶点的坐标。
    - `x3`, `y3` (浮点数): 旋转边界框第三个顶点的坐标。
    - `x4`, `y4` (浮点数): 旋转边界框第四个顶点的坐标。

**注意：** 四个顶点 `(x1,y1)`, `(x2,y2)`, `(x3,y3)`, `(x4,y4)` 按顺时针或逆时针顺序排列，可用于精确绘制旋转的边界框。

##### 高倍模型 (`scale: "high"`) 的 `json_results` 格式

高倍模型用于图像分类，每个图像的分类结果包含以下字段：

```json
{
  "source": "sample_high.tif",
  "type": "冠盘藻",
  "confidence": 0.9521
}
```

**字段说明：**

- `source` (字符串): 原始图像的文件名（包含扩展名）。**注意：即使图像在处理过程中被转换为 JPG 格式，此字段仍会保留原始文件名和扩展名。**
- `type` (字符串): 分类结果的类别名称（使用 `CLASS_MAPPING` 中定义的名称）。
- `confidence` (浮点数): 分类置信度，范围 0.0 到 1.0。

**重要提示：** 高倍模型不进行目标检测，因此没有 `diatoms` 数组和位置信息，只有整张图像的分类结果。

### 3.2. 关闭服务器端点 (`/shutdown`)

为了在主应用程序退出时能够安全地关闭后台运行的服务器进程，服务器提供了一个专用的关闭端点。

**端点地址:** `http://127.0.0.1:5000/shutdown`

**方法:** `POST`

**请求体:** 无需任何请求体。

**重要:**

- 为安全起见，此端点**仅接受来自本机 (`127.0.0.1`)** 的请求。任何来自其他IP地址的请求都将被拒绝。
- 客户端（例如Qt程序）应在其自身的退出逻辑中，向此端点发送一个POST请求，以确保服务器进程能够被正确终止。

**成功响应示例:**

```json
{
  "status": "success",
  "message": "Server is shutting down..."
}
```

收到此响应后，服务器进程将在大约1秒后完全退出。

## 4. 使用示例

### 示例 1: 文件夹输入与本地保存 (低倍)

这是处理一批图像并将结果保存到磁盘的标准模式。

**请求:**

```json
{
  "request_id": "req_batch_1",
  "task_type": "detection",
  "scale": "low",
  "input": {
    "input_type": "folder",
    "folder_path": "C:\\Users\\Admin\\Desktop\\DiatomImages"
  },
  "output_require": {
    "no_local_save": false,
    "save_folder": "C:\\Users\\Admin\\Desktop\\DiatomResults",
    "return_images": true
  },
  "config": {}
}
```

### 示例 2: 用于实时预览的Base64输入 (低倍)

此模式适用于实时预览功能，结果会立即显示而无需写入磁盘。

**请求:**

```json
{
  "request_id": "req_preview_2",
  "task_type": "detection",
  "scale": "low",
  "input": {
    "input_type": "imgbase64_list",
    "imgbase64_list": [
      {
        "image_name": "preview.png",
        "image_base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg=="
      }
    ]
  },
  "output_require": {
    "no_local_save": true,
    "return_images": true
  },
  "config": {
    "predict_config": {
        "conf": 0.25
    }
  }
}
```

### 示例 3: 高倍率分类请求

此模式用于高倍率图像分类。

**请求:**

```json
{
  "request_id": "req_high_scale_1",
  "task_type": "detection",
  "scale": "high",
  "input": {
    "input_type": "folder",
    "folder_path": "C:\\Users\\Admin\\Desktop\\HighMagImages"
  },
  "output_require": {
    "no_local_save": true,
    "return_images": false
  },
  "config": {}
}
```

### 示例 4: 通过文件夹方式请求低倍模型进行大规模输入的检测

大规模输入必须设置`no_response_save`为`true`且`return_images`为`false`, 可选设置`batch`略微加快检测速度。

**请求:**

```json
{
  "request_id": "req_batch_1",
  "task_type": "detection",
  "scale": "low",
  "input": {
    "input_type": "folder",
    "folder_path": "C:\\Users\\Admin\\Desktop\\DiatomImages"
  },
  "output_require": {
    "no_response_save": true,
    "save_folder": "C:\\Users\\Admin\\Desktop\\DiatomResults",
    "return_images": false
  },
  "config": {
    "predict_config": {
        "batch": 4
      }
  }
}
```