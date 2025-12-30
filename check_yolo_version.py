from ultralytics import YOLO

# 加载你的权重文件（支持.pt格式，若为onnx等格式需调整加载方式）
model = YOLO("logs\low_scale.pt")  # 替换为你的权重文件路径

# 打印模型详细摘要（包含参数量、计算量、网络结构）
model.info()

# 可选：若想快速查看关键参数，也可打印模型配置
print("模型配置信息：", model.model.yaml)