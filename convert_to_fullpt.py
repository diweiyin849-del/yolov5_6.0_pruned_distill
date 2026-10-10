import torch
from models.yolo import Model
from models.pruned_common import C3Pruned, SPPFPruned  # 如果用了剪枝模块

# 1. 加载 state_dict（FP32）
state_dict = torch.load('runs/train/exp2/weights/best.pt', map_location='cpu')

# 2. 构建模型结构（使用训练时的 YAML）
cfg = 'models/yolov5s_X.yaml'  # 或 'pruned_model.yaml'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model(cfg).to(device)

# 3. 加载权重
model.load_state_dict(state_dict)

# 4. 转为 FP16（半精度）
model.half()  # 将模型参数转为 FP16

# 5. 保存完整模型（包含结构和权重，FP16）
torch.save(model, '完整模型_fp16.pt')
print("✅ 完整 FP16 模型已保存")

