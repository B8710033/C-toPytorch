import torch
import torch.onnx

# 假設是一個簡單的模型
class SimpleModel(torch.nn.Module):
    def forward(self, x):
        return x + 100

# 創建模型實例並轉換
model = SimpleModel()
dummy_input = torch.randn(1)
torch.onnx.export(model, dummy_input, "simple_model.onnx")