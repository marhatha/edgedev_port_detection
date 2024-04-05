import torch
import torch.onnx

def export_onnx(model, output_size):
    dummy_input = torch.randn(1, 3, 224, 224)
    onnx_model_path = "/.models/model_v1.onnx"
    torch.onnx.export(model, dummy_input, onnx_model_path, opset_version=11, input_names=['input'], output_names=['output'], dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}})
    print(f"Model has been converted to ONNX format and saved to {onnx_model_path}")
