import torch
import time
from ovtrain.torch_backend.activations_functions.relu import OpenVINOReLU

def test_openvino_vs_torch_relu():
    relu_ov = OpenVINOReLU()
    relu_pt = torch.nn.ReLU()
    x = torch.randn(2, 3)

    # PyTorch ReLU
    start_pt = time.time()
    y_pt = relu_pt(x)
    elapsed_pt = time.time() - start_pt

    # OpenVINO ReLU
    start_ov = time.time()
    y_ov = relu_ov(x)
    elapsed_ov = time.time() - start_ov

    # Maximum absolute difference
    diff = (y_pt - y_ov).abs().max().item()

    print("Input:\n", x)
    print("PyTorch ReLU Output:\n", y_pt)
    print("OpenVINO ReLU Output:\n", y_ov)
    print(f"PyTorch execution time: {elapsed_pt:.6f} seconds")
    print(f"OpenVINO execution time: {elapsed_ov:.6f} seconds")
    print(f"Maximum absolute difference: {diff:.6e}")

if __name__ == "__main__":
    test_openvino_vs_torch_relu()