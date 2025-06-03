import torch
import torch.nn as nn
import numpy as np
from openvino.runtime import Core, opset8, Model

class OpenVINOReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = Core()
        # if Gpu is available, use it, otherwise use CPU
        available_devices = self.core.available_devices
        if "GPU" in available_devices:
            self.device = "GPU"
        else:
            self.device = "CPU"
        self.input_shape = None
        self.compiled_model = None
        self.infer_request = None
        self.output = None

    def _build_model(self, input_shape):
        param = opset8.parameter(input_shape, dtype=np.float32)
        relu_node = opset8.relu(param)
        model = Model([relu_node], [param], "ReLUModel")
        self.compiled_model = self.core.compile_model(model, self.device)
        self.infer_request = self.compiled_model.create_infer_request()
        self.output = self.compiled_model.output(0)

    def forward(self, x):
        x_np = x.detach().cpu().numpy().astype(np.float32)
        if self.input_shape != list(x_np.shape):
            self.input_shape = list(x_np.shape)
            self._build_model(self.input_shape)
        result = self.infer_request.infer({0: x_np})
        output = result[self.output]
        return torch.from_numpy(output).to(x.device)