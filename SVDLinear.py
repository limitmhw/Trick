# -*- coding: utf-8 -*-

import copy
import torch
import torch.linalg
import torch.nn as nn

__all__ = ["SVDLinear"]

class SVDLinear(nn.Module):
    def __init__(
        self, linear: nn.Linear, rank: int):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        assert(rank > 0)
        self.rank = rank
        self.a, self.b, self.r = nn.Linear(self.in_features, rank, bias=False), nn.Linear(rank, self.out_features, bias=False), nn.Linear(self.in_features, self.out_features, bias=False)
        self.reset_parameters(linear.weight)

    @torch.no_grad()
    def reset_parameters(self, weight: torch.Tensor) -> None:
        assert weight.ndim == 2, "LinearLoRAHook only supports 2D input tensor"
        device, dtype = weight.device, weight.dtype
        self.to(device=device, dtype=dtype)
        u, s, vh = torch.linalg.svd(weight.double())
        us = u[:, : self.rank] * s[: self.rank]
        vh = vh[: self.rank:]
        w_k = us @ vh
        w_r = weight - w_k
        assert not us.isnan().any(), "NaN in U * S"
        assert not vh.isnan().any(), "NaN in V^T"
        assert not us.isinf().any(), "Inf in U * S"
        assert not vh.isinf().any(), "Inf in V^T"
        self.a.weight.data.copy_(vh.to(dtype))
        self.b.weight.data.copy_(us.to(dtype))
        self.r.weight.data.copy_(w_r.to(dtype))


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.b(self.a(input)) + self.r(input)


if __name__ == "__main__":
    class SimpleNet(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
            self.fc2 = nn.Linear(hidden_size, output_size, bias=False)

        def forward(self, x):
            x = self.fc1(x) 
            x = self.fc2(x)
            return x

    input_size = 10 
    hidden_size = 15 
    output_size = 20 

    model = SimpleNet(input_size, hidden_size, output_size)
    print(model)
    dummy_input = torch.randn(1, input_size)
    dummy_input1 = copy.deepcopy(dummy_input)
    dummy_input2 = copy.deepcopy(dummy_input)
    output1 = model(dummy_input1)

    for name, value in model.named_modules():
        if isinstance(value, nn.Linear):
            setattr(model, name, SVDLinear(value, 10))

    output2 = model(dummy_input2)
    loss = torch.norm(output1 - output2, p='fro')
    print("loss: ", loss)
    assert(torch.allclose(output1, output2))

