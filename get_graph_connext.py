import torch
import torch.nn as nn
import hashlib
from transformers import AutoTokenizer, AutoModelForCausalLM

def tensor_md5(tensor: torch.Tensor, sample_rate: int = 100) -> str:
    tensor = tensor.flatten()
    if tensor.numel() > sample_rate:
        tensor = tensor[::sample_rate]
    byte_data = tensor.detach().cpu().numpy().tobytes()
    return hashlib.md5(byte_data).hexdigest()

def is_activation_or_reshape(module: nn.Module) -> bool:
    return isinstance(module, (
        nn.ReLU, nn.Sigmoid, nn.Tanh, nn.LeakyReLU, nn.GELU,
        nn.Flatten, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d
    ))

def has_weight(module: nn.Module) -> bool:
    return hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor)

# ------------ 主函数 ------------
def analyze_module_connections(model: nn.Module, sample_input: torch.Tensor):
    md5_dict = {}

    def register_hook(name, module):
        def hook_fn(module, input, output):
            input_md5 = [tensor_md5(i) for i in input if isinstance(i, torch.Tensor)]
            output_md5 = tensor_md5(output) if isinstance(output, torch.Tensor) else None
            md5_dict[name] = {
                'module': module,
                'type': str(type(module)),
                'input_md5': input_md5,
                'output_md5': output_md5
            }
        module.register_forward_hook(hook_fn)

    # 遍历 model 注册 hook
    for name, module in model.named_modules():
        if has_weight(module) or is_activation_or_reshape(module):
            register_hook(name, module)

    # 前向传播一次，触发 hook
    with torch.no_grad():
        model(sample_input)

    # 建立连接图
    md5_to_module = {}
    for name, entry in md5_dict.items():
        out_md5 = entry['output_md5']
        if out_md5:
            md5_to_module[out_md5] = name

    connections = []
    for name, entry in md5_dict.items():
        for in_md5 in entry['input_md5']:
            src = md5_to_module.get(in_md5, None)
            if src:
                connections.append((src, name))

    return md5_dict, connections


# === 使用示例 ===
if __name__ == "__main__":
    model_name = "Qwen/Qwen2-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, attn_implementation = "eager")
    model.eval()
    prompt = "Hello, Qwen!"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    md5_dict, connections = analyze_module_connections(model, input_ids)
    # 打印连接关系
    print("连接关系:")
    for src, dst in connections:
        print(f"{src} --> {dst}")
