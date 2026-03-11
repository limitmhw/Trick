import torch
import numpy as np
from functools import partial

import re
def save_calib_result(model):
    calib_info = {}
    for name, module in model.named_modules():
        calib_info[name] = {}
        if hasattr(module, "module_i_amax"):
            calib_info[name]["input"] = module.module_i_amax
        if hasattr(module, "module_w_amax"):
            calib_info[name]["weight"] = module.module_w_amax 
        if hasattr(module, "smoothquant_scale"):
            calib_info[name]["smoothquant_scale"] = module.smoothquant_scale
            if hasattr(module, "alpha"):
                calib_info[name]["alpha"] = module.alpha
            else:
                calib_info[name]["alpha"] = 1.0
    torch.save(calib_info, model.__class__.__name__ + "__calib.pt")
    

global_step = 0
flag_name = ""
def special_calib(module, tensor):
    
    global flag_name
    global global_step

    # make a global tag
    if flag_name == "":
        flag_name = module.module_name
    elif flag_name == module.module_name:
        global_step = global_step + 1

    if not isinstance(module, torch.nn.Conv2d):
        return False

    # step filter
    axis = None
    if global_step % 20 < 8:
        local_amax = collect_min_max(tensor, axis)
        if hasattr(module, "module_i_amax"):
            setattr(module, "module_i_amax", torch.minimum(module.module_i_amax, local_amax))
        else:
            setattr(module, "module_i_amax", local_amax)
    return True


def reduce_amax(input, axis=None, keepdims=True):
    with torch.no_grad():
        if axis is None:
            max_val = torch.max(input)
            min_val = torch.min(input)
            output = torch.maximum(torch.abs(max_val), torch.abs(min_val))
        else:
            if isinstance(axis, int):
                axis = (axis,)
            max_val = torch.amax(input, dim=axis, keepdim=keepdims)
            min_val = torch.amin(input, dim=axis, keepdim=keepdims)
            output = torch.maximum(torch.abs(max_val), torch.abs(min_val))
            if output.numel() == 1:
                output.squeeze_()
        return output
    
def collect_min_max(tensor, axis):
    axis = axis if isinstance(axis, (list, tuple)) else [axis]
    axis = [tensor.dim() + i if isinstance(i, int) and i < 0 else i for i in axis]
    reduce_axis = []
    for i in range(tensor.dim()):
        if i not in axis:
            reduce_axis.append(i)
    return reduce_amax(tensor, axis=reduce_axis).detach()

def calib_hook_func(module, args, kwargs):
    tensor = args[0].clone()
    setattr(args[0], "module_name", module.module_name)
    
    if isinstance(module, torch.nn.Linear):
        axis = -1

    if isinstance(module, torch.nn.Conv2d):
        axis = None

    if not hasattr(module, "module_w_amax"):
        setattr(module, "module_w_amax", collect_min_max(module.weight, 0))

    if special_calib(module, tensor):
        return
    local_amax = collect_min_max(tensor, axis)
    if hasattr(module, "module_i_amax"):
        setattr(module, "module_i_amax", torch.max(module.module_i_amax, local_amax))
    else:
        setattr(module, "module_i_amax", local_amax)
    setattr(args[0], "module_i_amax", module.module_i_amax)


def calib_regist(model):
    for name, module in model.named_modules():
        setattr(module, "module_name", name)
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            module.register_forward_pre_hook(calib_hook_func, with_kwargs=True)
    
def smooth_fcs(module, alpha=0.8):
    # It is important to keep scaling math in fp32 to be numerically safe
    act_amax = module.module_i_amax.float()
    act_device = act_amax.device
    # If model is split across devices, this tensor may be on wrong one
    act_amax = act_amax.to(module.weight.device)
    weight_scale = module.weight.abs().max(dim=0, keepdim=True)[0]
    scale_a = (weight_scale.pow(1 - alpha) / act_amax.pow(alpha)).squeeze()
    # Some channel could have 0 amax which causes scale_a to overflow. Explicitly mask them out here
    epsilon = 1.0 / (1 << 31)
    if act_amax.min() <= epsilon:
        zero_mask = act_amax <= epsilon
        zero_mask = zero_mask.squeeze()
        scale_a[zero_mask] = 1
    setattr(module, "smoothquant_scale", scale_a.to(module.weight.dtype).to(module.weight.device))

    
@torch.no_grad()
def do_smooth_linear(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if hasattr(module, "alpha"):
                smooth_fcs(module, module.alpha)
            else:
                smooth_fcs(module, 1.0)

calib_info = None
weight_only = False
skip_pattern = None

def filter_func(name):
    if skip_pattern is not None:
        return re.compile(skip_pattern).match(name) is not None
    pattern = re.compile(
        r".*(time_emb_proj|time_embedding|conv_in|conv_out|conv_shortcut|add_embedding).*"
    )
    return pattern.match(name) is not None

def quant_conv(module, *args, **kwargs):
    global calib_info
    quant_min = -128
    quant_max = 128
    input_data = args[0]
    if not weight_only:
        input_scale =  (calib_info[module.module_name]["input"] /  quant_max).to(torch.float32)
        zero_point = torch.zeros(input_scale.shape).to(torch.int32).to(input_scale.device)
        input_data = torch.fake_quantize_per_tensor_affine(input_data, input_scale, zero_point, quant_min, quant_max)
    if not hasattr(module, "org_weight"):
        weight_scale = (calib_info[module.module_name]["weight"] /  quant_max).to(torch.float32)
        zero_point = torch.zeros(weight_scale.shape).to(torch.int32).to(weight_scale.device)
        quant_weight = torch.fake_quantize_per_channel_affine(module.weight, weight_scale.view(-1), zero_point.view(-1), 0, quant_min, quant_max)
        setattr(module, "org_weight", module.weight)
        module.weight = torch.nn.Parameter(quant_weight) 
    args = list(args)
    args[0] = input_data
    return module.org_forward(*args, **kwargs)

def quant_linear(module, *args, **kwargs):
    global calib_info
    quant_min = -128
    quant_max = 128
    input_data = args[0]

    if not weight_only:
        if "smoothquant_scale" in  calib_info[module.module_name].keys():
            smoothquant_amax = torch.tensor(
                (calib_info[module.module_name]["input"] * calib_info[module.module_name]["smoothquant_scale"]).max().item(),
                dtype=module.weight.dtype,
                device=module.weight.device,
            )
            input_scale =  (smoothquant_amax / quant_max).to(torch.float32)
            zero_point = torch.zeros(input_scale.shape).to(torch.int32).to(input_scale.device)
            input_data = torch.fake_quantize_per_tensor_affine(input_data * calib_info[module.module_name]["smoothquant_scale"], input_scale, zero_point, quant_min, quant_max)
            if not hasattr(module, "smoothquant_weight"):
                setattr(module, "smoothquant_weight", True)
                module.weight = torch.nn.Parameter(module.weight / calib_info[module.module_name]["smoothquant_scale"])
                calib_info[module.module_name]["weight"] = collect_min_max(module.weight, 0)
        else:
            input_scale = (calib_info[module.module_name]["input"] /  quant_max).to(torch.float32)
            zero_point = torch.zeros(input_scale.shape).to(torch.int32).to(input_scale.device)
            input_data = torch.fake_quantize_per_channel_affine(input_data, input_scale.view(-1), zero_point.view(-1), 2, quant_min, quant_max)

    if not hasattr(module, "org_weight"):
        weight_scale = (calib_info[module.module_name]["weight"] /  quant_max).to(torch.float32)
        zero_point = torch.zeros(weight_scale.shape).to(torch.int32).to(weight_scale.device)
        quant_weight = torch.fake_quantize_per_channel_affine(module.weight, weight_scale.view(-1), zero_point.view(-1), 0, quant_min, quant_max)
        setattr(module, "org_weight", module.weight)
        module.weight = torch.nn.Parameter(quant_weight) 
    args = list(args)
    args[0] = input_data
    return module.org_forward(*args, **kwargs)

def quant_forward(self, *args, **kwargs):
    if isinstance(self, torch.nn.Conv2d):
        return quant_conv(self, *args, **kwargs)
    if isinstance(self, torch.nn.Linear):
        return quant_linear(self, *args, **kwargs)
        
def replace_conv2d(model):
    global calib_info
    if calib_info is None:
        calib_data_path = str(model.__class__.__name__) + "__calib.pt"
        calib_info = torch.load(calib_data_path)
        print(f"[Quant] Loaded calib_info from {calib_data_path}")
    else:
        print(f"[Quant] Using pre-loaded calib_info ({len(calib_info)} entries)")
    
    quantized_layers = []
    skipped_layers = []
    for name, module in model.named_modules():
        setattr(module, "module_name", name)
        if filter_func(name):
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                skipped_layers.append(name)
            continue
        if (isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear))  and not hasattr(module, "org_forward"):
            setattr(module, "org_forward", module.forward)
            module.forward = partial(quant_forward, module)
            quantized_layers.append(name)

    print(f"[Quant] Quantized layers: {len(quantized_layers)}")
    print(f"[Quant] Skipped  layers: {len(skipped_layers)}")
    if skipped_layers:
        for skipped_name in skipped_layers:
            print(f"  [SKIP] {skipped_name}")
    for quantized_name in quantized_layers:
        print(f"  [QUANT] {quantized_name}")
    return quantized_layers


@torch.no_grad()
def export_quantized_model(model, export_path):
    """Pre-quantize all hooked weights and save model state_dict + metadata.

    Must be called after replace_conv2d() so forward hooks and calib_info are ready.

    :param model: The model whose quantized layers will be exported.
    :param str export_path: File path to save the exported checkpoint.
    """
    global calib_info
    quant_min = -128
    quant_max = 128
    quantized_layer_names = []

    for module_name, module in model.named_modules():
        if not hasattr(module, "org_forward"):
            continue
        quantized_layer_names.append(module_name)

        if isinstance(module, torch.nn.Linear) and not weight_only:
            if module_name in calib_info and "smoothquant_scale" in calib_info[module_name]:
                if not hasattr(module, "smoothquant_weight"):
                    module.weight = torch.nn.Parameter(
                        module.weight / calib_info[module_name]["smoothquant_scale"])
                    calib_info[module_name]["weight"] = collect_min_max(module.weight, 0)
                    setattr(module, "smoothquant_weight", True)

        if not hasattr(module, "org_weight"):
            weight_scale = (calib_info[module_name]["weight"] / quant_max).to(torch.float32)
            zero_point = torch.zeros(weight_scale.shape, dtype=torch.int32, device=weight_scale.device)
            quant_weight = torch.fake_quantize_per_channel_affine(
                module.weight, weight_scale.view(-1), zero_point.view(-1), 0, quant_min, quant_max)
            setattr(module, "org_weight", module.weight)
            module.weight = torch.nn.Parameter(quant_weight)

    clean_state_dict = {}
    for param_name, param_value in model.state_dict().items():
        if ".org_weight" in param_name:
            continue
        clean_state_dict[param_name] = param_value

    exported_calib_info = {}
    for layer_name in quantized_layer_names:
        if layer_name in calib_info:
            exported_calib_info[layer_name] = calib_info[layer_name]

    torch.save({
        "state_dict": clean_state_dict,
        "quantized_layers": quantized_layer_names,
        "calib_info": exported_calib_info,
        "weight_only": weight_only,
    }, export_path)
    print(f"[Export] Saved {len(quantized_layer_names)} quantized layers "
          f"+ calib_info ({len(exported_calib_info)} entries) to {export_path}")


def load_quantized_model(model, export_path):
    """Load pre-quantized weights, calib_info, and mark layers to skip re-quantization.

    Must be called BEFORE replace_conv2d() so the org_weight guards take effect.
    If the checkpoint contains calib_info, it is loaded into the global calib_info
    so replace_conv2d() does not need to read the separate __calib.pt file.

    :param model: The model to load quantized weights into.
    :param str export_path: Path to the exported checkpoint.

    :return: Set of quantized layer names from the checkpoint.
    :rtype: set
    """
    global calib_info
    checkpoint = torch.load(export_path, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    quantized_layer_set = set(checkpoint["quantized_layers"])

    for module_name, module in model.named_modules():
        if module_name in quantized_layer_set:
            setattr(module, "org_weight", module.weight)
            setattr(module, "smoothquant_weight", True)

    if "calib_info" in checkpoint:
        calib_info = checkpoint["calib_info"]
        print(f"[Load] Restored calib_info ({len(calib_info)} entries) from checkpoint")

    loaded_weight_only = checkpoint.get("weight_only", False)
    print(f"[Load] Loaded {len(quantized_layer_set)} pre-quantized layers from {export_path}")
    print(f"[Load] Checkpoint weight_only={loaded_weight_only}")
    return quantized_layer_set
