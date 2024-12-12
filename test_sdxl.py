import argparse
import types
import torch
import torch.nn as nn
from diffusers import DiffusionPipeline
import os
import trick_min as trick
import pandas as pd
from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear


def lora_forward(self, x, scale=None):
    return self._torch_forward(x)


def replace_lora_layers(unet):
    for name, module in unet.named_modules():
        if isinstance(module, LoRACompatibleConv):
            in_channels = module.in_channels
            out_channels = module.out_channels
            kernel_size = module.kernel_size
            stride = module.stride
            padding = module.padding
            dilation = module.dilation
            groups = module.groups
            bias = module.bias

            new_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias is not None,
            )

            new_conv.weight.data = module.weight.data.clone().to(module.weight.data.device)
            if bias is not None:
                new_conv.bias.data = module.bias.data.clone().to(module.bias.data.device)

            # Replace the LoRACompatibleConv layer with the Conv2d layer
            path = name.split(".")
            sub_module = unet
            for p in path[:-1]:
                sub_module = getattr(sub_module, p)
            setattr(sub_module, path[-1], new_conv)
            new_conv._torch_forward = new_conv.forward
            new_conv.forward = types.MethodType(lora_forward, new_conv)

        elif isinstance(module, LoRACompatibleLinear):
            in_features = module.in_features
            out_features = module.out_features
            bias = module.bias

            new_linear = nn.Linear(in_features, out_features, bias=bias is not None)

            new_linear.weight.data = module.weight.data.clone().to(module.weight.data.device)
            if bias is not None:
                new_linear.bias.data = module.bias.data.clone().to(module.bias.data.device)

            # Replace the LoRACompatibleLinear layer with the Linear layer
            path = name.split(".")
            sub_module = unet
            for p in path[:-1]:
                sub_module = getattr(sub_module, p)
            setattr(sub_module, path[-1], new_linear)
            new_linear._torch_forward = new_linear.forward
            new_linear.forward = types.MethodType(lora_forward, new_linear)


def load_calib_prompts(batch_size, calib_data_path):
    df = pd.read_csv(calib_data_path, sep="\t")
    lst = df["caption"].tolist()
    return [lst[i: i + batch_size] for i in range(0, len(lst), batch_size)]

def do_test(base, calibration_prompts, **kwargs):
    start = g_args.start if g_args.start >= 0 else 0
    end = g_args.end if g_args.end < len(calibration_prompts) else len(calibration_prompts)
    if not os.path.exists(g_args.save_img_path):
        os.makedirs(g_args.save_img_path)
    for i_th, prompts in enumerate(calibration_prompts):
        if i_th < start or i_th >= end:
            continue
        img = base(
            prompt=prompts,
            num_inference_steps=kwargs["n_steps"],
            negative_prompt=[
                "normal quality, low quality, worst quality, low res, blurry, nsfw, nude"
            ]
            * len(prompts),
            latents=kwargs["latent"],
            guidance_scale=8.0,  # MLPerf requirements
        ).images[0]
        
        with open(os.path.join(g_args.save_img_path, str(i_th) + ".txt"), "w") as f:
            f.write(prompts[0])
        img.save(os.path.join(g_args.save_img_path, str(i_th) + ".png"))

def do_calibrate(base, calibration_prompts, **kwargs):
    for i_th, prompts in enumerate(calibration_prompts):
        base(
            prompt=prompts,
            num_inference_steps=kwargs["n_steps"],
            negative_prompt=[
                "normal quality, low quality, worst quality, low res, blurry, nsfw, nude"
            ]
            * len(prompts),
            latents=kwargs["latent"],
            guidance_scale=8.0,  # MLPerf requirements
        ).images

    
g_args = None
def main():
    parser = argparse.ArgumentParser()
    # Model hyperparameters
    parser.add_argument("--onnx-dir", default=None)
    parser.add_argument(
        "--pretrained-base",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0"
    )
    parser.add_argument("--format", default="int8", choices=["int8"])  # Now only support int8
    parser.add_argument(
        "--quant-level",
        default=3.0,
        type=float,
        choices=[1.0, 2.0, 2.5, 3.0],
        help="Quantization level, 1: CNN, 2: CNN+FFN, 2.5: CNN+FFN+QKV, 3: CNN+FC",
    )
    parser.add_argument("--n_steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)

    parser.add_argument("--test_data", type=str, default="./captions_test.tsv")
    parser.add_argument("--calib_data", type=str, default="./captions_calib.tsv")
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument("--latent", type=str, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=10000000)
    parser.add_argument("--save_img_path", type=str, default="output_trick")
    parser.add_argument("--type", type=str, default="calib")

    args = parser.parse_args()
    global g_args
    g_args = args
    device = torch.device(args.device)

    base = DiffusionPipeline.from_pretrained(
        args.pretrained_base,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        cache_dir="./cache",
    ).to(device)
    replace_lora_layers(base.unet)

    base.unet.to(device)
    init_latent = None
    if args.latent is not None:
        init_latent = torch.load(args.latent).to(torch.float16)

    base.unet.to(torch.float16).to(device)

    if g_args.type == 'calib':
        trick.calib_regist(base.unet)
        do_calibrate(
            base=base,
            calibration_prompts=load_calib_prompts(1, args.calib_data),
            n_steps=args.n_steps,
            latent=init_latent,
        )
        trick.do_smooth_linear(base.unet)
        trick.save_calib_result(base.unet)
        
    if g_args.type == 'test':
        trick.replace_conv2d(base.unet)
        do_test(
            base=base,
            calibration_prompts=load_calib_prompts(1, args.test_data),
            n_steps=args.n_steps,
            latent=init_latent,
        )

if __name__ == "__main__":
    main()
