# Copyright (C) 2024 AMD, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Real int8 inference benchmark for Flux2Klein using torch._int_mm.

Loads calibration info, replaces nn.Linear with real int8 matmul,
and compares speed / GPU memory / model size vs float baseline.

Usage:
    python test_flux2klein_int8mm.py
    python test_flux2klein_int8mm.py --skip_level 1 --num_benchmark_runs 5
"""

import argparse
import gc
import os
import random
import re
import time

import numpy as np
import pandas as pd
import torch
from diffusers import Flux2KleinPipeline
from PIL import Image


class Int8Linear(torch.nn.Module):
    """Linear layer with real int8 matmul via torch._int_mm.

    Weights are stored as int8 (1 byte per element instead of 2 for bf16).
    Forward pass quantizes activations to int8, runs int8 matmul on tensor cores,
    then dequantizes the int32 output back to the original dtype.

    :param torch.nn.Linear original_linear: The original float linear layer to replace.
    :param dict calib_entry: Calibration info dict with keys 'input', 'weight',
        and optionally 'smoothquant_scale'.
    :param bool weight_only: If True, only store weights as int8 but dequantize
        before matmul (saves memory but no int8 compute speedup).
    """

    def __init__(self, original_linear, calib_entry, weight_only=False):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.weight_only = weight_only

        weight = original_linear.weight.data.float()

        has_smoothquant = ("smoothquant_scale" in calib_entry) and not weight_only
        if has_smoothquant:
            smooth_scale = calib_entry["smoothquant_scale"].float().to(weight.device)
            self.register_buffer(
                "smoothquant_scale", smooth_scale.to(original_linear.weight.dtype)
            )
            weight = weight / smooth_scale
            weight_amax = weight.abs().amax(dim=1)
            input_amax_scalar = (
                calib_entry["input"].float().to(weight.device) * smooth_scale
            ).max()
        else:
            self.smoothquant_scale = None
            weight_amax = calib_entry["weight"].float().to(weight.device)
            input_amax_scalar = calib_entry["input"].float().to(weight.device)
            if input_amax_scalar.numel() > 1:
                input_amax_scalar = input_amax_scalar.max()

        weight_scale = (weight_amax / 127.0).clamp(min=1e-10).float()
        weight_int8 = (
            (weight / weight_scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
        )

        self.register_buffer("weight_int8_t", weight_int8.t().contiguous())
        self.register_buffer("weight_scale", weight_scale)

        if original_linear.bias is not None:
            self.register_buffer("bias", original_linear.bias.data.clone())
        else:
            self.bias = None

        if not weight_only:
            input_scale = (input_amax_scalar / 127.0).clamp(min=1e-10).float()
            self.register_buffer("input_scale", input_scale)
            self.register_buffer(
                "dequant_scale", (input_scale * weight_scale).unsqueeze(0)
            )

    def forward(self, x):
        original_shape = x.shape
        original_dtype = x.dtype

        if self.weight_only:
            weight_deq = (
                self.weight_int8_t.float() * self.weight_scale.unsqueeze(0)
            ).t().to(original_dtype)
            return torch.nn.functional.linear(x, weight_deq, self.bias)

        if self.smoothquant_scale is not None:
            x = x * self.smoothquant_scale

        x_int8 = (
            (x.float() / self.input_scale).round().clamp(-128, 127).to(torch.int8)
        )

        x_2d = x_int8.reshape(-1, self.in_features).contiguous()

        # torch._int_mm requires M > 16; pad to 32 if needed for alignment
        actual_m = x_2d.shape[0]
        if actual_m <= 16:
            x_2d = torch.nn.functional.pad(x_2d, (0, 0, 0, 32 - actual_m))

        y_int32 = torch._int_mm(x_2d, self.weight_int8_t)

        if actual_m <= 16:
            y_int32 = y_int32[:actual_m]

        y_float = y_int32.float() * self.dequant_scale

        if self.bias is not None:
            y_float = y_float + self.bias.float()

        return y_float.reshape(*original_shape[:-1], self.out_features).to(original_dtype)


def get_model_memory_mb(model):
    """Compute total memory occupied by model parameters and buffers.

    :param model: A PyTorch model.

    :return: Memory in megabytes.
    :rtype: float
    """
    total_bytes = 0
    for parameter in model.parameters():
        total_bytes += parameter.nelement() * parameter.element_size()
    for buffer in model.buffers():
        total_bytes += buffer.nelement() * buffer.element_size()
    return total_bytes / (1024 * 1024)


def should_skip_layer(layer_name, skip_pattern):
    """Check whether a layer should be skipped from quantization.

    :param str layer_name: Fully qualified module name.
    :param str skip_pattern: Regex pattern for layers to skip, or None.

    :return: True if the layer should be skipped.
    :rtype: bool
    """
    if skip_pattern is not None:
        return re.compile(skip_pattern).match(layer_name) is not None
    default_pattern = re.compile(
        r".*(time_emb_proj|time_embedding|conv_in|conv_out|conv_shortcut|add_embedding).*"
    )
    return default_pattern.match(layer_name) is not None


@torch.no_grad()
def replace_linear_with_int8mm(model, calib_info, skip_pattern=None,
                               weight_only=False, use_compile=False):
    """Replace nn.Linear layers with Int8Linear using real int8 matmul.

    :param model: The transformer model whose linear layers will be replaced.
    :param dict calib_info: Calibration info loaded from __calib.pt.
    :param str skip_pattern: Regex for layer names to skip. None uses default.
    :param bool weight_only: If True, only quantize weights (no int8 activation).
    :param bool use_compile: If True, torch.compile each Int8Linear for kernel fusion.

    :return: Tuple of (list of replaced layer names, list of skipped layer names).
    :rtype: tuple[list[str], list[str]]
    """
    replaced_layers = []
    skipped_layers = []
    module_dict = dict(model.named_modules())

    for layer_name, module in list(model.named_modules()):
        if not isinstance(module, torch.nn.Linear):
            continue

        if should_skip_layer(layer_name, skip_pattern):
            skipped_layers.append(layer_name)
            continue

        if layer_name not in calib_info:
            skipped_layers.append(layer_name)
            continue

        calib_entry = calib_info[layer_name]
        if "input" not in calib_entry or "weight" not in calib_entry:
            skipped_layers.append(layer_name)
            continue

        parts = layer_name.rsplit(".", 1)
        if len(parts) == 2:
            parent_name, attribute_name = parts
            parent_module = module_dict[parent_name]
        else:
            parent_module = model
            attribute_name = layer_name

        int8_module = Int8Linear(module, calib_entry, weight_only=weight_only)
        if use_compile:
            int8_module = torch.compile(int8_module, mode="max-autotune-no-cudagraphs")
        setattr(parent_module, attribute_name, int8_module)
        replaced_layers.append(layer_name)

    print(f"[Int8mm] Replaced layers: {len(replaced_layers)}")
    print(f"[Int8mm] Skipped  layers: {len(skipped_layers)}")
    for name in skipped_layers:
        print(f"  [SKIP]    {name}")
    for name in replaced_layers:
        print(f"  [INT8MM]  {name}")
    return replaced_layers, skipped_layers


def fixed_seed_point():
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_calib_prompts(batch_size, calibration_data_path):
    """Load prompts from a TSV file and split into batches.

    :param int batch_size: Number of prompts per batch.
    :param str calibration_data_path: Path to the TSV file with a 'caption' column.

    :return: List of prompt batches.
    :rtype: list[list[str]]
    """
    dataframe = pd.read_csv(calibration_data_path, sep="\t")
    prompt_list = dataframe["caption"].tolist()
    return [
        prompt_list[index : index + batch_size]
        for index in range(0, len(prompt_list), batch_size)
    ]


def compute_psnr(image_a, image_b):
    """Compute PSNR between two PIL images.

    :param PIL.Image image_a: First image.
    :param PIL.Image image_b: Second image.

    :return: PSNR value in dB.
    :rtype: float
    """
    array_a = np.array(image_a, dtype=np.float64)
    array_b = np.array(image_b, dtype=np.float64)
    mse_value = np.mean((array_a - array_b) ** 2)
    if mse_value == 0:
        return float("inf")
    return 10 * np.log10(255.0**2 / mse_value)


def generate_images(pipe, prompts, num_inference_steps, guidance_scale,
                    height, width, save_path, start, end):
    """Run inference and save output images.

    :param pipe: The pipeline instance.
    :param list prompts: List of prompt batches.
    :param int num_inference_steps: Number of denoising steps.
    :param float guidance_scale: Guidance scale.
    :param int height: Output image height.
    :param int width: Output image width.
    :param str save_path: Directory to save images.
    :param int start: Start index (inclusive).
    :param int end: End index (exclusive).
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    actual_end = min(end, len(prompts))
    for batch_index, prompt_batch in enumerate(prompts):
        if batch_index < start or batch_index >= actual_end:
            continue
        print(f"[Inference] Batch {batch_index + 1}/{actual_end}")
        generated_image = pipe(
            prompt=prompt_batch,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=torch.Generator(device="cuda").manual_seed(0),
        ).images[0]
        generated_image.save(os.path.join(save_path, f"{batch_index}.png"))


def benchmark_inference(pipe, prompt, num_inference_steps, guidance_scale,
                        height, width, num_warmup=2, num_runs=5):
    """Benchmark per-inference latency and peak GPU memory.

    :param pipe: The pipeline instance.
    :param list prompt: A prompt batch.
    :param int num_inference_steps: Number of denoising steps.
    :param float guidance_scale: Guidance scale.
    :param int height: Output image height.
    :param int width: Output image width.
    :param int num_warmup: Warmup iterations. Default: 2.
    :param int num_runs: Timed iterations. Default: 5.

    :return: Tuple of (avg_latency_sec, peak_memory_mb).
    :rtype: tuple[float, float]
    """
    inference_kwargs = dict(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=torch.Generator(device="cuda").manual_seed(0),
    )

    for warmup_index in range(num_warmup):
        print(f"[Benchmark] Warmup {warmup_index + 1}/{num_warmup}")
        pipe(**inference_kwargs)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time = time.time()
    for run_index in range(num_runs):
        print(f"[Benchmark] Run {run_index + 1}/{num_runs}")
        pipe(**inference_kwargs)
    torch.cuda.synchronize()
    total_time = time.time() - start_time

    average_latency = total_time / num_runs
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return average_latency, peak_memory_mb


def compute_image_metrics(float_img_path, int8_img_path, start, end):
    """Compare float and int8 images by PSNR and MSE.

    :param str float_img_path: Directory with float baseline images.
    :param str int8_img_path: Directory with int8 images.
    :param int start: Start index (inclusive).
    :param int end: End index (exclusive).

    :return: Tuple of (mean_psnr, mean_mse, count).
    :rtype: tuple[float, float, int]
    """
    psnr_values = []
    mse_values = []
    for image_index in range(start, end):
        float_path = os.path.join(float_img_path, f"{image_index}.png")
        int8_path = os.path.join(int8_img_path, f"{image_index}.png")
        if not os.path.exists(float_path) or not os.path.exists(int8_path):
            continue
        float_image = Image.open(float_path)
        int8_image = Image.open(int8_path)
        psnr_values.append(compute_psnr(float_image, int8_image))
        array_a = np.array(float_image, dtype=np.float64)
        array_b = np.array(int8_image, dtype=np.float64)
        mse_values.append(float(np.mean((array_a - array_b) ** 2)))
    if not psnr_values:
        return 0.0, 0.0, 0
    return float(np.mean(psnr_values)), float(np.mean(mse_values)), len(psnr_values)


SKIP_LEVEL_PATTERNS = {
    0: None,
    1: r".*(context_embedder).*",
    2: r".*(context_embedder|ff_context\.linear_out|ff\.linear_out).*",
    3: r".*(context_embedder|x_embedder|ff_context\.linear_out|ff\.linear_out"
       r"|modulation|time_guidance_embed|norm_out|proj_out|to_add_out).*",
}


def main():
    parser = argparse.ArgumentParser(
        description="Real int8 inference benchmark for Flux2Klein using torch._int_mm. "
                    "Compares speed, GPU memory, and model size vs float baseline."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/scratch/meng/black-forest-labs/FLUX.2-klein-9B",
        help="Path to pretrained Flux model or model identifier from huggingface.co/models. "
             "Example: 'black-forest-labs/FLUX.2-klein-9B' or '/path/to/local/model'.",
    )
    parser.add_argument(
        "--calib_path",
        type=str,
        default="Flux2Transformer2DModel__calib.pt",
        help="Path to the calibration info file (__calib.pt). "
             "Default: 'Flux2Transformer2DModel__calib.pt'.",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="./captions_test.tsv",
        help="Path to TSV file with test prompts. Default: './captions_test.tsv'.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Number of prompts per batch. Default: 1.",
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=4,
        help="Number of denoising steps per inference. Default: 4.",
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=3.5,
        help="Guidance scale for classifier-free guidance. Default: 3.5.",
    )
    parser.add_argument(
        "--height", type=int, default=1024,
        help="Height of generated image in pixels. Default: 1024.",
    )
    parser.add_argument(
        "--width", type=int, default=1024,
        help="Width of generated image in pixels. Default: 1024.",
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="Start index for prompt batches (inclusive). Default: 0.",
    )
    parser.add_argument(
        "--end", type=int, default=5,
        help="End index for prompt batches (exclusive). Default: 5.",
    )
    parser.add_argument(
        "--float_img_path", type=str, default="output_int8mm_float",
        help="Directory to save float baseline images. Default: 'output_int8mm_float'.",
    )
    parser.add_argument(
        "--int8_img_path", type=str, default="output_int8mm_int8",
        help="Directory to save int8 images. Default: 'output_int8mm_int8'.",
    )
    parser.add_argument(
        "--skip_level", type=int, default=1, choices=[0, 1, 2, 3],
        help="Sensitivity skip level. 0: skip nothing. 1: skip context_embedder. "
             "2: also skip ff outputs. 3: widest skip. Default: 1.",
    )
    parser.add_argument(
        "--skip_layers", type=str, default=None,
        help="Custom regex for layer names to skip (overrides --skip_level).",
    )
    parser.add_argument(
        "--weight_only", action="store_true",
        help="Only store weights as int8 (saves memory), but dequantize to bf16 "
             "for matmul (no int8 compute speedup). Default: False.",
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="none",
        choices=["none", "module", "full"],
        help="torch.compile strategy. "
             "'none': no compilation (eager mode). "
             "'module': compile each Int8Linear independently (faster compile, less speedup). "
             "'full': compile entire transformer (slower compile, best speedup via CUDA graphs). "
             "Default: 'none'.",
    )
    parser.add_argument(
        "--num_benchmark_runs", type=int, default=5,
        help="Number of timed runs for speed benchmarking. Default: 5.",
    )
    parser.add_argument(
        "--num_images", type=int, default=3,
        help="Number of images to generate for quality comparison. Default: 3.",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Device for inference. Default: 'cuda'.",
    )

    args = parser.parse_args()
    fixed_seed_point()
    device = torch.device(args.device)
    dtype = torch.bfloat16

    skip_pattern = (
        args.skip_layers if args.skip_layers
        else SKIP_LEVEL_PATTERNS.get(args.skip_level)
    )
    quant_mode = "WEIGHT-ONLY" if args.weight_only else "W+A"
    print(f"[Config] Mode: {quant_mode} | skip_level: {args.skip_level} | "
          f"skip_pattern: {skip_pattern}")
    print(f"[Config] Benchmark runs: {args.num_benchmark_runs} | "
          f"Images: {args.num_images} | compile: {args.compile_mode}")

    # ── Load pipeline ──────────────────────────────────────────────────
    time_start = time.time()
    pipe = Flux2KleinPipeline.from_pretrained(
        args.pretrained_model_name_or_path, torch_dtype=dtype
    )
    pipe.to(device)
    print(f"[Timer] Pipeline loading: {time.time() - time_start:.2f}s")

    test_prompts = load_calib_prompts(args.batch_size, args.test_data)
    benchmark_prompt = test_prompts[0] if test_prompts else ["a photo of a cat"]

    # ── Phase 1: Float baseline ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE 1: Float (BF16) Baseline")
    print("=" * 70)

    float_model_size = get_model_memory_mb(pipe.transformer)
    float_gpu_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
    print(f"[Float] Model param+buffer size: {float_model_size:.0f} MB")
    print(f"[Float] GPU allocated: {float_gpu_allocated:.0f} MB")

    actual_end = min(args.start + args.num_images, args.end)
    generate_images(
        pipe, test_prompts, args.num_inference_steps, args.guidance_scale,
        args.height, args.width, args.float_img_path, args.start, actual_end,
    )

    float_latency, float_peak_mem = benchmark_inference(
        pipe, benchmark_prompt, args.num_inference_steps, args.guidance_scale,
        args.height, args.width, num_runs=args.num_benchmark_runs,
    )
    print(f"[Float] Avg latency: {float_latency:.4f}s")
    print(f"[Float] Peak GPU memory: {float_peak_mem:.0f} MB")

    # ── Phase 2: Convert to int8 ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE 2: Converting to Int8 (torch._int_mm)")
    print("=" * 70)

    calib_info = torch.load(args.calib_path, weights_only=False)
    print(f"[Int8mm] Loaded calib_info from {args.calib_path}")

    replace_linear_with_int8mm(
        pipe.transformer, calib_info,
        skip_pattern=skip_pattern,
        weight_only=args.weight_only,
        use_compile=(args.compile_mode == "module"),
    )

    gc.collect()
    torch.cuda.empty_cache()

    if args.compile_mode == "full":
        print("[Int8mm] Applying torch.compile(full transformer, max-autotune)...")
        pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune")

    int8_model_size = get_model_memory_mb(pipe.transformer)
    int8_gpu_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
    print(f"[Int8]  Model param+buffer size: {int8_model_size:.0f} MB")
    print(f"[Int8]  GPU allocated: {int8_gpu_allocated:.0f} MB")

    # ── Phase 3: Int8 benchmark ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE 3: Int8 Inference")
    print("=" * 70)

    generate_images(
        pipe, test_prompts, args.num_inference_steps, args.guidance_scale,
        args.height, args.width, args.int8_img_path, args.start, actual_end,
    )

    int8_latency, int8_peak_mem = benchmark_inference(
        pipe, benchmark_prompt, args.num_inference_steps, args.guidance_scale,
        args.height, args.width, num_runs=args.num_benchmark_runs,
    )
    print(f"[Int8]  Avg latency: {int8_latency:.4f}s")
    print(f"[Int8]  Peak GPU memory: {int8_peak_mem:.0f} MB")

    # ── Phase 4: Quality comparison ───────────────────────────────────
    mean_psnr, mean_mse, compared_count = compute_image_metrics(
        args.float_img_path, args.int8_img_path, args.start, actual_end,
    )

    # ── Summary ───────────────────────────────────────────────────────
    model_size_savings = (1.0 - int8_model_size / float_model_size) * 100
    gpu_alloc_savings = (1.0 - int8_gpu_allocated / float_gpu_allocated) * 100
    peak_mem_savings = (1.0 - int8_peak_mem / float_peak_mem) * 100
    speedup = float_latency / int8_latency if int8_latency > 0 else 0

    print("\n" + "=" * 70)
    print("  COMPARISON SUMMARY: Float (BF16) vs Int8 (torch._int_mm)")
    print("=" * 70)
    print(f"  Quantization mode:   {quant_mode}")
    print(f"  Skip level:          {args.skip_level}")
    print(f"  Inference steps:     {args.num_inference_steps}")
    print(f"  Resolution:          {args.height}x{args.width}")
    print()
    print(f"  {'Metric':<25} {'Float':>12} {'Int8':>12} {'Change':>12}")
    print(f"  {'-' * 61}")
    print(f"  {'Model size (MB)':<25} {float_model_size:>12.0f} {int8_model_size:>12.0f} "
          f"{model_size_savings:>+11.1f}%")
    print(f"  {'GPU allocated (MB)':<25} {float_gpu_allocated:>12.0f} {int8_gpu_allocated:>12.0f} "
          f"{gpu_alloc_savings:>+11.1f}%")
    print(f"  {'Peak GPU mem (MB)':<25} {float_peak_mem:>12.0f} {int8_peak_mem:>12.0f} "
          f"{peak_mem_savings:>+11.1f}%")
    print(f"  {'Avg latency (s)':<25} {float_latency:>12.4f} {int8_latency:>12.4f} "
          f"{'':>5}{speedup:.2f}x")
    if compared_count > 0:
        print(f"  {'PSNR (dB)':<25} {'--':>12} {mean_psnr:>12.2f}")
        print(f"  {'MSE':<25} {'--':>12} {mean_mse:>12.1f}")
        print(f"  {'Compared images':<25} {'':>12} {compared_count:>12}")
    print("=" * 70)


if __name__ == "__main__":
    main()
