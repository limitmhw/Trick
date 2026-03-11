import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from diffusers import Flux2KleinPipeline
from PIL import Image

import trick_min as trick


def print_calib_summary(model):
    """Print detailed calibration summary: which layers got smoothquant, amax stats, outliers."""
    print("\n" + "=" * 90)
    print("[Calibration Summary] Layer-wise activation / weight amax after smooth_linear")
    print("=" * 90)
    print(f"{'Layer':<60} {'i_max':>10} {'w_max':>10} {'smooth':>6}")
    print("-" * 90)

    layer_stats = []
    for module_name, module in model.named_modules():
        if not isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            continue
        has_input_amax = hasattr(module, "module_i_amax")
        has_weight_amax = hasattr(module, "module_w_amax")
        has_smooth = hasattr(module, "smoothquant_scale")
        if not has_input_amax and not has_weight_amax:
            continue

        input_max = module.module_i_amax.max().item() if has_input_amax else 0.0
        weight_max = module.module_w_amax.max().item() if has_weight_amax else 0.0
        layer_stats.append((module_name, input_max, weight_max, has_smooth))

        smooth_flag = "Y" if has_smooth else "N"
        print(f"  {module_name:<58} {input_max:>10.2f} {weight_max:>10.4f} {smooth_flag:>6}")

    print("-" * 90)
    print(f"  Total calibrated layers: {len(layer_stats)}")

    if layer_stats:
        sorted_by_input = sorted(layer_stats, key=lambda x: x[1], reverse=True)
        print(f"\n  [!] Top 10 layers by input amax (potential sensitivity):")
        for layer_name, input_max_value, weight_max_value, _ in sorted_by_input[:10]:
            print(f"      {layer_name:<55} i_max={input_max_value:>12.2f}")
    print("=" * 90 + "\n")


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
    return [prompt_list[index: index + batch_size]
            for index in range(0, len(prompt_list), batch_size)]


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
    return 10 * np.log10(255.0 ** 2 / mse_value)


def compute_mse(image_a, image_b):
    """Compute mean squared error between two PIL images.

    :param PIL.Image image_a: First image.
    :param PIL.Image image_b: Second image.

    :return: MSE value.
    :rtype: float
    """
    array_a = np.array(image_a, dtype=np.float64)
    array_b = np.array(image_b, dtype=np.float64)
    return float(np.mean((array_a - array_b) ** 2))


def do_calibrate(pipe, calibration_prompts, num_inference_steps, guidance_scale,
                 height, width):
    """Run calibration inference passes over prompt batches to collect activation statistics.

    :param pipe: The Flux pipeline instance.
    :param list calibration_prompts: List of prompt batches.
    :param int num_inference_steps: Number of denoising steps.
    :param float guidance_scale: Classifier-free guidance scale.
    :param int height: Output image height.
    :param int width: Output image width.
    """
    for batch_index, prompts in enumerate(calibration_prompts):
        print(f"[Calibration] Batch {batch_index + 1}/{len(calibration_prompts)}")
        pipe(
            prompt=prompts,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=torch.Generator(device="cuda").manual_seed(0),
        ).images


def do_test(pipe, test_prompts, num_inference_steps, guidance_scale,
            height, width, save_img_path, start, end):
    """Run inference over prompt batches and save output images with captions.

    :param pipe: The Flux pipeline instance.
    :param list test_prompts: List of prompt batches.
    :param int num_inference_steps: Number of denoising steps.
    :param float guidance_scale: Classifier-free guidance scale.
    :param int height: Output image height.
    :param int width: Output image width.
    :param str save_img_path: Directory to save output images.
    :param int start: Start index for prompt batches (inclusive).
    :param int end: End index for prompt batches (exclusive).
    """
    if not os.path.exists(save_img_path):
        os.makedirs(save_img_path)

    actual_end = min(end, len(test_prompts))
    for batch_index, prompts in enumerate(test_prompts):
        if batch_index < start or batch_index >= actual_end:
            continue
        print(f"[Inference] Batch {batch_index + 1}/{actual_end}")
        generated_image = pipe(
            prompt=prompts,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=torch.Generator(device="cuda").manual_seed(0),
        ).images[0]

        with open(os.path.join(save_img_path, str(batch_index) + ".txt"), "w") as caption_file:
            caption_file.write(prompts[0])
        generated_image.save(os.path.join(save_img_path, str(batch_index) + ".png"))


def benchmark_speed_and_memory(pipe, prompt, num_inference_steps, guidance_scale,
                               height, width, num_warmup=2, num_runs=5):
    """Benchmark per-inference latency and peak GPU memory.

    :param pipe: The pipeline instance.
    :param str prompt: A single text prompt used for benchmarking.
    :param int num_inference_steps: Number of denoising steps.
    :param float guidance_scale: Classifier-free guidance scale.
    :param int height: Output image height.
    :param int width: Output image width.
    :param int num_warmup: Warmup iterations (not timed). Default: 2.
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


def compute_image_metrics(float_img_path, quant_img_path, start, end):
    """Compare float and quantized images by computing PSNR and MSE per pair.

    :param str float_img_path: Directory containing float baseline images.
    :param str quant_img_path: Directory containing int8 quantized images.
    :param int start: Start index for comparison (inclusive).
    :param int end: End index for comparison (exclusive).

    :return: Tuple of (mean_psnr, mean_mse, count).
    :rtype: tuple[float, float, int]
    """
    psnr_values = []
    mse_values = []
    for image_index in range(start, end):
        float_path = os.path.join(float_img_path, str(image_index) + ".png")
        quant_path = os.path.join(quant_img_path, str(image_index) + ".png")
        if not os.path.exists(float_path) or not os.path.exists(quant_path):
            continue
        float_image = Image.open(float_path)
        quant_image = Image.open(quant_path)
        psnr_values.append(compute_psnr(float_image, quant_image))
        mse_values.append(compute_mse(float_image, quant_image))

    if not psnr_values:
        return 0.0, 0.0, 0
    return float(np.mean(psnr_values)), float(np.mean(mse_values)), len(psnr_values)


g_args = None


def main():
    parser = argparse.ArgumentParser(
        description="Flux pipeline text-to-image with int8 quantization for pipe.transformer"
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/scratch/meng/black-forest-labs/FLUX.2-klein-9B",
        help="Path to pretrained Flux model or model identifier from huggingface.co/models. "
             "Example: 'black-forest-labs/FLUX.2-klein-9B' or '/path/to/local/model'.",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="calib",
        choices=["calib", "test", "float", "export"],
        help="Run mode. 'calib': calibrate for int8 quantization. "
             "'test': run int8 quantized inference. "
             "'float': run bfloat16 baseline inference. "
             "'export': apply quantization and save model weights. Default: 'calib'.",
    )
    parser.add_argument(
        "--calib_data",
        type=str,
        default="./captions_calib.tsv",
        help="Path to the TSV file containing calibration prompts (must have a 'caption' column). "
             "Default: './captions_calib.tsv'.",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="./captions_test.tsv",
        help="Path to the TSV file containing test prompts (must have a 'caption' column). "
             "Default: './captions_test.tsv'.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of prompts per batch for calibration and inference. Default: 1.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=4,
        help="Number of denoising steps per inference. Default: 4.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="Guidance scale for classifier-free guidance. Default: 3.5.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Height of the generated image in pixels. Default: 1024.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Width of the generated image in pixels. Default: 1024.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index for test prompt batches (inclusive). Default: 0.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=10000000,
        help="End index for test prompt batches (exclusive). Default: 10000000.",
    )
    parser.add_argument(
        "--save_img_path",
        type=str,
        default="output_flux2klein",
        help="Directory to save output images and captions. Default: 'output_flux2klein'.",
    )
    parser.add_argument(
        "--float_img_path",
        type=str,
        default="output_flux2klein_float",
        help="Directory containing float baseline images for metric comparison. "
             "Default: 'output_flux2klein_float'.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run inference on. Example: 'cuda' or 'cuda:0'. Default: 'cuda'.",
    )
    parser.add_argument(
        "--weight_only",
        action="store_true",
        help="Only quantize weights to int8, keep activations in original precision. Default: False.",
    )
    parser.add_argument(
        "--num_calib_samples",
        type=int,
        default=50,
        help="Maximum number of calibration samples to use. Default: 50.",
    )
    parser.add_argument(
        "--skip_level",
        type=int,
        default=1,
        choices=[0, 1, 2, 3],
        help="Sensitivity skip level for Flux2Klein transformer. "
             "0: skip nothing (quantize all layers). "
             "1: skip context_embedder only (i_max=22528, most sensitive). "
             "2: skip level-1 + ff_context.linear_out + ff.linear_out (high amax feed-forward outputs). "
             "3: skip level-2 + embedders + modulation + to_add_out + norm_out + proj_out (widest skip). "
             "Default: 1.",
    )
    parser.add_argument(
        "--skip_layers",
        type=str,
        default=None,
        help="Custom regex pattern for layer names to skip (overrides --skip_level). "
             "Example: '.*(context_embedder|proj_out).*'.",
    )
    parser.add_argument(
        "--num_benchmark_runs",
        type=int,
        default=5,
        help="Number of timed runs for speed/memory benchmarking. Default: 5.",
    )
    parser.add_argument(
        "--export_quant_path",
        type=str,
        default="flux2klein_int8.pt",
        help="Path to save the exported quantized model (used with --type export). "
             "Default: 'flux2klein_int8.pt'.",
    )
    parser.add_argument(
        "--load_quant_path",
        type=str,
        default=None,
        help="Path to a previously exported quantized model checkpoint. "
             "When set with --type test, loads pre-quantized weights instead of re-quantizing.",
    )

    args = parser.parse_args()
    global g_args
    g_args = args
    fixed_seed_point()
    device = torch.device(args.device)
    dtype = torch.bfloat16

    # Load pipeline
    time_start = time.time()
    pipe = Flux2KleinPipeline.from_pretrained(
        args.pretrained_model_name_or_path, torch_dtype=dtype
    )
    pipe.to(device)
    print(f"[Timer] Pipeline loading: {time.time() - time_start:.2f}s")

    skip_level_patterns = {
        0: None,
        1: r".*(context_embedder).*",
        2: r".*(context_embedder|ff_context\.linear_out|ff\.linear_out).*",
        3: r".*(context_embedder|x_embedder|ff_context\.linear_out|ff\.linear_out"
           r"|modulation|time_guidance_embed|norm_out|proj_out|to_add_out).*",
    }

    def apply_quant_config():
        """Set weight_only and skip_pattern globals based on CLI args."""
        if args.weight_only:
            trick.weight_only = True
        resolved_pattern = args.skip_layers if args.skip_layers else skip_level_patterns.get(args.skip_level)
        if resolved_pattern:
            trick.skip_pattern = resolved_pattern
            source = "--skip_layers" if args.skip_layers else f"--skip_level {args.skip_level}"
            print(f"[Quantization] Skip pattern ({source}): {resolved_pattern}")
        else:
            print("[Quantization] Skip level 0: quantizing all layers")

    if args.type == "calib":
        print("[Quantization] Starting calibration on pipe.transformer...")
        trick.calib_regist(pipe.transformer)
        calibration_prompts = load_calib_prompts(args.batch_size, args.calib_data)
        calibration_prompts = calibration_prompts[:args.num_calib_samples]
        print(f"[Quantization] Using {len(calibration_prompts)} calibration samples")
        do_calibrate(
            pipe=pipe,
            calibration_prompts=calibration_prompts,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
        )
        trick.do_smooth_linear(pipe.transformer)
        print_calib_summary(pipe.transformer)
        trick.save_calib_result(pipe.transformer)
        print("[Quantization] Calibration complete. Calibration results saved.")

    elif args.type == "export":
        quant_mode = "WEIGHT-ONLY" if args.weight_only else "W+A"
        print(f"[Export] Exporting int8 {quant_mode} quantized model...")
        apply_quant_config()
        trick.replace_conv2d(pipe.transformer)
        trick.export_quantized_model(pipe.transformer, args.export_quant_path)

    elif args.type == "test":
        quant_mode = "WEIGHT-ONLY" if args.weight_only else "W+A"
        print(f"[Quantization] Applying int8 {quant_mode} quantization to pipe.transformer...")
        apply_quant_config()

        if args.load_quant_path:
            print(f"[Load] Loading pre-quantized weights from {args.load_quant_path}")
            trick.load_quantized_model(pipe.transformer, args.load_quant_path)

        trick.replace_conv2d(pipe.transformer)

        time_start = time.time()
        test_prompts = load_calib_prompts(args.batch_size, args.test_data)
        do_test(
            pipe=pipe,
            test_prompts=test_prompts,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            save_img_path=args.save_img_path,
            start=args.start,
            end=args.end,
        )
        print(f"[Timer] Int8 inference total: {time.time() - time_start:.2f}s")

        mean_psnr, mean_mse, compared_count = compute_image_metrics(
            args.float_img_path, args.save_img_path, args.start,
            min(args.end, len(test_prompts)),
        )
        if compared_count > 0:
            print(f"[Metrics] Compared {compared_count} image pairs")
            print(f"[Metrics] Mean PSNR (float vs int8): {mean_psnr:.2f} dB")
            print(f"[Metrics] Mean MSE  (float vs int8): {mean_mse:.4f}")
        else:
            print(f"[Metrics] No float baseline images found in {args.float_img_path}. "
                  "Run with --type float --save_img_path {args.float_img_path} first.")

        if args.num_benchmark_runs > 0:
            benchmark_prompt = test_prompts[0] if test_prompts else ["a photo of a cat"]
            int8_latency, int8_peak_mem = benchmark_speed_and_memory(
                pipe=pipe,
                prompt=benchmark_prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                height=args.height,
                width=args.width,
                num_runs=args.num_benchmark_runs,
            )
            print(f"[Benchmark] Int8 avg latency: {int8_latency:.4f}s | "
                  f"Peak GPU memory: {int8_peak_mem:.0f} MB")

    elif args.type == "float":
        time_start = time.time()
        test_prompts = load_calib_prompts(args.batch_size, args.test_data)
        do_test(
            pipe=pipe,
            test_prompts=test_prompts,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            save_img_path=args.save_img_path,
            start=args.start,
            end=args.end,
        )
        print(f"[Timer] Float inference total: {time.time() - time_start:.2f}s")

        if args.num_benchmark_runs > 0:
            benchmark_prompt = test_prompts[0] if test_prompts else ["a photo of a cat"]
            float_latency, float_peak_mem = benchmark_speed_and_memory(
                pipe=pipe,
                prompt=benchmark_prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                height=args.height,
                width=args.width,
                num_runs=args.num_benchmark_runs,
            )
            print(f"[Benchmark] Float avg latency: {float_latency:.4f}s | "
                  f"Peak GPU memory: {float_peak_mem:.0f} MB")


if __name__ == "__main__":
    main()
