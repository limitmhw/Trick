CUDA_VISIBLE_DEVICES=0 python test_sdxl.py --type calib
CUDA_VISIBLE_DEVICES=0 python test_sdxl.py --type float  --start 0    --end 625 --save_img_path float_img  --latent inference/text_to_image/tools/latents.pt &
CUDA_VISIBLE_DEVICES=1 python test_sdxl.py --type float  --start 625  --end 1250 --save_img_path float_img  --latent inference/text_to_image/tools/latents.pt &
CUDA_VISIBLE_DEVICES=2 python test_sdxl.py --type float  --start 1250 --end 1875 --save_img_path float_img  --latent inference/text_to_image/tools/latents.pt &
CUDA_VISIBLE_DEVICES=3 python test_sdxl.py --type float  --start 1875 --end 2500 --save_img_path float_img  --latent inference/text_to_image/tools/latents.pt &
CUDA_VISIBLE_DEVICES=4 python test_sdxl.py --type float  --start 2500 --end 3125 --save_img_path float_img  --latent inference/text_to_image/tools/latents.pt &
CUDA_VISIBLE_DEVICES=5 python test_sdxl.py --type float  --start 3125 --end 3750 --save_img_path float_img  --latent inference/text_to_image/tools/latents.pt &
CUDA_VISIBLE_DEVICES=6 python test_sdxl.py --type float  --start 3750 --end 4375 --save_img_path float_img  --latent inference/text_to_image/tools/latents.pt &
CUDA_VISIBLE_DEVICES=7 python test_sdxl.py --type float  --start 4375 --end 5000 --save_img_path float_img  --latent inference/text_to_image/tools/latents.pt &

CUDA_VISIBLE_DEVICES=0 python test_sdxl.py --type test  --start 0    --end 625 --save_img_path quant_img  --latent inference/text_to_image/tools/latents.pt &
CUDA_VISIBLE_DEVICES=1 python test_sdxl.py --type test  --start 625  --end 1250 --save_img_path quant_img  --latent inference/text_to_image/tools/latents.pt &
CUDA_VISIBLE_DEVICES=2 python test_sdxl.py --type test  --start 1250 --end 1875 --save_img_path quant_img  --latent inference/text_to_image/tools/latents.pt &
CUDA_VISIBLE_DEVICES=3 python test_sdxl.py --type test  --start 1875 --end 2500 --save_img_path quant_img  --latent inference/text_to_image/tools/latents.pt &
CUDA_VISIBLE_DEVICES=4 python test_sdxl.py --type test  --start 2500 --end 3125 --save_img_path quant_img  --latent inference/text_to_image/tools/latents.pt &
CUDA_VISIBLE_DEVICES=5 python test_sdxl.py --type test  --start 3125 --end 3750 --save_img_path quant_img  --latent inference/text_to_image/tools/latents.pt &
CUDA_VISIBLE_DEVICES=6 python test_sdxl.py --type test  --start 3750 --end 4375 --save_img_path quant_img  --latent inference/text_to_image/tools/latents.pt &
CUDA_VISIBLE_DEVICES=7 python test_sdxl.py --type test  --start 4375 --end 5000 --save_img_path quant_img  --latent inference/text_to_image/tools/latents.pt &

wait
python eval.py --image_dir float_img
python eval.py --image_dir quant_img
