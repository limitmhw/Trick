
```
export ROOT=$PWD
git clone https://github.com/mlcommons/inference.git
cd inference
git checkout 87ba8cb8a6a4f6525f26255fa513d902b17ab060
cd ./text_to_image/tools/
sh ./download-coco-2014.sh --num-workers 5
sh ./download-coco-2014-calibration.sh -n 5
cd $ROOT

ln -s ./inference/text_to_image/coco2014/calibration/captions.tsv ./captions_calib.tsv
ln -s ./inference/text_to_image/coco2014/captions/captions_source.tsv  ./captions_test.tsv



python test_sdxl.py --type calib --latent inference/text_to_image/tools/latents.pt
python test_sdxl.py --type test --latent inference/text_to_image/tools/latents.pt --save_img_path quant_img

python test_sdxl.py --type float --latent inference/text_to_image/tools/latents.pt --save_img_path float_img


```

eval 
```
export PYTHONPATH=./inference/text_to_image/:./inference/text_to_image/tools:$PYTHONPATH
python eval.py --image_dir float_img

sdxl without quant:
clip_score: 31.219369116127492
fid: 23.52008524090229

python eval.py --image_dir quant_img

sdxl quant:
clip_score: 31.152094860374927
fid: 24.147928398214162
```
