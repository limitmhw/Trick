import os
import torch
import glob
import numpy as np
from PIL import Image
from tools.clip.clip_encoder import CLIPEncoder
from tools.fid.fid_score import compute_fid
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--image_dir', default='float_img')
    args = parser.parse_args()
    
    device = torch.device(args.device)

    def find_image_files(directory):
        image_files = []
        
        pattern_png = os.path.join(directory, '**', '*.png')
        png_files = glob.glob(pattern_png, recursive=True)
        image_files.extend(png_files)
        
        pattern_jpg = os.path.join(directory, '**', '*.jpg')
        jpg_files = glob.glob(pattern_jpg, recursive=True)
        image_files.extend(jpg_files)
        
        return image_files

    img_files = find_image_files(args.image_dir)
    image_numpy = []
    caption_list = []
    for img in img_files:
        image = Image.open(img)
        image_numpy.append(np.array(image, dtype=np.uint8))
        with open(img[:-3] + 'txt', 'r') as f:
            data = f.read()
            caption_list.append(data)

    # clip score
    clip_scores = []
    clip = CLIPEncoder(device=device)
    for k in range(len(caption_list)):
        caption = caption_list[k]
        generated = Image.fromarray(image_numpy[k])
        clip_scores.append(
            100 * clip.get_clip_score(caption, generated).item()
        )

    clip_score = np.mean(clip_scores)
    print("clip_score:", clip_score)
    #fid score
    statistics_path="./inference/text_to_image/tools/val2014.npz"
    fid = compute_fid(image_numpy, statistics_path, device)
    print("fid:", fid)
