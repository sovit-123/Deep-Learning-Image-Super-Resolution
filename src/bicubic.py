"""
This python script is for converting high-resolution images to low-resolution
bicubic images.
USAGE EXAMPLE:
python bicubic.py --path ../input/Set14 --scale-factor 2x
"""

from PIL import Image

import glob as glob
import os
import argparse

# construct the argument parser and parse the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', default='../input/Set14', 
                    help='path to the high-res images to convert to low-res')
parser.add_argument('-s', '--scale-factor', dest='scale_factor', default='2x', 
                    help='make low-res by how much factor', 
                    choices=['2x', '3x', '4x'])
args = vars(parser.parse_args())

path = args['path']
images = glob.glob(f"{path}/*.png")

# select scaling-factor
if args['scale_factor'] == '2x':
    scale_factor = 0.5
    os.makedirs('../input/bicubic_rgb_2x', exist_ok=True)
    save_path = '../input/bicubic_rgb_2x'
if args['scale_factor'] == '3x':
    scale_factor = 0.333
    os.makedirs('../input/bicubic_rgb_3x', exist_ok=True)
    save_path = '../input/bicubic_rgb_3x'
if args['scale_factor'] == '4x':
    scale_factor = 0.25
    os.makedirs('../input/bicubic_rgb_4x', exist_ok=True)
    save_path = '../input/bicubic_rgb_4x'

print(f"Scaling factor: {args['scale_factor']}")
print(f"Low resolution images save path: {save_path}")

for image in images:
    orig_img = Image.open(image)
    image_name = image.split(os.path.sep)[-1]
    w, h = orig_img.size[:]
    print(f"Original image dimensions: {w}, {h}")
    low_res_img = orig_img.resize((int(w*scale_factor), int(h*scale_factor)), Image.BICUBIC)

    # now upscale using BICUBIC
    high_res_upscale = low_res_img.resize((w, h), Image.BICUBIC)
    high_res_upscale.save(f"{save_path}/{image_name}")