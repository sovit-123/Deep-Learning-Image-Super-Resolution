from PIL import Image

import glob as glob
import os

images = glob.glob('../input/Set14/*.png')

for image in images:
    orig_img = Image.open(image)
    image_name = image.split(os.path.sep)[-1]
    w, h = orig_img.size[:]
    print(f"Original image dimensions: {w}, {h}")
    # 4x scaling factor == 1/4
    scale_factor = 0.25
    low_res_img = orig_img.resize((int(w*scale_factor), int(h*scale_factor)), Image.BICUBIC)
    # print(f"Low resolution image dimensions: {low_res_img.size[0]}, {low_res_img.size[1]}")
    # low_res_img.save('low_res_image.jpg')

    # now upscale using BICUBIC
    high_res_upscale = low_res_img.resize((w, h), Image.BICUBIC)
    high_res_upscale.save(f"../input/bicubic_rgb_4x/{image_name}")