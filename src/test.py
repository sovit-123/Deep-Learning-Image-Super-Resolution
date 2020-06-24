"""
USAGE EXAMPLE:
python test.py --input ../input/bicubic_rgb_2x
"""

import torch
import cv2
import srcnn
import numpy as np
import glob
import os
import argparse

from tqdm import tqdm

# construct the argument parser and parse the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='../input/bicubic_rgb_2x', 
                    help='path to the low resolution images')
args = vars(parser.parse_args())

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = srcnn.SRCNN4().to(device)
model.load_state_dict(torch.load('../outputs/model.pth'))

scale_factor = args['input'].split('_')[-1]
print(f"Applying SRCNN on {scale_factor} bicubic images...")

image_paths = glob.glob(f"{args['input']}/*")
for image_path in tqdm(image_paths, total=len(image_paths)):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_name = image_path.split(os.path.sep)[-1]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape(image.shape[0], image.shape[1], 3)
    image = image / 255. # normalize the pixel values


    model.eval()
    with torch.no_grad():
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image = torch.tensor(image, dtype=torch.float).to(device)
        image = image.unsqueeze(0)
        outputs = model(image)

    # save the image
    outputs = outputs.cpu()
    outputs = outputs.squeeze(0).numpy()
    outputs = np.transpose(outputs, (1, 2, 0))
    outputs = cv2.cvtColor(outputs, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"../outputs/outputs_{image_name}", outputs*255.) # *255. to make it cv2 compatible
    
print('DONE INFERENCING')