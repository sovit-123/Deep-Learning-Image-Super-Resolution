import torch
import cv2
import srcnn
import numpy as np
import glob
import os
import time
import math

from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = srcnn.SRCNN4().to(device)
model.load_state_dict(torch.load('../outputs/model.pth'))

cap = cv2.VideoCapture('../input/test_videos/w3.mp4')
 
if (cap.isOpened() == False):
    print('Error while trying to read video. Plese check again...')
 
# get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
 
# define codec and create VideoWriter object
out = cv2.VideoWriter('../outputs/w3.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width,frame_height))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        start = time.time()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame / 255. # normalize the pixel values

        model.eval()
        with torch.no_grad():
            frame = np.transpose(frame, (2, 0, 1))
            frame = torch.tensor(frame, dtype=torch.float).to(device)
            frame = frame.unsqueeze(0).to(device)
            outputs = model(frame)

        # save the image
        outputs = outputs.cpu().numpy()
        outputs = outputs.squeeze(0)
        outputs = np.transpose(outputs, (1, 2, 0))
        outputs = cv2.cvtColor(outputs, cv2.COLOR_RGB2BGR)
        end = time.time()
        fps = 1 / (end - start)
        wait_time = max(1, int(fps/4))
        cv2.imshow('outputs', outputs)
        outputs = (outputs*255.).astype(np.uint8)
        out.write(outputs)
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break
    else:
        break
        
print('DONE TESTING')