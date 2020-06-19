import matplotlib.pyplot as plt
import h5py
import cv2
import numpy as np

# input image dimensions
img_rows, img_cols = 33, 33
out_rows, out_cols = 33, 33

file = h5py.File('../input/train_mscale.h5')
# `in_train` has shape (21884, 33, 33, 1) which corresponds to
# 21884 image patches of 33 pixels height & width and 1 color channel
in_train = file['data'][:] # the training data
out_train = file['label'][:] # the training labels
file.close()

# change the values to float32
in_train = in_train.astype('float32')
out_train = out_train.astype('float32')
print(in_train.shape)

""" Visualizing """
plt.figure()
for i in range(10):
    image = in_train[0].reshape(img_rows, img_cols, 3)
    plt.subplot(2, 5, i+1)
    plt.imshow(image)
plt.show()