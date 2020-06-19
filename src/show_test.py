import cv2

image = cv2.imread('../input/T91/t1.png')
image = cv2.resize(image, (33, 33))
cv2.imshow('image', image)
cv2.waitKey(0)