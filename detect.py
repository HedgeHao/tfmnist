import cv2
import numpy as np
import tensorflow as tf

np.set_printoptions(suppress=True)

model = tf.saved_model.load('mnist_saved_model')

img = cv2.imread('images/three.jpg')
cv2.resize(img, (28, 28), img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = img / 255

print(img.shape)
predictions = model([img])
print(tf.nn.softmax(predictions))
