import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from numpy import *
from pylab import *

imname1 = 'image_test_opencv.jpg'
im1 = array(Image.open(imname1))
print(im1.shape)
figure()
imshow(im1)
# sift.plot_features(im1, l1, circle=True)
# show()
print("Please click")
x = plt.ginput(10)
print("clicked", x)
plt.show()