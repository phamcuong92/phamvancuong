from PIL import Image
from numpy import *
from pylab import *
import numpy as np

from numpy import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import pygame, pygame.image
from pygame.locals import *
import pickle

import camera, homography
from ref import sift

# def my_calibration(sz):
#     """take a size tuple and returns the calibration matrix
#     Here we assume the optical center to be the center of the image
#     image resolution: 2592x1936 pixels
#     fx = 2555
#     fy = 2586
#     """
#     row,col = sz
#     fx = 2555*col/2592
#     fy = 2586*row/1936
#     K = np.diag([fx,fy,1])
#     K[0,2] = 0.5*col
#     K[1,2] = 0.5*row
#     return K

def my_calibration(sz):
    """take a size tuple and returns the calibration matrix
    Here we assume the optical center to be the center of the image
    image resolution: x1936 pixels
    dX = 185
    dY = 240
    dZ = 297
    dx = 1178.217
    dy = 1585.465
    fx = 1891.5
    fy = 1962
    """
    row,col = sz
    fx = 1891*col/1440
    fy= 1962*row/2560
    # fx = 1962*col/1440
    # fy= 1891*row/2560
    K = np.diag([fx,fy,1])
    K[0,2] = 0.5*col
    K[1,2] = 0.5*row
    return K


def cube_points(c, wid):
    """ Creates a list of points for plotting
        a cube with plot. (the first 5 points are
        the bottom square, some sides repeated). """
    p = []
    # bottom
    p.append([c[0] - wid, c[1] - wid, c[2] - wid])
    p.append([c[0] - wid, c[1] + wid, c[2] - wid])
    p.append([c[0] + wid, c[1] + wid, c[2] - wid])
    p.append([c[0] + wid, c[1] - wid, c[2] - wid])
    p.append([c[0] - wid, c[1] - wid, c[2] - wid])  # same as first to close plot

    # top
    p.append([c[0] - wid, c[1] - wid, c[2] + wid])
    p.append([c[0] - wid, c[1] + wid, c[2] + wid])
    p.append([c[0] + wid, c[1] + wid, c[2] + wid])
    p.append([c[0] + wid, c[1] - wid, c[2] + wid])
    p.append([c[0] - wid, c[1] - wid, c[2] + wid])  # same as first to close plot

    # vertical sides
    p.append([c[0] - wid, c[1] - wid, c[2] + wid])
    p.append([c[0] - wid, c[1] + wid, c[2] + wid])
    p.append([c[0] - wid, c[1] + wid, c[2] - wid])
    p.append([c[0] + wid, c[1] + wid, c[2] - wid])
    p.append([c[0] + wid, c[1] + wid, c[2] + wid])
    p.append([c[0] + wid, c[1] - wid, c[2] + wid])
    p.append([c[0] + wid, c[1] - wid, c[2] - wid])

    return np.array(p).T

# Sift feature

#imname1 = 'book_frontal.JPG'
imname1 = 'image_test_opencv.jpg'

sift.process_image(imname1, 'im1.sift')
l0, d0 = sift.read_features_from_file('im1.sift')
l1 = []
d1 = []
for i in range(l0.shape[0]):
    x, y, s, w = l0[i]
    if (413<x<2030 and 219<y<1398):
        l1.append(l0[i])
        d1.append(d0[i])
l1 = array(l1)
d1 = array(d1)
im1 = array(Image.open(imname1))
print(im1.shape)
figure()
imshow(im1)
sift.plot_features(im1, l1, circle=True)
#show()

#imname2 = 'book_perspective.JPG'
#imname2 = 'image_test3.jpg'
imname2 = 'imagetes2.jpg'

sift.process_image(imname2, 'im2.sift')
l2, d2 = sift.read_features_from_file('im2.sift')

im2 = array(Image.open(imname2))
print(im2.shape)
#figure()
#imshow(im2)
#sift.plot_features(im2, l2, circle=True)
#show()

# match features and estimate homography

# im1 = array(Image.open(imname1))
# im2 = array(Image.open(imname2))
# print(im1.shape)
# print(im2.shape)
# figure()
# subplot(1, 2, 1)
# imshow(im1)
# sift.plot_features(im1, l1, circle=True)
# subplot(1, 2, 2)
# imshow(im2)
# sift.plot_features(im2, l2, circle=True)
# show()

matches = sift.match_twosided(d1, d2)
ndx = matches.nonzero()[0]
fp = homography.make_homog(l1[ndx,:2].T)
ndx2 = [int(matches[i]) for i in ndx]
tp = homography.make_homog(l2[ndx2, :2].T)

figure(figsize=(16, 16))
gray()
sift.plot_matches(im1, im2, l1, l2, matches)
#show()

model = homography.RansacModel()
H, inliers = homography.H_from_ransac(fp,tp,model)

# Camera Calibration
# camera calibration
K = my_calibration((2560,1440))

# 3D points at plane z=0 with sides of length 0.2
box = cube_points([0,0,0.1],0.1)

# project bottom square in first image
cam1 = camera.Camera( hstack((K,dot(K,array([[0],[0],[-1]])) )) )
# first points are the bottom square
box_cam1 = cam1.project(homography.make_homog(box[:,:5]))

# use H to transfer points to the second image
# box_trans = homography.normalize(dot(H,box_cam1))
#
# # compute second camera matrix from cam1 and H
# cam2 = camera.Camera(dot(H,cam1.P))
# A = dot(linalg.inv(K),cam2.P[:,:3])
# A = array([A[:,0],A[:,1],cross(A[:,0],A[:,1])]).T
# cam2.P[:,:3] = dot(K,A)
# # project with the second camera
# box_cam2 = cam2.project(homography.make_homog(box))

box_trans = homography.normalize(dot(H,box_cam1))
# compute second camera matrix from cam1 and H
cam2 = camera.Camera(dot(H,cam1.P))
A = dot(linalg.inv(K),cam2.P[:,:3])
A = array([A[:,0],A[:,1],cross(A[:,0],A[:,1])]).T
cam2.P[:,:3] = dot(K,A)
# project with the second camera
box_cam2 = cam2.project(homography.make_homog(box))

# test: projecting point on z=0 should give the same
point = array([1,1,0,1]).T
print(homography.normalize(dot(dot(H,cam1.P),point)))
print(cam2.project(point))

figure()
imshow(im1)
plot(box_cam1[0,:],box_cam1[1,:],linewidth=3)
title('2D projection of bottom square')
axis('off')

figure()
imshow(im2)
plot(box_trans[0,:],box_trans[1,:], linewidth=3)
title('2D projection transfered with H')
axis('off')

figure()
imshow(im2)
plot(box_cam2[0,:],box_cam2[1,:],linewidth=3)
title('3D points projected in second image')
axis('off')
show()

import pickle
with open('ar_camera.pkl','w') as f:
    pickle.dump(K,f)
    pickle.dump(np.dot(np.linalg.inv(K),cam2.P),f)

# #load camera data
# with open('ar_camera.pkl', 'r') as f:
#     K = pickle.load(f)
#     Rt = pickle.load(f)
#
# # Augmented Reality
# from numpy import *
# from OpenGL.GL import *
# from OpenGL.GLU import *
# from OpenGL.GLUT import *
# import pygame, pygame.image
# from pygame.locals import *
# import pickle