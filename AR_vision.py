from numpy import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import pygame, pygame.image
from pygame.locals import *
import pickle

width, height = 2560, 1440
def set_projection_from_camera(K, p=[0, 0, 0]):
    """ Set view from a camera calibration matrix. """

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    fx = K[0, 0]
    fy = K[1, 1]
    fovy = 2 * arctan(0.5 * height / fy) * 180 / pi
    #    aspect= float(width*fy)/(height*fx)
    aspect = (width * fy) / (height * fx)

    # define the near and far clipping planes
    near = 0.1
    far = 100.0

    # set perspective
    gluPerspective(fovy, aspect, near, far)
    glViewport(p[0], p[1], width, height)


def set_modelview_from_camera(Rt):
    """ Define the camera modelview matrix from camera postion """

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # rotate teapot 90 deg around x-axis so that z-axis is up
    Rx = array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

    # set rotation to best approximation
    R = Rt[:, :3]
    U, S, V = linalg.svd(R)
    R = dot(U, V)
    R[0, :] = -R[0, :]  # change sign of x-axis

    # set translation
    t = Rt[:, 3]

    # setup 4*4 matrix view matrix
    M = eye(4)
    M[:3, :3] = dot(R, Rx)
    M[:3, 3] = t

    # transpose and flatten to get column order
    M = M.T
    m = M.flatten()

    # replace model view with the new matrix
    glLoadMatrixf(m)


def draw_background(imname):
    """ Draw background image using a quad.  """

    # load background image (should be .bmp) to OpenGL texture
    bg_image = pygame.image.load(imname).convert()
    bg_data = pygame.image.tostring(bg_image, "RGBX", 1)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # bind the texture
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, glGenTextures(1))
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, bg_data)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

    # create quad to fill the whole window
    glBegin(GL_QUADS)
    glTexCoord2f(0.0, 0.0);
    glVertex3f(-1.0, -1.0, -1.0)
    glTexCoord2f(1.0, 0.0);
    glVertex3f(1.0, -1.0, -1.0)
    glTexCoord2f(1.0, 1.0);
    glVertex3f(1.0, 1.0, -1.0)
    glTexCoord2f(0.0, 1.0);
    glVertex3f(-1.0, 1.0, -1.0)
    glEnd()

    # clear the texture
    glDeleteTextures(1)


def draw_teapot(size):
    """ Draw a red teapot at the origin. """
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_DEPTH_TEST)
    glClear(GL_DEPTH_BUFFER_BIT)

    # draw red teapot
    glMaterialfv(GL_FRONT, GL_AMBIENT, [0, 0, 0, 0])
    glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.5, 0.0, 0.0, 0.0])
    glMaterialfv(GL_FRONT, GL_SPECULAR, [0.7, 0.6, 0.6, 0.0])
    glMaterialf(GL_FRONT, GL_SHININESS, 0.25 * 128.0)
    glutInit()
    glutSolidTeapot(size)


def setup():
    """ Setup window and pygame environment. """

    pygame.init()
    window = pygame.display.set_mode((width, height), OPENGL | DOUBLEBUF)
    pygame.display.set_caption('OpenGL AR demo')
    return window


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

"""*********************************************************************************"""
import pygame
from OpenGL.GL import *

def MTL(filename):
    contents = {}
    mtl = None
    for line in open(filename, "r"):
        if line.startswith('#'): continue
        values = line.split()
        if not values: continue
        if values[0] == 'newmtl':
            mtl = contents[values[1]] = {}
        elif mtl is None:
            raise (ValueError, "mtl file doesn't start with newmtl stmt")
        elif values[0] == 'map_Kd':
            # load the texture referred to by this declaration
            mtl[values[0]] = values[1]
            surf = pygame.image.load(mtl['map_Kd'])
            image = pygame.image.tostring(surf, 'RGBA', 1)
            ix, iy = surf.get_rect().size
            texid = mtl['texture_Kd'] = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texid)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0, GL_RGBA,
                GL_UNSIGNED_BYTE, image)
        else:
            mtl[values[0]] = map(float, values[1:])
    return contents

class OBJ:
    def __init__(self, filename, swapyz=False):
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []

        material = None
        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                v = map(float, values[1:4])
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == 'vn':
                v = map(float, values[1:4])
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == 'vt':
                self.texcoords.append(map(float, values[1:3]))
            elif values[0] in ('usemtl', 'usemat'):
                material = values[1]
            elif values[0] == 'mtllib':
                self.mtl = MTL(values[1])
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                self.faces.append((face, norms, texcoords, material))

        self.gl_list = glGenLists(1)
        glNewList(self.gl_list, GL_COMPILE)
        glEnable(GL_TEXTURE_2D)
        glFrontFace(GL_CCW)
        for face in self.faces:
            vertices, normals, texture_coords, material = face

            mtl = self.mtl[material]
            if 'texture_Kd' in mtl:
                # use diffuse texmap
                glBindTexture(GL_TEXTURE_2D, mtl['texture_Kd'])
            else:
                # just use diffuse colour
                glColor(*mtl['Kd'])

            glBegin(GL_POLYGON)
            for i in range(len(vertices)):
                if normals[i] > 0:
                    glNormal3fv(self.normals[normals[i] - 1])
                if texture_coords[i] > 0:
                    glTexCoord2fv(self.texcoords[texture_coords[i] - 1])
                glVertex3fv(self.vertices[vertices[i] - 1])
            glEnd()
        glDisable(GL_TEXTURE_2D)
        glEndList()
"""******************************************************************************************"""


def load_and_draw_model(filename):
    """ Loads a model from an .obj file using objloader.py.
    Assumes there is a .mtl material file with the same name. """
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_DEPTH_TEST)
    glClear(GL_DEPTH_BUFFER_BIT)

    # set model color
    glMaterialfv(GL_FRONT, GL_AMBIENT, [0.1, 0.1, 0.1, 0])
    glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.5, 0.75, 1.0, 0.0])
    glMaterialf(GL_FRONT, GL_SHININESS, 0.25 * 128.0)

    glEnable(GL_NORMALIZE)
    glScalef(0.0085, 0.0085, 0.0085)
    # load from file
    # import objloader
    obj = OBJ(filename)
    glCallList(obj.gl_list)

#
# # load camera data
# with open('ar_camera.pkl', 'rb') as f:
#     K = pickle.load(f)
#     Rt = pickle.load(f)

# wood.jpg is downloaded from http://www.oyonale.com/modeles.php and converted to wood.png
# Color is brightened up
# Material is defined in wood.mtl
# Change toyplane.obj to refer to wood.mtl and replace all materials with wood
# Color is a bit off. Need to fix


# load camera data
with open('ar_camera.pkl', 'r') as f:
    K = pickle.load(f)
    Rt = pickle.load(f)

# wood.jpg is downloaded from http://www.oyonale.com/modeles.php and converted to wood.png
# Color is brightened up
# Material is defined in wood.mtl
# Change toyplane.obj to refer to wood.mtl and replace all materials with wood
# Color is a bit off. Need to fix


try:
    window = setup()
    draw_background('imagetes2.jpg')

    #draw_background('image_test3.jpg')
    set_projection_from_camera(K)
    set_modelview_from_camera(Rt)
    draw_teapot(0.02)
    pygame.image.save(window, "screenshot1.jpeg")

    while True:
        event = pygame.event.poll()
        if event.type in (QUIT, KEYDOWN):
            break
        pygame.display.flip()

finally:
    pygame.display.quit()

# load camera data
# with open('ar_camera.pkl', 'rb') as f:
#     K = pickle.load(f)
#     Rt = pickle.load(f)

# wood.jpg is downloaded from http://www.oyonale.com/modeles.php and converted to wood.png
# Color is brightened up
# Material is defined in wood.mtl
# Change toyplane.obj to refer to wood.mtl and replace all materials with wood
# Color is a bit off. Need to fix

# try:
#     setup()
#     draw_background('imagetes2.jpg')
#     set_projection_from_camera(K)
#     set_modelview_from_camera(Rt)
#     load_and_draw_model('toyplane.obj')
#
#     while True:
#         event = pygame.event.poll()
#         if event.type in (QUIT, KEYDOWN):
#             break
#         pygame.display.flip()
# finally:
#     pygame.display.quit()

# import matplotlib.pyplot as plt
# import numpy as np
# t = np.arange(10)
# plt.plot(t, np.sin(t))
# print("Please click")
# x = plt.ginput(3)
# print("clicked", x)
# plt.s