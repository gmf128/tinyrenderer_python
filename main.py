# -*- coding: utf-8 -*-
import jittor as jt
import numpy
import numpy as np
import math
import tqdm
import cv2 as cv
import Mesh, soft_rasterize_cuda, Transformation, SoftRasterizer, Lighting
import time
import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from numba import cuda
import time
import numpy as np
import math
import imageio



def create_img():
    '''
    create a window to show image
    :return:
    '''

    height = 512
    width = 512
    RGB = 3
    image = np.ndarray((height, width, RGB), np.uint8)
    for i in range(0, height):
        for j in range(0, width):
            image[i, j, :] = [255, 0, 0]
    for i in range(0, int(height/2)):
        for j in range(0, int(width/2)):
            image[i, j, :] = [0, 255, 0]
    image[0, 0, :] = [255, 255, 255]
    cv. imwrite("textureimage.png", image)


def show_image(image_path='test.png'):
    app = QtWidgets.QApplication(sys.argv)
    w = QWidget()
    w.label = QLabel(w)
    w.label.setText("   显示图片")
    w.label.setFixedSize(1152, 1152)
    w.label.move(0, 0)
    w.label.setStyleSheet("QLabel{background:white;}"
                             "QLabel{color:rgb(300,300,300,120);font-size:10px;font-weight:bold;font-family:宋体;}"
                             )
    # resize()方法调整窗口的大小。这离是250px宽150px高
    w.resize(1152, 1152)
    # move()方法移动窗口在屏幕上的位置到x = 300，y = 300坐标。
    w.move(0, 0)
    # 设置窗口的标题
    w.setWindowTitle('3D Viewer')
    jpg = QtGui.QPixmap(image_path).scaled(w.label.width(), w.label.height())
    w.label.setPixmap(jpg)
    # 显示在屏幕上
    w.show()
    sys.exit(app.exec_())


width = 1024
height = 1024
RGB = 3
RGBA = 4
GRAY = 1
light = np.array([0., 0., -1.])
mesh = Mesh.MeshModel('C:\\Users\\314\\Desktop\\jittorRenderer\\obj\\spot_triangulated.obj', texture_exist=True, texture_res=8)
def render(theta, mesh, gamma):
    # 初始化
    start = time.time()
    '''
    camera_position = np.array([-2.*math.sin(math.radians(theta)), 0., -2.*math.cos(math.radians(theta))])
    camera_look_at = np.array([math.sin(math.radians(theta)), 0., math.cos(math.radians(theta))])
    camera_up_dir = np.array([0., 1., 0.])
    camera_angle = math.radians(50)
    '''
    camera_position = np.array([0., 0., 2.])
    camera_look_at = np.array([0., 0., -1.])
    camera_up_dir = np.array([0., 1., 0.])
    camera_angle = math.radians(50)
    # triangle: 3,4,5 ; -1,-1,-1; 2,0-2
    camera = Transformation.camera(camera_position, camera_look_at, camera_up_dir, camera_angle)
    image = np.ndarray((height, width, RGB), np.uint8)

    lighting = Lighting.Lighting()
    soft_rasterizer = SoftRasterizer.SoftRasterizer(aggr_func_alpha='hard', width=width, height=height, gamma_val=gamma)
    transformation = Transformation.Transformation(camera, width, height)
    mesh = lighting.execute(mesh)
    mesh = transformation.execute(mesh)
    image = soft_rasterizer.execute(mesh)[0, :, :, 0:3].astype(np.uint8)
    center = (width // 2, height // 2)
    M = cv.getRotationMatrix2D(center, -90, 1)
    image = cv.warpAffine(image, M, (width, height))
    cv.imwrite("test.png", cv.flip(image, 0)) #cv.flip(image, 0))

    return image


def frame_to_gif(frame_list):
    gif = imageio.mimsave('./result.gif', frame_list, 'GIF', duration=0.00085)  # 0.00085


if __name__ == '__main__':
    start = time.time()
    framelist = []
    '''
    loop = tqdm.tqdm(range(0, 5))
    loop.set_description('Drawing rotation')
    for i in loop:
        framelist.append(render(0, mesh, math.pow(10, -i))[::-1, :, ::-1])

    frame_to_gif(framelist)
    #显示窗口
    end = time.time()
    '''
    #print("execute: ", end - start, "s")
    render(0, mesh, 1e-5)
    show_image()




