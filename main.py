# -*- coding: utf-8 -*-
import Render
import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

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
    #resize()方法调整窗口的大小。这离是250px宽150px高
    w.resize(1152, 1152)
    #move()方法移动窗口在屏幕上的位置到x = 300，y = 300坐标。
    w.move(0, 0)
    #设置窗口的标题
    w.setWindowTitle('3D Viewer')
    jpg = QtGui.QPixmap(image_path).scaled(w.label.width(), w.label.height())
    w.label.setPixmap(jpg)
    #显示在屏幕上
    w.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    Renderer = Render.Renderer()
    Renderer.render()
    #显示窗口
    show_image()


'''进行图片属性定义
'''

'''
***打开已经存在的文件：
image = cv.imread("D:\\test code\\tinyrender\images\output.png")
cv.imshow("image", image)
cv.waitKey(0)
cv.destroyAllWindows()

***保存生成的图片文件
cv.imwrite("test.png",image)

***对像素值进行操作，本质上是对数组进行操作
   image = np.ndarray((height,width,RGB),np.uint8
   for i in range(0,height):
    for j in range(0,width):
        image[i,j] = [0,0,0]
        
** 旋转操作 说不定将来有用呢
center = (width//2, height//2)
M = cv.getRotationMatrix2D(center, 90, 1)
cv.warpAffine(image, M, (width, height))

**单纯地绘制点
for vertex in model.vertices:
    screen_result = np.transpose(geo.M_viewport(width, height)@(vertex.get_cord()))
    screen_cord = screen_result[0, 0:2]
    image[int(screen_cord[1]), int(screen_cord[0])] = [255, 255, 255]
    zbuffer[int(screen_cord[1]), int(screen_cord[0])] = [int(screen_result[0, 2])]
'''
'''opencv采用的是BGR算法


'''