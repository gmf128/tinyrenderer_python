import time

import numba
from numba import jit, float64
import cv2 as cv
import numpy as np
import Geometry as geo
import Model
import Shader
import math


def sigmoid(inx):
    if inx >= 0:  # 对sigmoid函数的优化，避免了出现极大的数据溢出
        return 1.0 / (1 + np.exp(-inx))
    else:
        return np.exp(inx) / (1 + np.exp(inx))

def world_to_screen(vert_cord, M):
    return np.transpose(M @ (vert_cord)) # 1*4 向量



class Renderer :
    # 参数设定

    width = 32
    height = 32
    RGB = 3
    RGBA = 4
    GRAY = 1
    light = np.array([0., 0., -1.])
    camera_position = np.array([0, 0, 3])
    camera_look_at = geo.normalize(np.array([0, 0, -1]))
    camera_up_dir = geo.normalize(np.array([0, 1, 0]))
    camera_angle = math.radians(50)
    camera = Shader.Camera(camera_position, camera_look_at, camera_up_dir, camera_angle)
    triangles = []
    model = Model.MeshModel
    def __init__(self):

        self.model = Model.MeshModel("african_head.obj")

    def render(self):
        # 初始化
        start = time.time()
        model = self.model
        image = np.ndarray((self.height, self.width, self.RGB), np.uint8)
        zbuffer = np.ndarray((self.height, self.width, self.GRAY), np.uint8)
        shader = Shader.PhongShader(self.light)
        for i in range(0, self.height):
            for j in range(0, self.width):
                image[i, j] = [0, 0, 0]
                zbuffer[i, j] = [0]
        print(model.num_of_vertices())
        print(model.num_of_faces())
        # 绘制
        # View Transformation
        M_transform = geo.M_viewport(self.width, self.height) @ geo.M_projection(model, self.camera, self.width, self.height) @ geo.M_camera(
            self.camera_position, self.camera_look_at, self.camera_up_dir)

        for face in model.faces:
            self.triangles.append(face)
            A = face.A
            B = face.B
            C = face.C
            scr_par_A = geo.homogenilized(world_to_screen(A.get_cord(), M_transform))
            scr_par_B = geo.homogenilized(world_to_screen(B.get_cord(), M_transform))
            scr_par_C = geo.homogenilized(world_to_screen(C.get_cord(), M_transform))
            A.set_screencord(scr_par_A[0, 0:2].reshape((1, 2)))  # 1*2 向量
            B.set_screencord(scr_par_B[0, 0:2].reshape((1, 2)))
            C.set_screencord(scr_par_C[0, 0:2].reshape((1, 2)))
            shader.triangle(scr_par_A, scr_par_B, scr_par_C, image, zbuffer)
            face.get_norm()

        pixel_cord = np.array([0, 0])
        Dij = []
        color = 0
        sigma = 1e-4
        delta = 0
        for i in range(0, self.width):
            for j in range(0, self.height):
                index = (i + 1) * (j + 1)
                pixel_cord[0] = i + 0.5
                pixel_cord[1] = j + 0.5
                for k in range(0, len(self.triangles)):
                    if self.triangles[k].cal_barycentric(pixel_cord[0], pixel_cord[1]):
                        delta = 1
                    else:
                        delta = -1
                    uvw = 1
                    for m in range(0, 3):
                       # uvw += math.pow(self.triangles[k].barycentric_cord[m], 2)
                        uvw *= abs(self.triangles[k].barycentric_cord[m])
                    dij = sigmoid(delta * uvw / sigma)
                    Dij.append(dij)
                    color *= 1 - dij
                color = 1 - color
                image[j, i] = [255 * color, 255 * color, 255 * color]
                color = 1

        for vertex in model.vertices:
            screen_result = geo.homogenilized(world_to_screen(vertex.get_cord(), M_transform))
            screen_cord = screen_result[0, 0:2]
            if int(screen_cord[1] + 1 / 2) < 0 or int(screen_cord[0] + 1 / 2) < 0:
                continue
            if int(screen_cord[1] + 1 / 2) > self.width or int(screen_cord[0] + 1 / 2) > self.height:
                continue
            image[int(screen_cord[1]), int(screen_cord[0])] = [0, 255, 0]
            zbuffer[int(screen_cord[1]), int(screen_cord[0])] = [int(screen_result[0, 2])]

        cv.imwrite("test.png", cv.flip(image, 0))
        cv.imwrite("zbuffer.png", cv.flip(zbuffer, 0))

        end = time.time()
        print("execute: ", end-start, "s")

'''
        for face in model.faces:
            A = face.A
            B = face.B
            C = face.C
            scr_par_A = self.world_to_screen(A.get_cord(), M_transform)
            scr_par_B = self.world_to_screen(B.get_cord(), M_transform)
            scr_par_C = self.world_to_screen(C.get_cord(), M_transform)
            shader.triangle(scr_par_A, scr_par_B, scr_par_C, image, zbuffer)
            
            
        pixel_cord = np.array([0, 0])
        Dij = []
        color = 0
        sigma = 1e-4
        delta = 0
        for i in range(0, self.width):
            for j in range(0, self.height):
                index = (i+1)*(j+1)
                pixel_cord[0] = i+0.5
                pixel_cord[1] = j+0.5
                for k in range(0, len(self.triangles)):
                    if self.triangles[k].cal_barycentric(pixel_cord[0], pixel_cord[1]):
                        delta = 1
                    else:
                        delta = -1
                    uvw = 1
                    for m in range(0, 3):
                        uvw *= abs(self.triangles[k].barycentric_cord[m])
                    dij = sigmoid(delta*uvw/sigma)
                    Dij.append(dij)
                    color *= 1 - dij
                color = 1 - color
                image[j, i] = [255*color, 255*color, 255*color]
                color = 1
                
                
        for vertex in model.vertices:
            screen_result = geo.homogenilized(world_to_screen(vertex.get_cord(), M_transform))
            screen_cord = screen_result[0, 0:2]
            if int(screen_cord[1] + 1 / 2) < 0 or int(screen_cord[0] + 1 / 2) < 0:
                continue
            if int(screen_cord[1] + 1 / 2) > self.width or int(screen_cord[0] + 1 / 2) > self.height:
                continue
            image[int(screen_cord[1]), int(screen_cord[0])] = [0, 255, 0]
            zbuffer[int(screen_cord[1]), int(screen_cord[0])] = [int(screen_result[0, 2])]
'''

