import numpy as np
import cv2
import Geometry as geo
import math
class Camera:
    camera_position = np.array([0, 0, 0])
    camera_look_at = geo.normalize(np.array([0, 0, -1]))
    camera_up_dir = geo.normalize(np.array([0, 1, 0]))
    camera_angle = math.radians(50)
    def __init__(self, camera_position, camera_look_at, camera_up_dir, camera_angle):
        self.camera_position = camera_position
        self.camera_look_at = camera_look_at
        self.camera_up_dir = camera_up_dir
        self.camera_angle = camera_angle

class Shader:
    light = np.zeros((3, 1))
    def __init__(self, light):
        self.light = light

    def triangle(self, scr_cord_A, scr_cord_B, scr_cord_C, image, zbuffer):
        return

class FlatShader(Shader):
    def triangle(self, scr_cord_A, scr_cord_B, scr_cord_C, image, zbuffer):
       '''
     @todo
       @param scr_cord_A: vertex A 齐次坐标 (after transformation)
       @param scr_cord_B: vertex B 齐次坐标 (after transformation)
       @param scr_cord_C: vertex C 齐次坐标 (after transformation)
       @param image: image
       @param zbuffer: zbuffer
       @return: void
       '''
    # calculate bounding box
       return
class GouraudShader(Shader):
    def triangle(self, scr_cord_A, scr_cord_B, scr_cord_C, image, zbuffer):
        '''
     @todo
        @param scr_cord_A: vertex A 齐次坐标 (after transformation)
        @param scr_cord_B: vertex B 齐次坐标 (after transformation)
        @param scr_cord_C: vertex C 齐次坐标 (after transformation)
        @param image:
        @param zbuffer:
        @return:
        '''
        # calculate bounding box
        return

class PhongShader(Shader):
    def triangle(self, scr_cord_A, scr_cord_B, scr_cord_C, image, zbuffer):
       '''
    @todo
       @param scr_cord_A: vertex A 齐次坐标 (after transformation)
       @param scr_cord_B: vertex B 齐次坐标 (after transformation)
       @param scr_cord_C: vertex C 齐次坐标 (after transformation)
       @param image:
       @param zbuffer:
       @return:
       '''
       #     # calculate boundox
       return