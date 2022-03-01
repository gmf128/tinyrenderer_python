import cv2 as cv
import numpy as np
import Geometry as geo
import Model
import Shader
import math
#参数设定

width = 512
height = 512
RGB = 3
RGBA = 4
GRAY = 1
light = np.array([0., 0., -1.])
camera_position = np.array([1, 1, 1])
camera_look_at = geo.normalize(np.array([-1, -1, -1]))
camera_up_dir = geo.normalize(np.array([2, 0, -2]))
camera_angle = math.radians(50)
camera = Shader.Camera(camera_position, camera_look_at, camera_up_dir, camera_angle)

#绘制函数
#1.视图变换函数
def world_to_screen(vert_cord, M):
    return np.transpose(M@(vert_cord)
    )


#初始化
image = np.ndarray((height, width, RGB), np.uint8)
zbuffer = np.ndarray((height, width, GRAY), np.uint8)
shader = Shader.PhongShader(light)
for i in range(0,height):
    for j in range(0, width):
        image[i, j] = [0, 0, 0]
        zbuffer[i, j] = [0]
model = Model.MeshModel("african_head.obj")
print(model.num_of_vertices())
print(model.num_of_faces())
#绘制
#View Transformation
M_transform = geo.M_viewport(width, height) @ geo.M_projection(model, camera, width, height) @ geo.M_camera(
        camera_position, camera_look_at, camera_up_dir)

for face in model.faces:
    A = face.A
    B = face.B
    C = face.C
    scr_par_A = world_to_screen(A.get_cord(), M_transform)
    scr_par_B = world_to_screen(B.get_cord(), M_transform)
    scr_par_C = world_to_screen(C.get_cord(), M_transform)
    shader.triangle(scr_par_A, scr_par_B, scr_par_C, image, zbuffer)


for vertex in model.vertices:
    screen_result = geo.homogenilized(world_to_screen(vertex.get_cord(), M_transform))
    #print(screen_result)
    screen_cord = screen_result[0, 0:2]
    if int(screen_cord[1]+1/2) < 0 or int(screen_cord[0]+1/2) < 0:
        continue
    if int(screen_cord[1]+1/2) > width or int(screen_cord[0]+1/2) > height:
        continue
    image[int(screen_cord[1]), int(screen_cord[0])] = [255, 255, 255]
    zbuffer[int(screen_cord[1]), int(screen_cord[0])] = [int(screen_result[0, 2])]


cv.imwrite("test.png", cv.flip(image, 0))
cv.imwrite("zbuffer.png", cv.flip(zbuffer, 0))


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