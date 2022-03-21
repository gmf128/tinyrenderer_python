import math

import numpy as np

#视图变换矩阵

def homogenilized(v):
    '''
    齐次坐标归一化
    @param v:
    @return:
    '''
    return v/v[0, 3]

def normalize(v):
    '''

    @param v: vector
    @return: v_normalized
    '''
    return v/np.linalg.norm(v)

def M_viewport(nx, ny):
    '''

    @param nx: the width of pictire(along x_cord)
    @param ny: the height of picture(along y-cord)
    @return:   Mvp: View-port Matrix
    '''
    Mvp = np.zeros((4, 4), np.float64)
    Mvp[0, :] = [nx/2, 0., 0., (nx-1)/2]
    Mvp[1, :] = [0., ny/2, 0., (ny-1)/2]
    Mvp[2, :] = [0., 0., -255/2, 127.]
    Mvp[3, :] = [0., 0., 0., 1.]
    Mvp.reshape((4,4))
    return Mvp

def M_camera(position, look_at, up):
    M_move = np.zeros((4, 4), np.float64)
    M_move[:, 0] = [1, 0, 0, 0]
    M_move[:, 1] = [0, 1, 0, 0]
    M_move[:, 2] = [0, 0, 1, 0]
    M_move[:, 3] = [-position[0], -position[1], -position[2], 1]
    third_dir = normalize(np.cross(look_at, up))
    M_rot = np.zeros((4, 4), np.float64)
    M_rot[:, 0] = [third_dir[0], third_dir[1], third_dir[2], 0]
    M_rot[:, 1] = [up[0], up[1], up[2], 0]
    M_rot[:, 2] = [-look_at[0], -look_at[1], -look_at[2], 0]
    M_rot[:, 3] = [0, 0, 0, 1]
    M_cam = np.transpose(M_rot)@M_move
    return M_cam

def M_orth(l, r, t, b, n, f):
    M_orth = np.zeros((4, 4))
    M_orth[:, 0] = [2/(r-l), 0, 0, 0]
    M_orth[:, 1] = [0, 2/(t-b), 0, 0]
    M_orth[:, 2] = [0, 0, 2/(n-f), 0]
    #M_orth[:, 3] = [-(r+l)/(r-l), -(t+b)/(t-b), -(n+f)/(n-f), 1]
    M_orth[:, 3] = [0, 0, -(n + f) / (n - f), 1]
    return M_orth

def M_p2orth(n, f):
    M_p2orth = np.zeros((4, 4))
    M_p2orth[:, 0] = [n, 0, 0, 0]
    M_p2orth[:, 1] = [0, n, 0, 0]
    M_p2orth[:, 2] = [0, 0, n+f, 1]
    M_p2orth[:, 3] = [0, 0, -f*n, 0]
    return M_p2orth

def M_projection(model, camera, width, height):
    '''
    First, create the frustum ; Second, calculate and return the matrix
    @param model: using to calculate and create the frustum
    @return: projection matrix
    '''
    Z_near = -float("inf")
    Z_far = 0
    for vertex in model.vertices:
        M = M_camera(camera.camera_position,camera.camera_look_at,camera.camera_up_dir)
        result = M@vertex.get_cord()
        Z = result[2, 0]
        if Z < Z_far:
            Z_far = Z
        if Z > Z_near and Z < 0 :
            Z_near = Z
    t = -Z_near * math.tan(camera.camera_angle)
    b = -t
    r = t * (width/height)
    l = -r
    M_po = M_p2orth(Z_near, Z_far)
    M_o = M_orth(l, r, t, b, Z_near, Z_far)
    return M_o @ M_po
#几何对象
class vertex:
    #成员变量
    cord = np.zeros((4, 1), np.float64)  # 4*1 的向量
    norm = np.array(3, np.float64)
    uv = np.array(2, np.float64)
    screen_cord = np.zeros((1, 2), np.float64)  # 1*2 向量
    #成员函数
    def __init__(self, x, y, z):
        self.cord = np.array([x, y, z, 1], np.float64).reshape((4, 1))

    def get_cord(self):
        '''
        @return: self.cord
        '''
        return self.cord

    def set_norm(self, norm_cord):
        self.norm = norm_cord

    def get_norm(self):
        return self.norm

    def set_uv(self, uv_cord):
         self.uv = uv_cord

    def get_uv(self):
        return self.uv

    def set_screencord(self, screen_cord):  # 2*1 array
        self.screen_cord = screen_cord

class face:
    #成员变量
    A = vertex(0, 0, 0)
    B = vertex(0, 0, 0)
    C = vertex(0, 0, 0)
    norm = np.array(3, np.float64)
    barycentric_cord = [0., 0., 0.]

    def __init__(self, v1, v2, v3):
        self.vertex_set = [v1, v2, v3]
        self.A = v1
        self.B = v2
        self.C = v3

    def get_norm(self):
        '''
        计算出面的法向量
        @return: 面的法向量
        '''
        AB = self.A.cord[0:3, 0] - self.B.cord[0:3, 0]    #3*1的向量
        AC = self.A.cord[0:3, 0] - self.C.cord[0:3, 0]    #3*1的向量
        tempnorm = np.cross(AB, AC)
        self.norm = normalize(tempnorm)
        return self.norm

    def cal_barycentric(self, x, y):
        AB = self.B.screen_cord - self.A.screen_cord
        AC = self.C.screen_cord - self.A.screen_cord
        s = np.cross(AB, AC)
        P_cord = np.zeros((1, 2))
        P_cord[0, 0] = x
        P_cord[0, 1] = y
        PA = self.A.screen_cord - P_cord
        PB = self.B.screen_cord - P_cord
        PC = self.C.screen_cord - P_cord
        u = np.cross(PB, PC)/s
        v = np.cross(PC, PA)/s
        w = np.cross(PA, PB)/s
        self.barycentric_cord[0] = u
        self.barycentric_cord[1] = v
        self.barycentric_cord[2] = w
        if u > 0 and v > 0 and w > 0:
            return True
        else:
            return False
