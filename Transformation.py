
import numpy as np
import math



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
    Mvp[2, :] = [0., 0., 1., 0.]
    Mvp[3, :] = [0., 0., 0., 1.]
    Mvp.reshape((4, 4))
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

def M_projection(Z_near, Z_far, camera, width, height):
    '''
    First, create the frustum ; Second, calculate and return the matrix
    @param model: using to calculate and create the frustum
    @return: projection matrix
    '''
    t = -Z_near * math.tan(camera.horizon_angle)
    b = -t
    r = t * (width/height)
    l = -r
    M_po = M_p2orth(Z_near, Z_far)
    M_o = M_orth(l, r, t, b, Z_near, Z_far)
    return M_o @ M_po

class camera:
    def __init__(self, camera_position, camera_look_at, camera_up, horizon_angle):
        self.camera_position = camera_position
        self.look_at = camera_look_at
        self.camera_up = camera_up
        self.horizon_angle = horizon_angle

class Transformation():
    def __init__(self, camera, width, height):
        super(Transformation, self).__init__()
        M_cam = np.array(M_camera(camera.camera_position, camera.look_at, camera.camera_up))
        M_view = np.array(M_viewport(width, height))
        '''下面默认朝向【0， 0， -1】如果不是这样，需要调整！'''
        near = math.sqrt(pow(camera.camera_position[0], 2) + pow(camera.camera_position[1], 2) +\
                         pow(camera.camera_position[2], 2))
        near = 1
        far = 100
        M_project = np.array(M_projection(-near, -far, camera, width, height))
        self.M = M_view @ M_project @ M_cam


    def execute(self, mesh):
        M = self.M
        vertices = mesh.vertices
        nb = vertices.shape[0]
        nv = vertices.shape[1]
        for i in range(0, nb):
            for j in range(0, nv):
                vertices[i, j, :] = M @ vertices[i, j, :]
                vertices[i, j, :] = vertices[i, j, :]/vertices[i, j, 3]
        mesh.vertices = vertices
        mesh.get_face_vertices()
        return mesh




