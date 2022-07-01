import sys

import numpy as np


import soft_rasterize_cuda



class SoftRasterizeFunction():


    def execute(self, face_vertices, textures, width=256, height=256,
                background_color=[0, 0, 0], near=1, far=100,
                fill_back=True, eps=1e-3,
                sigma_val=1e-5, dist_func='barycentic', dist_eps=1e-4,
                gamma_val=1e-4, aggr_func_rgb='softmax', aggr_func_alpha='prod',
                texture_type='surface'):

        # face_vertices: [batch size, number of faces, 3, 4]
        # textures: [nb, nf, res**2, 3]

        func_dist_map = {'hard': 0, 'barycentric': 1, 'euclidean': 2}
        func_rgb_map = {'hard': 0, 'softmax': 1}
        func_alpha_map = {'hard': 0, 'sum': 1, 'prod': 2}
        func_map_sample = {'surface': 0, 'vertex': 1}

        self.width = width
        self.height = height
        self.background_color = background_color
        self.near = near
        self.far = far
        self.eps = eps
        self.sigma_val = sigma_val
        self.gamma_val = gamma_val
        self.func_dist_type = func_dist_map[dist_func]
        self.dist_eps = np.log(1. / dist_eps - 1.)
        self.func_rgb_type = func_rgb_map[aggr_func_rgb]
        self.func_alpha_type = func_alpha_map[aggr_func_alpha]
        self.texture_type = func_map_sample[texture_type]
        self.fill_back = fill_back
        self.batch_size, self.num_faces = face_vertices.shape[:2]

        faces_info = np.zeros(
            (self.batch_size, self.num_faces, 9*3),
            dtype='float32',
            )  # [inv*9, sym*9, obt*3, 0*6]
        aggrs_info = np.zeros(
            (self.batch_size, self.width, self.height, 2),
            dtype='float32',
            )
        if aggr_func_alpha == 'hard':
            soft_colors = np.ones(
                (self.batch_size, self.width, self.height, 3),
                dtype='float32',
            )
        else:
            soft_colors = np.ones(
            (self.batch_size, self.width, self.height, 4),
            dtype='float32',
            )

        soft_colors[:, :, :, 0] *= background_color[0]
        soft_colors[:, :, :, 1] *= background_color[1]
        soft_colors[:, :, :, 2] *= background_color[2]

        faces_info, aggrs_info, soft_colors = \
            soft_rasterize_cuda.forward_soft_rasterize(face_vertices, textures,
                                                       faces_info, aggrs_info,
                                                       soft_colors,
                                                       width, height, near, far, eps,
                                                       sigma_val, self.func_dist_type, self.dist_eps,
                                                       gamma_val, self.func_rgb_type, self.func_alpha_type,
                                                       self.texture_type, fill_back)

        self.save_for_backward = (face_vertices, textures, soft_colors, faces_info, aggrs_info)

        return soft_colors



def soft_rasterize(face_vertices, textures, width=256, height=256,
                   background_color=[0, 0, 0], near=1, far=100,
                   fill_back=True, eps=1e-3,
                   sigma_val=1e-5, dist_func='barycentric', dist_eps=1e-4,
                   gamma_val=1e-4, aggr_func_rgb='softmax', aggr_func_alpha='prod',
                   texture_type='surface'):

    soft = SoftRasterizeFunction()
    return soft.execute(face_vertices, textures, width, height,
                                       background_color, near, far,
                                       fill_back, eps,
                                       sigma_val, dist_func, dist_eps,
                                       gamma_val, aggr_func_rgb, aggr_func_alpha,
                                       texture_type)