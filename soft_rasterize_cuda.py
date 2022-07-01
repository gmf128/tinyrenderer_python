import numpy as np
from numba import cuda
import numpy
import math
import torch

@cuda.jit()
def face_info_cal(face_vertices, face_info, nb):
    '''
    cuda 执行： i: 第i个面
    :param face_vertices: [nb, nf, 3, 3]
    :param face_info: [nb, nf, 27]
    :return:
    '''
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    nf = face_vertices.shape[1]
    if i >= nf: return
    xA = face_vertices[nb, i, 0, 0]
    yA = face_vertices[nb, i, 0, 1]
    xB = face_vertices[nb, i, 1, 0]
    yB = face_vertices[nb, i, 1, 1]
    xC = face_vertices[nb, i, 2, 0]
    yC = face_vertices[nb, i, 2, 1]
    face_inv_determinant = xC*(yA - yB) + xA*(yB - yC) + xB*(yC - yA)
    inv_star = [yB-yC, xC-xB, xB*yC-xC*yB, yC-yA, xA-xC, xC*yA-xA*yC, yA-yB, xB-xA, xA*yB-xB*yA]
    if face_inv_determinant > 0:
        face_inv_determinant = max(face_inv_determinant, 1e-10)
    else:
        face_inv_determinant = min(face_inv_determinant, -1e-10)
    for j in range(3):
        for k in range(3):
            face_info[nb, i, j*3+k] = inv_star[j*3+k]/face_inv_determinant


@cuda.jit
def soft_coef_cal(face_vertices, D_ijk, sigma, nb):
    '''
    gpu function: calculate all coefficients of each pixel and each  triangle
    :param face_vertices: [nb, nf, 3, 3]
    :param D_ijk: store for soft-fragments
    :param sigma: sigma value
    :param nb : bitch number
    '''

    i, j, k = cuda.grid(3)
    if i < D_ijk.shape[0] and j < D_ijk.shape[1] and k < D_ijk.shape[2]:
        u, v, w = barycentric_cord(face_vertices[nb, k, 0, 0:2],
                               face_vertices[nb, k, 1, 0:2],
                               face_vertices[nb, k, 2, 0:2],
                               i+0.5, j+0.5)
        if u > 0 and v > 0 and w > 0:
            delta = 1
        else:
            delta = -1
        uvw = 3 * pow(min(u, v, w), 2)
        #uvw = u * v * w
        if uvw < 0:
            inx = -delta * uvw / sigma
        else:
            inx = delta * uvw / sigma
        if delta == 1:
            dij = 1.0 / (1 + math.exp(-inx))
        else:
            dij = math.exp(inx) / (1 + math.exp(inx))

        D_ijk[i, j, k] = dij

@cuda.jit
def silhouette(soft_colors, D_ijk, nb):
    '''
    gpu function: using soft coefficients to generate silhouette image
    :param soft_colors: [nb, width, height, 3/4]
    :param D_ijk: [width, height, nf]
    :return:
    '''
    color = 1
    i, j = cuda.grid(2)
    if i < soft_colors.shape[1] and j < soft_colors.shape[2]:
        for k in range(0, D_ijk.shape[2]):
            color *= 1 - D_ijk[i, j, k]
    color = 1 - color
    soft_colors[nb, j, i, 0] = 255 * color
    soft_colors[nb, j, i, 1] = 255 * color
    soft_colors[nb, j, i, 2] = 255 * color


@cuda.jit(device=True)
def barycentric_cord(A, B, C, x, y):
    xA = A[0]
    yA = A[1]
    xB = B[0]
    yB = B[1]
    xC = C[0]
    yC = C[1]
    D = (xB - xA)*(yC - yA) - (xC - xA)*(yB - yA)
    Dx = (x - xB)*(y - yC) - (y - yB)*(x - xC)
    Dy = (x - xC)*(y - yA) - (y - yC)*(x - xA)
    u = Dx / D
    v = Dy / D
    w = 1 - u -v
    return (u, v, w)

def barycentric_cord_test(A, B, C, x, y):
    xA = A[0]
    yA = A[1]
    xB = B[0]
    yB = B[1]
    xC = C[0]
    yC = C[1]
    D = (xB - xA)*(yC - yA) - (xC - xA)*(yB - yA)
    Dx = (x - xB)*(y - yC) - (y - yB)*(x - xC)
    Dy = (x - xC)*(y - yA) - (y - yC)*(x - xA)
    u = Dx / D
    v = Dy / D
    w = 1 - u -v
    return (u, v, w)


def boundingbox_test(A, B, C, x, y):
    xA = A[0]
    yA = A[1]
    xB = B[0]
    yB = B[1]
    xC = C[0]
    yC = C[1]
    x_min = min(xA, min(xB, xC))
    x_max = max(xA, max(xB, xC))
    y_min = min(yC, min(yB, yA))
    y_max = max(yC, max(yB, yA))
    return x < x_min or x > x_max or y < y_min or y > y_max

@cuda.jit(device=True)
def boundingbox(A, B, C, x, y):
    xA = A[0]
    yA = A[1]
    xB = B[0]
    yB = B[1]
    xC = C[0]
    yC = C[1]
    x_min = min(xA, min(xB, xC))
    x_max = max(xA, max(xB, xC))
    y_min = min(yC, min(yB, yA))
    y_max = max(yC, max(yB, yA))
    return x < x_min or x > x_max or y < y_min or y > y_max

@cuda.jit
def soft_forward(face_vertices, textures,
                           # faces_info, aggrs_info,
                           soft_colors,
                           width, height, near, far, eps,
                           sigma_val, func_dist_type, dist_eps,
                           gamma_val, func_rgb_type, func_alpha_type,
                           texture_type, nb):
    if near > 0:
        near = -near
    if far > 0:
        far = -far
    i, j = cuda.grid(2)
    if i >= width or j >= height:
        return
    tmp = i
    i = j
    j = tmp
    nf = face_vertices.shape[1]
    softmax_sum = math.exp(eps / gamma_val)
    softmax_max = eps
    soft_color_k_0 = soft_colors[nb, i, j, 0] * softmax_sum
    soft_color_k_1 = soft_colors[nb, i, j, 1] * softmax_sum
    soft_color_k_2 = soft_colors[nb, i, j, 2] * softmax_sum
    #gamma_val = 1e-9
    for k in range(0, nf):
        # soft fragment
        u, v, w = barycentric_cord(face_vertices[nb, k, 0, 0:2],
                                        face_vertices[nb, k, 1, 0:2],
                                        face_vertices[nb, k, 2, 0:2],
                                        i + 0.5, j + 0.5)

        if u > 0 and v > 0 and w > 0:
            delta = 1
        else:
            delta = -1
        # func_dist = barycentric
        uvw = 3 * min(u, v, w)
        if uvw < 0:
            inx = -delta * uvw / sigma_val
        else:
            inx = delta * uvw / sigma_val
        if delta == 1:
            dij = 1.0 / (1 + math.exp(-inx))
        else:
            dij = math.exp(inx) / (1 + math.exp(inx))

        # color mapping

        zp = u * face_vertices[nb, k, 0, 2] + v * face_vertices[nb, k, 1, 2] + w * face_vertices[
            nb, k, 2, 2]
        zp = (2 * far * near / (near - far)) * (1 / ((near + far) / (near - far) - zp))
        if zp > near or zp < far:
            continue
        zp_norm = (far - zp) / (far - near)  # [0, 1]
        if boundingbox(face_vertices[nb, k, 0, 0:2],
                                        face_vertices[nb, k, 1, 0:2],
                                        face_vertices[nb, k, 2, 0:2],
                                        i + 0.5, j + 0.5):
            zp_norm = eps
        exp_delta_z = 1.
        if zp_norm > softmax_max:
            exp_delta_z = math.exp((softmax_max - zp_norm) / gamma_val)
            softmax_max = zp_norm
        exp_z = math.exp((zp_norm - softmax_max) / gamma_val)
        softmax_sum = exp_z * dij + exp_delta_z * softmax_sum
        texture_res = int(math.sqrt(textures.shape[2]))
        # calculte w_clip
        w_0 = max(min(u, 1.), 0.)
        w_1 = max(min(v, 1.), 0.)
        w_2 = max(min(w, 1.), 0.)
        w_sum = max(w_0 + w_1 + w_2, 1e-5)
        w_0 /= w_sum
        w_1 /= w_sum
        w_2 /= w_sum

        # soft_color_k[n] = soft_color_k[n] + exp_z * dij * color_k  # * soft_fragment;
        color_k_0 = forward_sample_texture(textures, w_0, w_1, texture_res, 0, k, texture_type, nb)
        soft_color_k_0 = soft_color_k_0 * exp_delta_z + exp_z * dij * color_k_0
        color_k_1 = forward_sample_texture(textures, w_0, w_1, texture_res, 1, k, texture_type, nb)
        soft_color_k_1 = soft_color_k_1 * exp_delta_z + exp_z * dij * color_k_1
        color_k_2 = forward_sample_texture(textures, w_0, w_1, texture_res, 2, k, texture_type, nb)
        soft_color_k_2 = soft_color_k_2 * exp_delta_z + exp_z * dij * color_k_2


    # final aggregate

    soft_colors[nb, i, j, 0] = int(soft_color_k_0 / softmax_sum)
    soft_colors[nb, i, j, 1] = int(soft_color_k_1 / softmax_sum)
    soft_colors[nb, i, j, 2] = int(soft_color_k_2 / softmax_sum)

    # print(i, j, softmax_sum, zp_norm, soft_colors[nb, i, j, 0], soft_colors[nb, i, j, 1], soft_colors[nb, i, j, 2])
    # aggrs_info[nb, i, j, 0] = softmax_sum
    # aggrs_info[nb, i, j, 1] = softmax_max

@cuda.jit(device=True)
def forward_sample_texture(textures, w_0, w_1, R, n, k, texture_sample_type, nb):
    '''

    :param textures:
    :param w:
    :param R:
    :param n: R/G/B
    :param k: index k face
    :param texture_sample_type: 0:face/ 1:vertex
    :param nb: batch no.
    :return: color
    '''

    R = int(R)
    texture_k = 0
    if texture_sample_type == 0:
        # sample surface color with resolution as R
        w_x = int(w_0 * R - 1/2)
        w_y = int(w_1 * R - 1/2)
        if (w_0 + w_1) * R - w_x - w_y <= 1:  # 误差和小于1
            #texture_k = textures[nb, k, w_y*R + w_x, n] * 255
            texture_k = textures[nb, k, w_x + w_y * R, n] * 256
        else:
            texture_k = textures[nb, k, (R - 1 - w_y)*R + (R - 1 - w_x), n] * 256

    '''
        w_x = int(w_0 * R)
        w_y = int(w_1 * R)
        if (w_0 + w_1) * R - w_x - w_y <= 1:
            pos_x = w_0 * R
            pos_y = w_1 * R
            weight_x1 = pos_x - int(pos_x)
            weight_x0 = 1 - weight_x1
            weight_y1 = pos_y - int(pos_y)
            weight_y0 = 1 - weight_y1
            texture_k += textures[nb, k, w_x+w_y*R, n] * 255 * weight_x0 * weight_y0
            if((w_x+1)+w_y*R >= R*R):
                texture_k += textures[nb, k, R*R-1, n] * 255 * weight_x1 * weight_y0
            else:
                texture_k += textures[nb, k, (w_x+1)+w_y*R, n] * 255 * weight_x1 * weight_y0
            if(w_x+(w_y+1)*R >= R*R):
                texture_k += textures[nb, k, R * R - 1, n] * 255 * weight_x0 * weight_y1
            else:
                texture_k += textures[nb, k, w_x+(w_y+1)*R, n] * 255 * weight_x0 * weight_y1
            if(w_x+1+(w_y+1)*R >= R*R):
                texture_k += textures[nb, k, R * R - 1, n] * 255 * weight_x1 * weight_y1
            else:
                texture_k += textures[nb, k, w_x+1+(w_y+1)*R, n] * 255 * weight_x1 * weight_y1

        else:
            pos_x = R - w_0 * R
            pos_y = R - w_1 * R
            weight_x1 = pos_x - int(pos_x)
            weight_x0 = 1 - weight_x1
            weight_y1 = pos_y - int(pos_y)
            weight_y0 = 1 - weight_y1
            w_x = R - 1 - w_x
            w_y = R - 1 - w_y
            texture_k += textures[nb, k, w_x + w_y * R, n] * 255 * weight_x0 * weight_y0
            if ((w_x + 1) + w_y * R >= R * R):
                texture_k += textures[nb, k, 0, n] * 255 * weight_x1 * weight_y0
            else:
                texture_k += textures[nb, k, (w_x + 1) + w_y * R, n] * 255 * weight_x1 * weight_y0
            if (w_x + (w_y + 1) * R >= R * R):
                texture_k += textures[nb, k, 0, n] * 255 * weight_x0 * weight_y1
            else:
                texture_k += textures[nb, k, w_x + (w_y + 1) * R, n] * 255 * weight_x0 * weight_y1
            if (w_x + 1 + (w_y + 1) * R >= R * R):
                texture_k += textures[nb, k, 0, n] * 255 * weight_x1 * weight_y1
            else:
                texture_k += textures[nb, k, w_x + 1 + (w_y + 1) * R, n] * 255 * weight_x1 * weight_y1
        '''
    return texture_k

def forward_sample_texture_test(textures, w_0, w_1, R, n, k, texture_sample_type, nb):
        '''

        :param textures:
        :param w:
        :param R:
        :param n: R/G/B
        :param k: index k face
        :param texture_sample_type: 0:face/ 1:vertex
        :param nb: batch no.
        :return: color
        '''
        R = int(R)
        texture_k = 0
        if texture_sample_type == 0:
            # sample surface color with resolution as R
            w_x = int(w_0 * R - 1/2)
            w_y = int(w_1 * R - 1/2)
            if (w_0 + w_1) * R - w_x - w_y <= 1:  # 误差和小于1
                texture_k = textures[nb, k, w_y * R + w_x, n] * 255
            else:
                texture_k = textures[nb, k, (R - 1 - w_y) * R + (R - 1 - w_x), n] * 255

        return texture_k



def forward_soft_rasterize(face_vertices, textures,
                           faces_info, aggrs_info,
                           soft_colors,
                           width, height, near, far, eps,
                           sigma_val, func_dist_type, dist_eps,
                           gamma_val, func_rgb_type, func_alpha_type,
                           texture_type, fill_back):
    '''

    :param face_vertices: [nb, nf, 3, 4]
    :param textures:   [nb, nf, res**2, 3]
    :param faces_info: [nb, nf, 27]
    :param aggrs_info:  [nb, width, height, 2] : soft_sum and soft_max
    :param soft_colors:  [nb, width, height, 4]
    :param width: image_size
    :param height: image_size
    :param near: perspective box near
    :param far: perspective box far
    :param eps:
    :param sigma_val: param in Dij
    :param func_dist_type: {'hard':0, 'barycentric':1, 'euclidean':2}
    :param dist_eps: param in calculating distance
    :param gamma_val: param in color mapping
    :param func_rgb_type: {'hard':0, 'softmax':1}
    :param func_alpha_type: {'hard':0, 'sum':1, 'prod':2}
    :param texture_type: {'vertex':0, 'surface':1}
    :param fill_back:
    :return:
    '''

    # use cuda


    triangles_num = face_vertices.shape[1]
    soft_colors_device = cuda.to_device(soft_colors)
    textures = textures.numpy()
    textures_device = cuda.to_device(textures)
    face_vertices_device = cuda.to_device(face_vertices)
    # faces_info_device = cuda.to_device(faces_info.numpy())
    # aggrs_info_device = cuda.to_device(aggrs_info.numpy())

    for nb in range(0, face_vertices.shape[0]):

        # generating image
        threads_per_block = (16, 16)
        blocks_per_grid_x = int(math.ceil(width / threads_per_block[0]))
        blocks_per_grid_y = int(math.ceil(height / threads_per_block[1]))
        blocksPerGrid = (blocks_per_grid_x, blocks_per_grid_y)


        soft_forward[blocksPerGrid, threads_per_block](face_vertices_device, textures_device,
                           # faces_info_device, aggrs_info_device,
                           soft_colors_device,
                           width, height, near, far, eps,
                           sigma_val, func_dist_type, dist_eps,
                           gamma_val, func_rgb_type, func_alpha_type,
                           texture_type, nb)

        cuda.synchronize()

        face_vertices = face_vertices_device.copy_to_host()
        textures = textures_device.copy_to_host()
        # faces_info = faces_info_device.copy_to_host()
        # aggrs_info = aggrs_info_device.copy_to_host()
        soft_colors = soft_colors_device.copy_to_host()

    return faces_info, aggrs_info, soft_colors