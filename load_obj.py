import os
import numpy as np
import cv2 as cv
import pybind11
import Functions as F

def load_mtl(filename_mtl):
    '''
    load color (Kd) and filename of textures from *.mtl
    '''
    texture_filenames = {}
    colors = {}
    material_name = ''
    with open(filename_mtl) as f:
        for line in f.readlines():
            if len(line.split()) != 0:
                if line.split()[0] == 'newmtl':
                    material_name = line.split()[1]
                if line.split()[0] == 'map_Kd':
                    texture_filenames[material_name] = line.split()[1]
                if line.split()[0] == 'Kd':
                    colors[material_name] = np.array(list(map(float, line.split()[1:4])))
    #print("mtl done")
    return colors, texture_filenames


def load_textures(filename_obj, filename_mtl, texture_res):
    # load vertices
    vertices = []
    with open(filename_obj) as f:
        lines = f.readlines()
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'vt':
            vertices.append([float(v) for v in line.split()[1:3]])
    vertices = np.vstack(vertices).astype(np.float32)

    # load faces for textures
    faces = []
    material_names = []
    material_name = ''
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'f':
            vs = line.split()[1:]
            nv = len(vs)
            if '/' in vs[0] and '//' not in vs[0]:
                v0 = int(vs[0].split('/')[1])
            else:
                v0 = 0
            for i in range(nv - 2):
                if '/' in vs[i + 1] and '//' not in vs[i + 1]:
                    v1 = int(vs[i + 1].split('/')[1])
                else:
                    v1 = 0
                if '/' in vs[i + 2] and '//' not in vs[i + 2]:
                    v2 = int(vs[i + 2].split('/')[1])
                else:
                    v2 = 0
                faces.append((v0, v1, v2))
                material_names.append(material_name)
        if line.split()[0] == 'usemtl':
            material_name = line.split()[1]
    faces = np.vstack(faces).astype(np.int32) - 1
    faces = vertices[faces]
    faces[1 < faces] = faces[1 < faces] % 1

    colors, texture_filenames = load_mtl(filename_mtl)

    texture_res = 8

    for material_name, filename_texture in list(texture_filenames.items()):
        #filename_texture = os.path.join(os.path.dirname(filename_obj), filename_texture)
        filename_texture = 'textureimage.png'
        image = cv.imread(filename_texture).astype(np.float32)
        texture_size = image.shape[0]
        nf = faces.shape[0]
        tmp = 2 * texture_size // nf
        if tmp >= texture_res:
            texture_res = tmp


    textures = np.ones((faces.shape[0], texture_res**2, 3), dtype='float32')  # face[0] ok

    #
    for material_name, color in list(colors.items()):
        for i, material_name_f in enumerate(material_names):
            if material_name == material_name_f:
                textures[i, :, :] = color[None, :]

    for material_name, filename_texture in list(texture_filenames.items()):
        filename_texture = os.path.join(os.path.dirname(filename_obj), filename_texture)

        image = cv.imread(filename_texture).astype(np.float32)

        # texture image may have one channel (grey color)
        if len(image.shape) == 2:
            image = np.stack((image,)*3, -1)
        # or has extral alpha channel shoule ignore for now
        if image.shape[2] == 4:
            image = image[:, :, :3]

        width = image.shape[0]
        height = image.shape[1]
        center = (height // 2, width // 2)
        M = cv.getRotationMatrix2D(center, -90, 1)
        image = cv.warpAffine(image, M, (height, width))

        cv.imwrite("image.png", image)
        is_update = (np.array(material_names) == material_name).astype(np.int32)
        '''
        
        注意注意注意啦，这里要加一个新函数
        
        '''
        textures = F.texture_mapping(image, texture_res, textures, faces, is_update, image.shape[0], image.shape[1])
    return textures


def load_obj(filename_obj, load_texture=False, texture_res=4, texture_type='surface', normalization=False):
    """
    Load Wavefront .obj file.
    This function only supports vertices (v x x x) and faces (f x x x).
    """

    assert texture_type in ['surface', 'vertex']

    # load vertices and vertex normals
    vertices = []
    vert_norms = []
    vertices_3D = []
    with open(filename_obj) as f:
        lines = f.readlines()
    x_min = 0
    x_max = 0
    y_min = 0
    y_max = 0
    z_min = 0
    z_max = 0
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'v':
            vertices.append([float(v) for v in line.split()[1:4]]+[1.]) # 齐次坐标
            if float(line.split()[1]) < x_min:
                x_min = float(line.split()[1])
            if float(line.split()[1]) > x_max:
                x_max = float(line.split()[1])
            if float(line.split()[2]) < y_min:
                y_min = float(line.split()[2])
            if float(line.split()[2]) > y_max:
                y_max = float(line.split()[2])
            if float(line.split()[3]) < z_min:
                z_min = float(line.split()[3])
            if float(line.split()[3]) > z_max:
                z_max = float(line.split()[3])
            vertices_3D.append([float(v) for v in line.split()[1:4]]) #非齐次坐标
        elif line.split()[0] == 'vn':
            vert_norms.append([float(v) for v in line.split()[1:4]])
    vertices = np.vstack(vertices).astype(np.float32)
    if len(vert_norms) == 0:
        vert_norms = np.zeros_like(vertices)
    else:
        vert_norms = np.vstack(vert_norms).astype(np.float32)
    vertices_3D = np.vstack(vertices_3D).astype(np.float32)
    # load faces
    faces = []
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'f':
            vs = line.split()[1:]
            nv = len(vs)
            v0 = int(vs[0].split('/')[0])
            for i in range(nv - 2):
                v1 = int(vs[i + 1].split('/')[0])
                v2 = int(vs[i + 2].split('/')[0])
                faces.append((v0, v1, v2))
    faces = np.vstack(faces).astype(np.int32) - 1
    print([x_min, x_max, y_min, y_max, z_min, z_max])
    scale = max(abs(x_min), abs(x_max), abs(y_min), abs(y_max), abs(z_min), abs(z_max))
    for i in range(0, vertices.shape[0]):
        vertices[i, 0] /= scale
        vertices[i, 1] /= scale
        vertices[i, 2] /= scale
        vertices_3D[i, 0] /= scale
        vertices_3D[i, 1] /= scale
        vertices_3D[i, 2] /= scale


    # load textures
    if load_texture and texture_type == 'surface':
        textures = None
        for line in lines:
            if line.startswith('mtllib'):
                filename_mtl = os.path.join(os.path.dirname(filename_obj), line.split()[1])
                textures = load_textures(filename_obj, filename_mtl, texture_res)
        if textures is None:
            raise Exception('Failed to load textures.')
    elif load_texture and texture_type == 'vertex':
        textures = []
        for line in lines:
            if len(line.split()) == 0:
                continue
            if line.split()[0] == 'v':
                textures.append([float(v) for v in line.split()[4:7]])
        textures = np.vstack(textures).astype(np.float32)

    # normalize into a unit cube centered zero
    if normalization:
        vertices -= vertices.min(0)[0][None, :]
        vertices /= np.abs(vertices).max()
        vertices *= 2
        vertices -= vertices.max(0)[0][None, :] / 2

    if load_texture:
        return vertices, faces, textures, vert_norms, vertices_3D
    else:
        return vertices, faces, vert_norms, vertices_3D
