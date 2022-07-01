import math

import numpy as np
import jittor as jt
import load_obj

def normalize(v):
    '''

    @param v: vector
    @return: v_normalized
    '''

    return v/math.sqrt(np.linalg.norm(v))

class MeshModel(object):
    dist = {
        "triangleMesh": 0
    }
    def __init__(self, file_obj, mode="triangleMesh", texture_exist = False, texture_type='surface', texture_res = 4):
        if MeshModel.dist[mode] == 0:

            self.vertices, self.faces, self.textures = \
                            self.from_Mesh(file_obj, texture_exist, texture_type, texture_res)

            self.batch_size = self.vertices.shape[0]
            self.num_vertices = self.vertices.shape[1]
            self.num_faces = self.vertices.shape[1]
            self.texture_type = texture_type
            self.texture_res = int(np.sqrt(self.textures.shape[2]))

            '''
            print("vertices", self.vertices.shape)
            print("faces", self.faces.shape)
            print("textures", self.textures.shape)
            '''


    def from_Mesh(self, file_obj, texture_exist=True, texture_type='surface', texture_res=4):
        if not texture_exist:
            vertices, faces, vert_norms, vertices_3D = load_obj.load_obj(file_obj, texture_exist)
            textures = None
        else:
            vertices, faces, textures, vert_norms, vertices_3D \
                = load_obj.load_obj(file_obj, texture_exist, texture_res, texture_type)



        assert isinstance(vertices, np.ndarray)

        assert isinstance(vertices_3D, np.ndarray)

        assert isinstance(vert_norms, np.ndarray)

        assert isinstance(faces, np.ndarray)

        if len(vertices.shape) == 2:
            vertices = vertices[None, :, :]
        if len(faces.shape) == 2:
            faces = faces[None, :, :]
        if len(vert_norms.shape) == 2:
            vert_norms = vert_norms[None, :, :]
        if len(vertices_3D.shape) == 2:
            vertices_3D = vertices_3D[None, :, :]


        batch_size = vertices.shape[0]
        num_vertices = vertices.shape[1]
        num_faces = faces.shape[1]
        self.vert_norms = vert_norms
        self.vertices_3D = vertices_3D

        # create textures
        if textures is None:
            if texture_type == 'surface':
                textures = np.ones((batch_size, num_faces, texture_res**2, 3),
                                            dtype='float32')
                self.texture_res = texture_res
            elif texture_type == 'vertex':
                textures = np.ones((batch_size, num_vertices, 3),
                                            dtype='float32')
                self.texture_res = 1  # vertex color doesn't need resolution
        else:
            if isinstance(textures, np.ndarray):
                textures = textures.astype(float)
            if len(textures.shape) == 3 and texture_type == 'surface':
                textures = textures[None, :, :, :]
            if len(textures.shape) == 2 and texture_type == 'vertex':
                textures = textures[None, :, :]

        return vertices, faces, textures


    def get_faces(self):
        return self.faces


    def get_vertices(self):
        return self.vertices


    def get_textures(self):
        return self.textures


    def get_face_vertices(self):
        self.face_vertices = self.to_facevertices(self.vertices, self.faces)
        return self.to_facevertices(self.vertices, self.faces)

    @property
    def surface_normals(self):
        face_vertices = self.to_facevertices(self.vertices_3D, self.faces)
        v10 = face_vertices[:, :, 0] - face_vertices[:, :, 1]
        v12 = face_vertices[:, :, 2] - face_vertices[:, :, 1]
        v = np.cross(v12, v10)

        for i in range(0, v.shape[0]):
            for j in range(0, v.shape[1]):
                v0 = v[i, j, 0]
                v1 = v[i, j, 1]
                v2 = v[i, j, 2]
                sum = math.sqrt(pow(v0, 2) + pow(v1, 2) + pow(v2, 2))
                v[i, j, 0] = v0/sum
                v[i, j, 1] = v1/sum
                v[i, j, 2] = v2/sum
        return v

    @property
    def vertex_normals(self):
        return self.vert_norms

    def to_facevertices(self, vertices, faces):
        """
            :param vertices_3D: [batch size, number of vertices, 3]
            :param faces: [batch size, number of faces, 3]
            :return: [batch size, number of faces, 3, 3]
            """
        assert (len(vertices.shape) == 3)
        assert (len(faces.shape) == 3)
        assert (vertices.shape[0] == faces.shape[0])
        assert (vertices.shape[2] == 3 or vertices.shape[2] == 4)
        assert (faces.shape[2] == 3)

        if vertices.shape[2] == 3:
            bs, nv = vertices.shape[:2]
            bs, nf = faces.shape[:2]
            device = vertices
            faces = faces + (np.arange(bs, dtype='int32') * nv)[:, None, None]
            vertices = vertices.reshape((bs * nv, 3))
            face_vertices = np.array(vertices[faces])
            return face_vertices
        else:
            bs, nv = vertices.shape[:2]
            bs, nf = faces.shape[:2]
            device = vertices
            faces = faces + (np.arange(bs, dtype='int32') * nv)[:, None, None]
            vertices = vertices.reshape((bs * nv, 4))
            face_vertices = np.array(vertices[faces])
            return face_vertices
    @property
    def face_textures(self):
        if self.texture_type in ['surface']:
            return self.textures
        elif self.texture_type in ['vertex']:
            return self.face_vertices(self.textures, self.faces)
        else:
            raise ValueError('texture type not applicable')