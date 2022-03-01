import numpy as np
import re
import Geometry as geo
class Model:
    def __init__(self, filename):
        return

class MeshModel(Model):
    vertices = []
    faces = []
    vert_norms = []
    vert_uv = []
    errorcheck = 0
    def __init__(self, filename):
        '''
        本构造函数的主要目的是解析.obj文件
        @param filename: 输入的obj文件地址
        '''
        objfile = open(filename,'r')
        while(1):
            self.errorcheck = self.errorcheck + 1
            tempString = objfile.readline()
            if tempString == "":
                break
            tempString = tempString.rstrip("\n")
            if tempString.startswith("v "):
              templist = tempString.split(' ')
              vertex = geo.vertex(templist[1], templist[2], templist[3])
              self.vertices.append(vertex)
            elif tempString.startswith("vn"):
                templist = tempString.split(' ')
                self.vert_norms.append(np.array([templist[1], templist[2], templist[3]]))
            elif tempString.startswith("vt"):
                templist = tempString.split(' ')
                self.vert_uv.append(np.array([templist[1], templist[2]]))
            elif tempString.startswith("f "):
                '''1.建立面'''
                templist = re.split(r' |/',tempString)
                vertcord = list([templist[1], templist[4], templist[7]])
                face = geo.face(self.getvertex(int(vertcord[0])), self.getvertex(int(vertcord[1])), self.getvertex(int(vertcord[2])))
                self.faces.append(face)
                '''2.填充点法向量'''
                vert_norm_list = list([templist[3], templist[6], templist[9]])
                for i in range(0, 3):
                    self.getvertex(int(vertcord[i])).set_norm(self.vert_norms[int(vert_norm_list[i])-1])
                '''填充纹理坐标'''
                vert_uv_list = list([templist[2], templist[5], templist[8]])
                for i in range(0, 3):
                    self.getvertex(int(vertcord[i])).set_uv(self.vert_uv[int(vert_uv_list[i])-1])

        objfile.close()
    def num_of_vertices(self):
        return len(self.vertices)

    def num_of_faces(self):
        return len(self.faces)

    def getvertex(self, i):
        return self.vertices[i-1] ## !!! 注意这个-1 这是由于.obj文件的计数是从1开始的，而我们的.obj文件的计数是由0开始的

    def getfaces(self, i):
        return self.faces[i]

