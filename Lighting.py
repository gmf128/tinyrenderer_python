import Functions
import numpy as np
import jittor as jt
import Functions as F

jt.flags.use_cuda = 0
class AmbientLighting():
    def __init__(self, lightIntensity=0.5, light_color=[1, 1, 1]):

        self.light_intensity = lightIntensity
        self.light_color = light_color

    def execute(self, light):
        return F.ambient_lighting(light, self.light_intensity, self.light_color)



class DirectionalLighting():
    def __init__(self, light_intensity=0.5, light_color=(1, 1, 1), light_direction=(0, 1, 0)):

        self.light_intensity = light_intensity
        self.light_color = light_color
        self.light_direction = light_direction

    def execute(self, light, normals):
        return F.directional_lighting(light, normals, self.light_intensity, self.light_color)


class Lighting():
   def __init__(self, intensity_ambient=0.5, color_ambient=[1, 1, 1],
                 intensity_directionals=0.5, color_directionals=[1, 1, 1],
                 directions=[0, 1, 0]):

       self.ambient = AmbientLighting(intensity_ambient, color_ambient)
       self.directionals = [DirectionalLighting(intensity_directionals,
                                                              color_directionals,
                                                              directions)]

   def execute(self, mesh):
        if mesh.texture_type == 'surface':
            light = jt.zeros_like(mesh.faces)  # mesh.faces: [bs, nf, 3]
            light = self.ambient.execute(light)
            for directional in self.directionals:
                light = directional.execute(light, mesh.surface_normals)
            new_textures = mesh.textures * light[:, :, None, :]  # 此处实锤了是最简单的单面shading

        elif mesh.texture_type == 'vertex':
            light = jt.zeros_like(mesh.vertices)
            light = light
            light = self.ambient.execute(light)
            for directional in self.directionals:
                light = directional.execute(light, mesh.vertex_normals)
            new_textures = mesh.textures * light

        mesh.textures = new_textures

        return mesh


