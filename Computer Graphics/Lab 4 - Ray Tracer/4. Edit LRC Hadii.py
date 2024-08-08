import json
from myshapes import Sphere, Triangle, Plane
import numpy as np
import matplotlib.pyplot as plt

scene_fn = "scene_1.json"
res = 256

# Scene Loader
def loadScene(scene_fn):
    with open(scene_fn) as f:
        data = json.load(f)

    spheres = [Sphere(sphere["Center"], sphere["Radius"], sphere["Mdiff"], sphere["Mspec"],
                      sphere["Mgls"], sphere["Refl"], sphere["Kd"], sphere["Ks"], sphere["Ka"])
               for sphere in data["Spheres"]]

    triangles = [Triangle(triangle["A"], triangle["B"], triangle["C"], triangle["Mdiff"],
                          triangle["Mspec"], triangle["Mgls"], triangle["Refl"], triangle["Kd"],
                          triangle["Ks"], triangle["Ka"])
                 for triangle in data["Triangles"]]

    planes = [Plane(plane["Normal"], plane["Distance"], plane["Mdiff"], plane["Mspec"],
                    plane["Mgls"], plane["Refl"], plane["Kd"], plane["Ks"], plane["Ka"])
              for plane in data["Planes"]]

    objects = spheres + triangles + planes

    camera = {
        "LookAt": np.array(data["Camera"]["LookAt"]),
        "LookFrom": np.array(data["Camera"]["LookFrom"]),
        "Up": np.array(data["Camera"]["Up"]),
        "FieldOfView": data["Camera"]["FieldOfView"]
    }

    light = {
        "DirectionToLight": np.array(data["Light"]["DirectionToLight"]),
        "LightColor": np.array(data["Light"]["LightColor"]),
        "AmbientLight": np.array(data["Light"]["AmbientLight"]),
        "BackgroundColor": np.array(data["Light"]["BackgroundColor"]),
    }

    return camera, light, objects

def define_viewing_rays(camera, res):
    aspect_ratio = res[1] / res[0]
    fov_adjust = np.tan(np.deg2rad(camera['FieldOfView']) / 2)

    # Gram-Schmidt Orthogonalization to determine camera axes
    w = normalize(camera['LookFrom'] - camera['LookAt'])
    u = normalize(np.cross(camera['Up'], w))
    v = np.cross(w, u)

    rays = np.zeros((res[0], res[1], 3))   # Initialize array to store ray directions

    for i in range(res[0]):
        for j in range(res[1]):
            s = np.array([
                ((2 * (i + 0.5) / res[0]) - 1) * fov_adjust * aspect_ratio,
                ((1 - 2 * (j + 0.5) / res[1])) * fov_adjust,
                -1
            ])
            
            # Calculate ray direction
            ray_direction = normalize(s[0] * u + s[1] * v - s[2] * w)
            rays[i, j] = ray_direction

    return rays

def normalize(v):
    return v / np.linalg.norm(v) 

def render(camera, light, objects, rays, res):    
    image = np.zeros((res[0], res[1], 3))


    left_color = np.array([204, 0, 204]) / 255.0      # Replace with the RGB values of the left side color
    middle_color = np.array([0, 0, 255]) / 255.0      # Replace with the RGB values of the middle color
    right_color = np.array([204, 0, 204]) / 255.0     # Replace with the RGB values of the right side color
    top_color = np.array([94, 172, 180]) / 255.0      # Replace with the RGB values of the top color
    bottom_color = np.array([94, 172, 180]) / 255.0   # Replace with the RGB values of the bottom color

    for i in range(res[0]):  
         for j in range(res[1]):  
            t = j / res[1]
            color = (1 - t) * left_color + t * middle_color * right_color  # Interpolate between left color
            image[i, j] = color


    for i in range(res[0]):  
         for j in range(res[1]):  
            t = j / res[1]
            color = (1 - t) * top_color + t * middle_color * bottom_color  # Interpolate between left color
            image[i, j] = color





    # for i in range(res[0]):  
    #     for j in range(res[1]):  
    #         # Interpolate between left and right colors
    #         t_horizontal = j / res[1]
    #         horizontal_color = (1 - t_horizontal) * left_color + t_horizontal * right_color

    #         # Interpolate between top and bottom colors
    #         t_vertical = i / res[0]
    #         vertical_color = (1 - t_vertical) * top_color + t_vertical * bottom_color

    #         # Combine horizontal and vertical interpolation to get final color
    #         color = 0.5 * (horizontal_color + vertical_color)
    #         image[i, j] = color



#     # for i in range(res[0]):  
#     #      for j in range(res[1]):  
#     #         t = j / res[1]
#     #         color = (1 - t) * middle_color    # Interpolate between left color
#     #         image[i, j] = color


#     # for i in range(res[0]):  
#     #      for j in range(res[1]):  
#     #         t = j / res[1]
#     #         color = (1 - t) * right_color
#     #         image[i, j] = color


#     # for i in range(res[0]):  
#     #      for j in range(res[1]):  
#     #         t = j / res[1]
#     #         color = (1 - t) * Top_color
#     #         image[i, j] = color


#     # for i in range(res[0]):  
#     #      for j in range(res[1]):  
#     #         t = j / res[1]
#     #         color = (1 - t) * Bottom_color
#     #         image[i, j] = color


#     # for i in range(res):  
#     #      for j in range(res):  
#     #         t = j / (res-1)
#     #         color = (1 - t) * right_color + t * middle_color  # Interpolate between right color
#     #         image[i, j] = color

#     return image





# def render(camera, light, objects, rays, res):    
#     image = np.zeros((res[0], res[1], 3))

#     left_color = np.array([204, 0, 204]) / 255.0      # Replace with the RGB values of the left side color
#     middle_color = np.array([0, 0, 255]) / 255.0      # Replace with the RGB values of the middle color
#     right_color = np.array([204, 0, 204]) / 255.0     # Replace with the RGB values of the right side color
#     top_color = np.array([94, 172, 180]) / 255.0      # Replace with the RGB values of the top color
#     bottom_color = np.array([94, 172, 180]) / 255.0   # Replace with the RGB values of the bottom color

#     for i in range(res[0]):  
#         for j in range(res[1]):  
#             # Interpolate between left and right colors
#             t_horizontal = j / res[1]
#             horizontal_color = (1 - t_horizontal) * left_color + t_horizontal * right_color

#             # Interpolate between top and bottom colors
#             t_vertical = i / res[0]
#             vertical_color = (1 - t_vertical) * top_color + t_vertical * bottom_color

#             # Combine horizontal and vertical interpolation to get final color
#             color = 0.5 * (horizontal_color + vertical_color)
#             image[i, j] = color

#     return image





# def render(camera, light, objects, rays, res):
#     image = np.zeros((res[0], res[1], 3))
#     top_color = np.array([204, 0, 204]) / 255.0  # Purple color
#     bottom_color = np.array([0, 0, 255]) / 255.0  # Blue color

#     for i in range(res[0]):
#         for j in range(res[1]):
#             t = i / res[0]
#             color = (1 - t) * top_color + t * bottom_color  # Interpolate between top and bottom colors
#             image[i, j] = color

#     return image






# def render(camera, light, objects, rays, res):
#     image = np.zeros((res[0], res[1], 3))

#     for i in range(res[0]):
#         for j in range(res[1]):
#             ray_direction = rays[i, j]

#             # Implement ray-object intersection and shading here
#             # Calculate pixel color based on intersections with objects and lighting conditions
#             # image[i, j] = ... (color calculation based on ray-object intersection and shading)

#             # For now, set the pixel color to a constant value (e.g., blue)
#             image[i, j] = [0, 0, 1]  # Blue color for testing

#     return image





# Ray Tracer
# Load the scene
camera, light, objects = loadScene(scene_fn)
image = np.zeros((res,res,3), dtype=np.float32)

# Define viewing rays
rays = define_viewing_rays(camera, (res, res))

print(rays)

# Render the scene
image = render(camera, light, objects, rays, (res, res))

# # Save and display the output
# plt.imsave("output_path, image_switched")
# plt.imsave("output.png", image)

# plt.imshow(image)
# plt.show()

# #Render the image with the colors switched
# image_switched = "render_switched_colors(res)"

# # Save and display the outputoutput_path = "switched_gradient_output.png"
# plt.imsave("output_path, image_switched")
# plt.imshow(image_switched)
# plt.axis('off')   # Turn off axisplt.show()


# Save and display the output
output_path = "output.png"  # Define the output path
plt.imsave(output_path, image)

plt.imshow(image)
plt.show()

# Render the image with the colors switched
image_switched = np.flip(image, axis=1)  # Correct the image_switched assignment

# Save and display the output
output_path_switched = "switched_gradient_output.png"  # Define the output path for the switched image
plt.imsave(output_path_switched, image_switched)

plt.imshow(image_switched)
plt.axis('off')  # Turn off axis
plt.show()








# import json
# from myshapes import Sphere, Triangle, Plane
# import numpy as np
# import matplotlib.pyplot as plt

# scene_fn = "scene_1.json"
# res = 256

# def loadScene(scene_fn):
#     with open(scene_fn) as f:
#         data = json.load(f)

#     spheres = [Sphere(sphere["Center"], sphere["Radius"], sphere["Mdiff"], sphere["Mspec"],
#                       sphere["Mgls"], sphere["Refl"], sphere["Kd"], sphere["Ks"], sphere["Ka"])
#                for sphere in data["Spheres"]]

#     triangles = [Triangle(triangle["A"], triangle["B"], triangle["C"], triangle["Mdiff"],
#                           triangle["Mspec"], triangle["Mgls"], triangle["Refl"], triangle["Kd"],
#                           triangle["Ks"], triangle["Ka"])
#                  for triangle in data["Triangles"]]

#     planes = [Plane(plane["Normal"], plane["Distance"], plane["Mdiff"], plane["Mspec"],
#                     plane["Mgls"], plane["Refl"], plane["Kd"], plane["Ks"], plane["Ka"])
#               for plane in data["Planes"]]

#     objects = spheres + triangles + planes

#     camera = {
#         "LookAt": np.array(data["Camera"]["LookAt"]),
#         "LookFrom": np.array(data["Camera"]["LookFrom"]),
#         "Up": np.array(data["Camera"]["Up"]),
#         "FieldOfView": data["Camera"]["FieldOfView"]
#     }

#     light = {
#         "DirectionToLight": np.array(data["Light"]["DirectionToLight"]),
#         "LightColor": np.array(data["Light"]["LightColor"]),
#         "AmbientLight": np.array(data["Light"]["AmbientLight"]),
#         "BackgroundColor": np.array(data["Light"]["BackgroundColor"]),
#     }

#     return camera, light, objects

# def define_viewing_rays(camera, res):
#     aspect_ratio = res[1] / res[0]
#     fov_adjust = np.tan(np.deg2rad(camera['FieldOfView']) / 2)

#     w = normalize(camera['LookFrom'] - camera['LookAt'])
#     u = normalize(np.cross(camera['Up'], w))
#     v = np.cross(w, u)

#     rays = np.zeros((res[0], res[1], 3))

#     for i in range(res[0]):
#         for j in range(res[1]):
#             s = np.array([
#                 ((2 * (i + 0.5) / res[0]) - 1) * fov_adjust * aspect_ratio,
#                 ((1 - 2 * (j + 0.5) / res[1])) * fov_adjust,
#                 -1
#             ])
#             ray_direction = normalize(s[0] * u + s[1] * v - s[2] * w)
#             rays[i, j] = ray_direction

#     return rays

# def normalize(v):
#     return v / np.linalg.norm(v)

# def render(camera, light, objects, rays, res):    
#     image = np.zeros((res[0], res[1], 3))

#     left_color = np.array([204, 0, 204]) / 255.0      # Replace with the RGB values of the left side color
#     middle_color = np.array([0, 0, 255]) / 255.0      # Replace with the RGB values of the middle color
#     right_color = np.array([204, 0, 204]) / 255.0     # Replace with the RGB values of the right side color

#     for i in range(res[0]):  
#         t = i / res[0]
#         for j in range(res[1]):  
#             color = (1 - t) * left_color + t * right_color  # Interpolate between left and right colors
#             image[i, j] = color

#     return image

# # Load the scene
# camera, light, objects = loadScene(scene_fn)

# # Define viewing rays
# rays = define_viewing_rays(camera, (res, res))

# # Render the scene
# image = render(camera, light, objects, rays, (res, res))

# # Save and display the output
# output_path = "output.png"
# plt.imsave(output_path, image)

# plt.imshow(image)
# plt.show()

# # Render the image with the colors switched
# image_switched = np.flip(image, axis=1)

# # Save and display the output
# output_path_switched = "switched_gradient_output.png"
# plt.imsave(output_path_switched, image_switched)

# plt.imshow(image_switched)
# plt.axis('off')  # Turn off axis
# plt.show()
