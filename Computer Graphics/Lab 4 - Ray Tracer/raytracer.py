import json
from myshapes import Sphere, Triangle, Plane
import numpy as np
import matplotlib.pyplot as plt

scene_fn = "scene_1.json"
res = 256

#### Scene Loader

def loadScene(scene_fn):
	with open(scene_fn) as f:
		data = json.load(f)

	spheres = []
	
	for sphere in data["Spheres"]:
		spheres.append(
			Sphere(sphere["Center"], sphere["Radius"], 
		 	sphere["Mdiff"], sphere["Mspec"], sphere["Mgls"], sphere["Refl"],
		 	sphere["Kd"], sphere["Ks"], sphere["Ka"]))
		
	triangles = []

	for triangle in data["Triangles"]:
		triangles.append(
			Triangle(triangle["A"], triangle["B"], triangle["C"],
			triangle["Mdiff"], triangle["Mspec"], triangle["Mgls"], triangle["Refl"],
			triangle["Kd"], triangle["Ks"], triangle["Ka"]))
	
	planes = []

	for plane in data["Planes"]:
		planes.append(
			Plane(plane["Normal"], plane["Distance"],
			plane["Mdiff"], plane["Mspec"], plane["Mgls"], plane["Refl"],
			plane["Kd"], plane["Ks"], plane["Ka"]))
	
	objects = spheres + triangles + planes

	camera = {
		"LookAt": np.array(data["Camera"]["LookAt"],),
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

### Ray Tracer

camera, light, objects = loadScene(scene_fn)

image = np.zeros((res,res,3), dtype=np.float32)







# def reflect_ray(ray_direction, normal):
#     return ray_direction - 2 * np.dot(ray_direction, normal) * normal

# def cast_ray(origin, direction, objects, depth=0):
#     if depth > 3:  # Max depth to avoid infinite recursion
#         return np.zeros(3)

#     closest_t = float('inf')
#     closest_obj = None
#     intersect_point = None

#     for obj in objects:
#         t = obj.intersect(origin, direction)
#         if t and t < closest_t:
#             closest_t = t
#             closest_obj = obj
#             intersect_point = origin + t * direction

#     if closest_obj is None:
#         return light['BackgroundColor']

#     normal = closest_obj.normal_at(intersect_point)
#     reflect_dir = reflect_ray(direction, normal)

#     color = closest_obj.shade(light, intersect_point, reflect_dir)

#     if closest_obj.Reflectivity > 0:
#         reflected_color = cast_ray(intersect_point + EPSILON * reflect_dir, reflect_dir, objects, depth + 1)
#         color = color * (1 - closest_obj.Reflectivity) + closest_obj.Reflectivity * reflected_color

#     return color

# def render(camera, light, objects, image):
#     aspect_ratio = image.shape[1] / image.shape[0]
#     fov_adjust = np.tan(np.deg2rad(camera['FieldOfView']) / 2)

#     for x in range(res):
#         for y in range(res):
#             ray_direction = np.array([
#                 (2 * (x + 0.5) / res - 1) * fov_adjust * aspect_ratio,
#                 (1 - 2 * (y + 0.5) / res) * fov_adjust,
#                 -1
#             ])

#             ray_direction /= np.linalg.norm(ray_direction)

#             image[y, x] = cast_ray(camera['LookFrom'], ray_direction, objects)

# EPSILON = 0.001

# render(camera, light, objects, image)






def define_viewing_rays(camera, res):
    aspect_ratio = res[1] / res[0]
    fov_adjust = np.tan(np.deg2rad(camera['FieldOfView']) / 2)

    # Gram-Schmidt Orthogonalization to determine camera axes
    w = (camera['LookFrom'] - camera['LookAt']) / np.linalg.norm(camera['LookFrom'] - camera['LookAt'])
    u = np.cross(camera['Up'], w) / np.linalg.norm(np.cross(camera['Up'], w))
    v = np.cross(w, u)


    rays= np.zeros((res[0], res[1], 3))  # Initialize array to store ray directions

    for i in range(res[0]):
        for j in range(res[1]):
            # Determine s coordinate for each pixel
            s = np.array([
                ((2 * (i + 0.5) / res[0]) - 1) * fov_adjust * aspect_ratio,
                ((1 - 2 * (j + 0.5) / res[1])) * fov_adjust,
                -1
            ])

            # Calculate ray direction
            ray_direction = normalize(s[0] * u + s[1] * v - s[2] * w)
            rays[i, j] = ray_direction



    # for i in range(res[0]):
    #     for j in range(res[1]):
	# 		# Calculate ray direction
    #         ray_direction = rays[i, j]
	# 		# Assigning value based on the ray direction
    #         image[i, j] = np.abs(ray_direction)

    return rays

def normalize(v):
    return v / np.linalg.norm(v)

# Define viewing rays
rays = define_viewing_rays(camera, (res, res))

print(rays)


        


# def cast_ray(origin, direction, objects, background_color):
#     closest_t = float('inf')
#     closest_object = None

#     for obj in objects:
#         t = obj.intersect(origin, direction)
#         if t is not None and t < closest_t:
#             closest_t = t
#             closest_object = obj
    
#     if closest_object is None:
#         return background_color
#     else:
#         # For now, set pixel color to the diffuse color of the closest object
#         return closest_object.diffuse_color

# # Iterate through each pixel in the image
# for i in range(res):
#     for j in range(res):
#         ray_direction = rays[i, j]
#         pixel_color = cast_ray(camera["LookFrom"], ray_direction, objects, light["BackgroundColor"])
#         image[i, j] = pixel_color






# def reflect_ray(ray_direction, normal):
#     return ray_direction - 2 * np.dot(ray_direction, normal) * normal

# def intersect_ray_sphere(origin, direction, sphere):
#     oc = origin - sphere.center
#     a = np.dot(direction, direction)
#     b = 2.0 * np.dot(oc, direction)
#     c = np.dot(oc, oc) - sphere.radius**2
#     discriminant = b**2 - 4*a*c

#     if discriminant > 0:
#         t = (-b - np.sqrt(discriminant)) / (2.0 * a)
#         if t > 0:
#             return t
#     return None

# def intersect_ray_plane(origin, direction, plane):
#     denominator = np.dot(plane.normal, direction)
#     if abs(denominator) > 1e-6:
#         t = np.dot(plane.normal, plane.center - origin) / denominator
#         if t > 0:
#             return t
#     return None

# def cast_ray(origin, direction, objects, light, depth=0):
#     if depth > 3:  # Max depth to avoid infinite recursion
#         return light['BackgroundColor']

#     closest_t = float('inf')
#     closest_obj = None
#     intersect_point = None

#     for obj in objects:
#         if isinstance(obj, Sphere):
#             t = intersect_ray_sphere(origin, direction, obj)
#         elif isinstance(obj, Plane):
#             t = intersect_ray_plane(origin, direction, obj)

#         if t and t < closest_t:
#             closest_t = t
#             closest_obj = obj
#             intersect_point = origin + t * direction

#     if closest_obj is None:
#         return light['BackgroundColor']

#     normal = closest_obj.normal_at(intersect_point)
#     reflect_dir = reflect_ray(direction, normal)

#     color = closest_obj.shade(light, intersect_point, reflect_dir)

#     if closest_obj.Refl > 0:
#         reflected_color = cast_ray(intersect_point + EPSILON * reflect_dir, reflect_dir, objects, light, depth + 1)
#         color = color * (1 - closest_obj.Refl) + closest_obj.Refl * reflected_color

#     return color

# def render(camera, light, objects, image):
#     aspect_ratio = image.shape[1] / image.shape[0]
#     fov_adjust = np.tan(np.deg2rad(camera['FieldOfView']) / 2)

#     for x in range(res):
#         for y in range(res):
#             ray_direction = np.array([
#                 (2 * (x + 0.5) / res - 1) * fov_adjust * aspect_ratio,
#                 (1 - 2 * (y + 0.5) / res) * fov_adjust,
#                 -1
#             ])

#             ray_direction /= np.linalg.norm(ray_direction)

#             image[y, x] = cast_ray(camera['LookFrom'], ray_direction, objects, light)

# EPSILON = 0.001

# # Define viewing rays
# rays = define_viewing_rays(camera, (res, res))

# # Render the scene
# render(camera, light, objects, image)







# def define_viewing_rays(camera, res):
#     aspect_ratio = res[1] / res[0]
#     fov_adjust = np.tan(np.deg2rad(camera['FieldOfView']) / 2)

#     w = (camera['LookFrom'] - camera['LookAt']) / np.linalg.norm(camera['LookFrom'] - camera['LookAt'])
#     u = np.cross(camera['Up'], w) / np.linalg.norm(np.cross(camera['Up'], w))
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

# def reflect_ray(ray_direction, normal):
#     return ray_direction - 2 * np.dot(ray_direction, normal) * normal

# def intersect_ray_sphere(origin, direction, sphere):
#     oc = origin - sphere.center
#     a = np.dot(direction, direction)
#     b = 2.0 * np.dot(oc, direction)
#     c = np.dot(oc, oc) - sphere.radius**2
#     discriminant = b**2 - 4*a*c

#     if discriminant > 0:
#         t = (-b - np.sqrt(discriminant)) / (2.0 * a)
#         if t > 0:
#             return t
#     return None

# def intersect_ray_plane(origin, direction, plane):
#     denominator = np.dot(plane.normal, direction)
#     if abs(denominator) > 1e-6:
#         t = np.dot(plane.normal, plane.center - origin) / denominator
#         if t > 0:
#             return t
#     return None

# def cast_ray(origin, direction, objects, light, depth=0):
#     if depth > 3:  # Max depth to avoid infinite recursion
#         return light['BackgroundColor']

#     closest_t = float('inf')
#     closest_obj = None
#     intersect_point = None

#     for obj in objects:
#         if isinstance(obj, Sphere):
#             t = intersect_ray_sphere(origin, direction, obj)
#         elif isinstance(obj, Plane):
#             t = intersect_ray_plane(origin, direction, obj)

#         if t and t < closest_t:
#             closest_t = t
#             closest_obj = obj
#             intersect_point = origin + t * direction

#     if closest_obj is None:
#         return light['BackgroundColor']

#     normal = closest_obj.normal_at(intersect_point)
#     reflect_dir = reflect_ray(direction, normal)

#     color = closest_obj.shade(light, intersect_point, reflect_dir)

#     if closest_obj.Refl > 0:
#         reflected_color = cast_ray(intersect_point + EPSILON * reflect_dir, reflect_dir, objects, light, depth + 1)
#         color = color * (1 - closest_obj.Refl) + closest_obj.Refl * reflected_color

#     return color

# def render(camera, light, objects, image):
#     aspect_ratio = image.shape[1] / image.shape[0]
#     fov_adjust = np.tan(np.deg2rad(camera['FieldOfView']) / 2)

#     for x in range(res):
#         for y in range(res):
#             ray_direction = np.array([
#                 (2 * (x + 0.5) / res - 1) * fov_adjust * aspect_ratio,
#                 (1 - 2 * (y + 0.5) / res) * fov_adjust,
#                 -1
#             ])
#             ray_direction /= np.linalg.norm(ray_direction)

#             image[y, x] = cast_ray(camera['LookFrom'], ray_direction, objects, light)

# EPSILON = 0.001

# # Load the scene
# camera, light, objects = loadScene(scene_fn)

# # Create an empty image
# image = np.zeros((res, res, 3))

# # Define viewing rays
# rays = define_viewing_rays(camera, (res, res))

# # Render the scene
# render(camera, light, objects, image)

# # Save and display the output
# plt.imsave("output.png", image)
# plt.imshow(image)
# plt.show()
	






### Save and Display Output
plt.imsave("output.png", image)
plt.imshow(image);plt.show()
