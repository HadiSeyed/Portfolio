import json
from myshapes import Sphere, Triangle, Plane
import numpy as np
import matplotlib.pyplot as plt

scene_fn = "scene_1.json"
res = 256

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

    w = normalize(camera['LookFrom'] - camera['LookAt'])
    u = normalize(np.cross(camera['Up'], w))
    v = np.cross(w, u)

    rays = np.zeros((res[0], res[1], 3))

    for i in range(res[0]):
        for j in range(res[1]):
            s = np.array([
                ((2 * (i + 0.5) / res[0]) - 1) * fov_adjust * aspect_ratio,
                ((1 - 2 * (j + 0.5) / res[1])) * fov_adjust,
                -1
            ])
            ray_direction = normalize(s[0] * u + s[1] * v - s[2] * w)
            rays[i, j] = ray_direction

    return rays

def normalize(v):
    return v / np.linalg.norm(v)

def render(camera, light, objects, rays, res):
    image = np.zeros((res[0], res[1], 3))

    for i in range(res[0]):
        for j in range(res[1]):
            ray_direction = rays[i, j]
            # Implement ray-object intersection and shading here
            # Calculate pixel color based on intersections with objects and lighting conditions
            # image[i, j] = ... (color calculation based on ray-object intersection and shading)

            # For now, set the pixel color to a constant value (e.g., blue)
            image[i, j] = [0, 0, 1]  # Blue color for testing

    return image

# Load the scene
camera, light, objects = loadScene(scene_fn)

# Define viewing rays
rays = define_viewing_rays(camera, (res, res))

# Render the scene
image = render(camera, light, objects, rays, (res, res))

# Save and display the output
plt.imsave("output.png", image)
plt.imshow(image)
plt.show()
