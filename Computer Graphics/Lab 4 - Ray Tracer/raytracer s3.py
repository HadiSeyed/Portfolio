import json
from myshapes import Sphere, Triangle, Plane
import numpy as np
import matplotlib.pyplot as plt

scene_fn = "scene_3.json"
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


# YOUR CODE HERE


# Step 1: Use Gram-Schmidt Orthogonalization to determine camera axes
def gram_schmidt(A):
    Q, R = np.linalg.qr(A)
    return Q

def get_camera_axes(LookAt, LookFrom, Up):
    w = (LookFrom - LookAt) / np.linalg.norm(LookFrom - LookAt)
    u = np.cross(Up, w) / np.linalg.norm(np.cross(Up, w))
    v = np.cross(w, u)
    return gram_schmidt(np.array([u, v, w]))

# Step 2: Determine dimensions of the window based on field of view
def get_window_dimensions(FieldOfView, aspect_ratio):
    # FieldOfView is in degrees
    height = 2 * np.tan(np.radians(FieldOfView) / 2)
    width = height * aspect_ratio
    return width, height

# Step 3: Determine s coordinate for each pixel based on resolution
def get_pixel_coordinates(resolution):
    width, height = resolution
    s_coordinates = []
    for i in range(height):
        for j in range(width):
            s_coordinates.append((j / width, i / height))
    return s_coordinates

# Step 4: Set ray origin and direction
def get_ray(LookFrom, s, camera_axes):
    u, v, w = camera_axes
    ro = LookFrom
    rd = u * s[0] + v * s[1] - w
    rd /= np.linalg.norm(rd)
    return ro, rd


# Step 5: Ray-sphere intersection test
def ray_sphere_intersection(ro, rd, sphere_center, sphere_radius):
    oc = ro - sphere_center
    a = np.dot(rd, rd)
    b = 2.0 * np.dot(oc, rd)
    c = np.dot(oc, oc) - sphere_radius**2
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return -1  # No intersection
    else:
        t = (-b - np.sqrt(discriminant)) / (2 * a)
        if t > 0:
            return t  # Intersection at t
        else:
            return -1  # Intersection behind ray origin
     
# Step 6
def trace_ray(ray_origin, ray_direction, depth, objects, light, epsilon, background_color):
    if depth <= 0:
        return background_color

    closest_t = np.inf
    closest_obj = None
    color = np.zeros(3)  # Initialize color to black
    for obj in objects:
        t = obj.intersect(ray_origin, ray_direction)
        if 0 < t < closest_t:
            closest_t = t
            closest_obj = obj

    if closest_obj is None:
        return background_color
    else:
         return closest_obj.getDiffuse()
    
    '''
    # Get intersection point and normal
    intersection_point = ray_origin + closest_t * ray_direction
    normal = closest_obj.getNormal(intersection_point)
    normal = normal / np.linalg.norm(normal)  # Normalize the normal

    # Compute reflection ray
    reflect_dir = ray_direction - 2 * np.dot(ray_direction, normal) * normal
    reflect_origin = intersection_point + epsilon * normal  # Nudge the origin a bit above the surface to prevent self-intersection

    # Calculate reflection color recursively
    if closest_obj.refl > 0:
        reflected_color = trace_ray(reflect_origin, reflect_dir, depth - 1, objects, light, epsilon, background_color)
        #print(reflected_color)
        reflected = closest_obj.refl * reflected_color
    else:
        reflected = 0
   
    # If the object is not a perfect mirror, compute its own color
    if closest_obj.refl < 1:
        # Ambient color
        ambient = closest_obj.Ka * light['AmbientLight']
        #color += ambient  # Add the ambient light to the final color
    
        # Diffuse and specular components are added if the point is not in shadow
        shadow_ray = reflect_origin + light['DirectionToLight']
        is_shadowed = any(obj.intersect(reflect_origin, shadow_ray) > 0 for obj in objects)
        if not is_shadowed:
            # Light direction
            light_dir = light['DirectionToLight'] - intersection_point
            light_dir = light_dir / np.linalg.norm(light_dir)  # Normalize the light direction

            # Diffuse color
            diffuse = closest_obj.Kd * max(np.dot(normal, light_dir), 0) * light['LightColor']
            #color += diffuse

            # Specular color
            view_dir = LookFrom - intersection_point
            view_dir = view_dir / np.linalg.norm(view_dir)  # Normalize the view direction
            reflect_dir = 2 * np.dot(normal, light_dir) * normal - light_dir
            reflect_dir = reflect_dir / np.linalg.norm(reflect_dir)  # Normalize the reflection direction
            specular = closest_obj.Ks * np.power(max(np.dot(view_dir, reflect_dir), 0), closest_obj.gloss) * light['LightColor']
            #color += specular
        else:
            diffuse = 0
            specular = 0

    color = ambient + diffuse + specular + reflected

    return color
    
'''


# Example usage
LookAt = np.array([0, 0, 0])
LookFrom = np.array([0, 0, 1])
Up = np.array([0, 1, 0])
FieldOfView = 90  # degrees
aspect_ratio = 1.0  # assume square pixels
resolution = [256, 256]
max_depth = 3
sphere_center = np.array([0, 0, 0])
sphere_radius = 0.4
epsilon = 1e-5
camera_axes = get_camera_axes(LookAt, LookFrom, Up)
width, height = get_window_dimensions(FieldOfView, aspect_ratio)
s_coordinates = get_pixel_coordinates(resolution)



for i in range(resolution[1]):
    for j in range(resolution[0]):
        # Convert pixel positions to normalized device coordinates (NDC) and then to world coordinates
        x = (j + 0.5) / resolution[0] * 2 - 1  # NDC x-coordinate [-1, 1]
        y = 1 - (i + 0.5) / resolution[1] * 2  # NDC y-coordinate [1, -1], flipped
        x_world = x * width / 2
        y_world = y * height / 2
        ray_dir = np.array([x_world, y_world, -1])  # Assuming camera looks towards -z
        ray_dir = ray_dir / np.linalg.norm(ray_dir)  # Normalize ray direction
        ray_dir_world = camera_axes @ ray_dir  # Convert to world coordinates

        ray_origin = LookFrom
        closest_t = np.inf
        #pixel_color = trace_ray(ray_origin, ray_dir_world, max_depth, objects, light, epsilon, light['BackgroundColor'])
        
        # Default ambient component, used if no intersections are found
        default_ambient = np.array(light['AmbientLight'])
    
        for obj in objects:
            t = obj.intersect(ray_origin, ray_dir_world)
            if 0 < t < closest_t:
                closest_t = t
                intersection_point = ray_origin + t * ray_dir_world
                normal = obj.getNormal(intersection_point)
                normal /= np.linalg.norm(normal)  # Normalize normal
                
                # Generate shadow ray
                to_light = np.array(light['Position']) - intersection_point if 'Position' in light else np.array(light['DirectionToLight'])
                shadow_ray_origin = intersection_point + epsilon * normal  # Nudge start point
                shadow_ray_dir = to_light / np.linalg.norm(to_light)

                shadowed = False
                
                
                # Check for obstructions
                for blocker in objects:
                    t_shadow = blocker.intersect(shadow_ray_origin, shadow_ray_dir)
                    if 0 < t_shadow < np.linalg.norm(to_light):  # Intersection before reaching light
                        shadowed = True
                        break

                # Ambient component
                ambient = obj.Ka * np.array(light['AmbientLight'])

                if not shadowed:
                    # Light direction
                    light_dir = to_light / np.linalg.norm(to_light)
                    
                    # Diffuse component
                    diffuse = obj.Kd * max(np.dot(normal, light_dir), 0) * np.array(obj.getDiffuse()) * np.array(light['LightColor'])
                    
                    # View direction (from intersection point to the camera)
                    view_dir = np.array(camera['LookFrom']) - intersection_point
                    view_dir /= np.linalg.norm(view_dir)
                    
                    # Specular component
                    reflect_dir = 2 * np.dot(normal, light_dir) * normal - light_dir
                    specular = obj.Ks * (max(np.dot(view_dir, reflect_dir), 0) ** obj.gloss) * np.array(obj.getSpecular()) * np.array(light['LightColor'])
                    
                    if obj.refl > 0:
                        reflect_dir = ray_dir - 2 * np.dot(ray_dir, normal) * normal
                        reflect_origin = intersection_point + epsilon * normal  # Nudge the origin a bit above the surface to prevent self-intersection

                        reflected_color = trace_ray(intersection_point, reflect_dir, max_depth, objects, light, epsilon, light['BackgroundColor'])
                        reflected = obj.refl * reflected_color
                    else:
                        reflected = 0

                    # Final color for this pixel
                    pixel_color = ambient + diffuse + specular + reflected
                else:
                    # In shadow: Apply only ambient component
                    pixel_color = ambient
                break  # Exit the loop after finding the closest intersection

        if closest_t == np.inf:  # No intersection found for this pixel
            pixel_color = default_ambient  # Use a default ambient light color or background color


        image[i, j] = np.clip(pixel_color, 0, 1)  # Clip values to [0, 1] range

# Display and save the output image
plt.imshow(image)
plt.axis('off')  # Turn off the axis
plt.show()
