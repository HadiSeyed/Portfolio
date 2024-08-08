# Import a library of functions called 'pygame'
import pygame
import numpy as np
from math import pi, sin, cos, tan

class Point:
    def __init__(self,x,y):
        self.x = x
        self.y = y

class Point3D:
    def __init__(self,x,y,z):
        self.x = x
        self.y = y
        self.z = z
        
class Line3D():

    def __init__(self, start, end):
        self.start = start
        self.end = end

def loadOBJ(filename):
    
    vertices = []
    indices = []
    lines = []

    f = open(filename, "r")
    for line in f:
        t = str.split(line)
        if not t:
            continue
        if t[0] == "v":
            vertices.append(Point3D(float(t[1]),float(t[2]),float(t[3])))
            
        if t[0] == "f":
            for i in range(1,len(t) - 1):
                index1 = int(str.split(t[i],"/")[0])
                index2 = int(str.split(t[i+1],"/")[0])
                indices.append((index1,index2))
            
    f.close()

    #Add faces as lines
    for index_pair in indices:
        index1 = index_pair[0]
        index2 = index_pair[1]
        lines.append(Line3D(vertices[index1 - 1],vertices[index2 - 1]))
        
    #Find duplicates
    duplicates = []
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            line1 = lines[i]
            line2 = lines[j]
            
            # Case 1 -> Starts match
            if line1.start.x == line2.start.x and line1.start.y == line2.start.y and line1.start.z == line2.start.z:
                if line1.end.x == line2.end.x and line1.end.y == line2.end.y and line1.end.z == line2.end.z:
                    duplicates.append(j)
            # Case 2 -> Start matches end
            if line1.start.x == line2.end.x and line1.start.y == line2.end.y and line1.start.z == line2.end.z:
                if line1.end.x == line2.start.x and line1.end.y == line2.start.y and line1.end.z == line2.start.z:
                    duplicates.append(j)
                    
    duplicates = list(set(duplicates))
    duplicates.sort()
    duplicates = duplicates[::-1]

    #Remove duplicates
    for j in range(len(duplicates)):
        del lines[duplicates[j]]

    return lines

def loadHouse():
    house = []
    #Floor
    house.append(Line3D(Point3D(-5, 0, -5), Point3D(5, 0, -5)))
    house.append(Line3D(Point3D(5, 0, -5), Point3D(5, 0, 5)))
    house.append(Line3D(Point3D(5, 0, 5), Point3D(-5, 0, 5)))
    house.append(Line3D(Point3D(-5, 0, 5), Point3D(-5, 0, -5)))
    #Ceiling
    house.append(Line3D(Point3D(-5, 5, -5), Point3D(5, 5, -5)))
    house.append(Line3D(Point3D(5, 5, -5), Point3D(5, 5, 5)))
    house.append(Line3D(Point3D(5, 5, 5), Point3D(-5, 5, 5)))
    house.append(Line3D(Point3D(-5, 5, 5), Point3D(-5, 5, -5)))
    #Walls
    house.append(Line3D(Point3D(-5, 0, -5), Point3D(-5, 5, -5)))
    house.append(Line3D(Point3D(5, 0, -5), Point3D(5, 5, -5)))
    house.append(Line3D(Point3D(5, 0, 5), Point3D(5, 5, 5)))
    house.append(Line3D(Point3D(-5, 0, 5), Point3D(-5, 5, 5)))
    #Door
    house.append(Line3D(Point3D(-1, 0, 5), Point3D(-1, 3, 5)))
    house.append(Line3D(Point3D(-1, 3, 5), Point3D(1, 3, 5)))
    house.append(Line3D(Point3D(1, 3, 5), Point3D(1, 0, 5)))
    #Roof
    house.append(Line3D(Point3D(-5, 5, -5), Point3D(0, 8, -5)))
    house.append(Line3D(Point3D(0, 8, -5), Point3D(5, 5, -5)))
    house.append(Line3D(Point3D(-5, 5, 5), Point3D(0, 8, 5)))
    house.append(Line3D(Point3D(0, 8, 5), Point3D(5, 5, 5)))
    house.append(Line3D(Point3D(0, 8, 5), Point3D(0, 8, -5)))
	
    return house

def loadCar():
    car = []
    #Front Side
    car.append(Line3D(Point3D(-3, 2, 2), Point3D(-2, 3, 2)))
    car.append(Line3D(Point3D(-2, 3, 2), Point3D(2, 3, 2)))
    car.append(Line3D(Point3D(2, 3, 2), Point3D(3, 2, 2)))
    car.append(Line3D(Point3D(3, 2, 2), Point3D(3, 1, 2)))
    car.append(Line3D(Point3D(3, 1, 2), Point3D(-3, 1, 2)))
    car.append(Line3D(Point3D(-3, 1, 2), Point3D(-3, 2, 2)))

    #Back Side
    car.append(Line3D(Point3D(-3, 2, -2), Point3D(-2, 3, -2)))
    car.append(Line3D(Point3D(-2, 3, -2), Point3D(2, 3, -2)))
    car.append(Line3D(Point3D(2, 3, -2), Point3D(3, 2, -2)))
    car.append(Line3D(Point3D(3, 2, -2), Point3D(3, 1, -2)))
    car.append(Line3D(Point3D(3, 1, -2), Point3D(-3, 1, -2)))
    car.append(Line3D(Point3D(-3, 1, -2), Point3D(-3, 2, -2)))
    
    #Connectors
    car.append(Line3D(Point3D(-3, 2, 2), Point3D(-3, 2, -2)))
    car.append(Line3D(Point3D(-2, 3, 2), Point3D(-2, 3, -2)))
    car.append(Line3D(Point3D(2, 3, 2), Point3D(2, 3, -2)))
    car.append(Line3D(Point3D(3, 2, 2), Point3D(3, 2, -2)))
    car.append(Line3D(Point3D(3, 1, 2), Point3D(3, 1, -2)))
    car.append(Line3D(Point3D(-3, 1, 2), Point3D(-3, 1, -2)))

    return car

def loadTire():
    tire = []
    #Front Side
    tire.append(Line3D(Point3D(-1, .5, .5), Point3D(-.5, 1, .5)))
    tire.append(Line3D(Point3D(-.5, 1, .5), Point3D(.5, 1, .5)))
    tire.append(Line3D(Point3D(.5, 1, .5), Point3D(1, .5, .5)))
    tire.append(Line3D(Point3D(1, .5, .5), Point3D(1, -.5, .5)))
    tire.append(Line3D(Point3D(1, -.5, .5), Point3D(.5, -1, .5)))
    tire.append(Line3D(Point3D(.5, -1, .5), Point3D(-.5, -1, .5)))
    tire.append(Line3D(Point3D(-.5, -1, .5), Point3D(-1, -.5, .5)))
    tire.append(Line3D(Point3D(-1, -.5, .5), Point3D(-1, .5, .5)))

    #Back Side
    tire.append(Line3D(Point3D(-1, .5, -.5), Point3D(-.5, 1, -.5)))
    tire.append(Line3D(Point3D(-.5, 1, -.5), Point3D(.5, 1, -.5)))
    tire.append(Line3D(Point3D(.5, 1, -.5), Point3D(1, .5, -.5)))
    tire.append(Line3D(Point3D(1, .5, -.5), Point3D(1, -.5, -.5)))
    tire.append(Line3D(Point3D(1, -.5, -.5), Point3D(.5, -1, -.5)))
    tire.append(Line3D(Point3D(.5, -1, -.5), Point3D(-.5, -1, -.5)))
    tire.append(Line3D(Point3D(-.5, -1, -.5), Point3D(-1, -.5, -.5)))
    tire.append(Line3D(Point3D(-1, -.5, -.5), Point3D(-1, .5, -.5)))

    #Connectors
    tire.append(Line3D(Point3D(-1, .5, .5), Point3D(-1, .5, -.5)))
    tire.append(Line3D(Point3D(-.5, 1, .5), Point3D(-.5, 1, -.5)))
    tire.append(Line3D(Point3D(.5, 1, .5), Point3D(.5, 1, -.5)))
    tire.append(Line3D(Point3D(1, .5, .5), Point3D(1, .5, -.5)))
    tire.append(Line3D(Point3D(1, -.5, .5), Point3D(1, -.5, -.5)))
    tire.append(Line3D(Point3D(.5, -1, .5), Point3D(.5, -1, -.5)))
    tire.append(Line3D(Point3D(-.5, -1, .5), Point3D(-.5, -1, -.5)))
    tire.append(Line3D(Point3D(-1, -.5, .5), Point3D(-1, -.5, -.5)))
    
    return tire


DISPLAY_HEIGHT = 512
DISPLAY_WIDTH = 512

# TODO: Define global variables for camera position and properties

# Set up camera parameters
camera_location = np.array([0, 0, 10.0])
camera_rotation_y = np.radians(180.0)

# Define camera movement speed and rotation angle
camera_move_speed = 0.1
camera_rotation_speed = np.radians(5)

def buildProjectionMatrix():	

    near_field = 1.0
    far_field = 1000.0
    fov_x = np.radians(100)
    fov_y = np.radians(100)

    # Build world-to-camera matrix

    translate = np.array([[1, 0, 0, -camera_location[0]],
                         [0, 1, 0, -camera_location[1]],
                         [0, 0, 1, -camera_location[2]],
                         [0,0,0,1]])

    rotate = np.array([[np.cos(camera_rotation_y ), 0, np.sin(camera_rotation_y ),0],
                      [0, 1, 0, 0],
                      [-np.sin(camera_rotation_y ), 0, np.cos(camera_rotation_y ),0],
                      [0 , 0, 0, 1]])

    world_to_camera = np.matmul(rotate,translate)

    # Build projection matrix
    
    zoomx = 1/np.tan(fov_x/2)
    zoomy = 1/np.tan(fov_y/2)

    projection = np.array([[zoomx,0,0,0],
                          [0,zoomy,0,0],
                          [0,0,(far_field + near_field)/(far_field-near_field),(-2*near_field*far_field)/(far_field-near_field)],
                          [0,0,1,0]])

    # Combine matrices
    
    projection_matrix = np.matmul(projection, world_to_camera)
    
    return projection_matrix


def clipTest(pt1,pt2):

    x1, y1, z1, w1 = pt1[0, 0], pt1[1, 0], pt1[2, 0], pt1[3, 0]
    x2, y2, z2, w2 = pt2[0, 0], pt2[1, 0], pt2[2, 0], pt2[3, 0]
    #if x < -w or x > w or y < -w or y > w or z < -w or z > w:

    if x1 < -w1 and x2 < -w2:
        return False
    if x1 > w1 and x2 > w2:
        return False
    if y1 < -w1 and y2 < -w2:
        return False
    if y1 > w1 and y2 > w2:
        return False
    if z1 < -w1 or z2 < -w2:
        return False
    if z1 > w1 and z2 > w2:
        return False

    return True

def toScreen(pt):
    '''
    This code converts pt to screen coordinate space
    '''

    # Homogenous divide
    point_canonical = pt / pt[3]

    ## Homogenous Translation
    #translate = np.array([[1, 0, 0, 0],
    #                     [0, 1, 0, 0],
    #                     [0, 0, 1, 0],
    #                     [0, 0, 0, 1]])
    
    ## Homogenous Scaling
    #scale = np.array([[3,0,0,0],
    #                 [0,1,0,0],
    #                 [0,0,2,0],
    #                 [0,0,0,1]])

    #viewport_matrix = np.matmul(scale, translate)

    viewport_matrix = np.array([[DISPLAY_WIDTH/2, 0, 0, DISPLAY_WIDTH/2],
                         [0, -DISPLAY_HEIGHT/2, 0, DISPLAY_HEIGHT/2],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

    # Multiply by the viewport matrix
    point_screen = np.matmul(viewport_matrix, point_canonical)

    # Extract x, y coordinates
    x = float(point_screen[0, 0])
    y = float(point_screen[1, 0])

    return Point(x, y)

    return Point(20*pt[0][0].item()+200,-20*pt[1][0].item()+200) #BOGUS DRAWING PARAMETERS SO YOU CAN SEE THE HOUSE WHEN YOU START UP


# # Function to handle camera movement
# def move_camera(direction):
#     global camera_location
#     if direction == 'w':  # Move forward
#         camera_location[2] -= camera_move_speed
#     elif direction == 's':  # Move backward
#         camera_location[2] += camera_move_speed
#     elif direction == 'a':  # Move left
#         camera_location[0] -= camera_move_speed
#     elif direction == 'd':  # Move right
#         camera_location[0] += camera_move_speed
#     elif direction == 'r':  # Move up
#         camera_location[1] += camera_move_speed
#     elif direction == 'f':  # Move down
#         camera_location[1] -= camera_move_speed

# # Function to handle camera rotation
# def rotate_camera(direction):
#     global camera_rotation_y
#     if direction == 'q':  # Turn left
#         camera_rotation_y -= camera_rotation_speed
#     elif direction == 'e':  # Turn right
#         camera_rotation_y += camera_rotation_speed

# # Function to reset camera to original position and orientation
# def reset_camera():
#     global camera_location, camera_rotation_y
#     camera_location = np.array([0, 0, 10])
#     camera_rotation_y = np.radians(180)

# Function to handle camera movement
#def move_camera(direction, camera_location, camera_rotation_y):


# Function to handle camera rotation
#def rotate_camera(direction, camera_rotation_y):
#    camera_rotation_speed = np.radians(5)

# Reset camera function remains unchanged
def reset_camera():
    global camera_location, camera_rotation_y
    camera_location = np.array([0, 0, 10])
    camera_rotation_y = np.radians(180)


## Controller Code
#for event in pygame.event.get():
#    if event.type == pygame.QUIT:  # If user clicked close
#        done = True
#    elif event.type == pygame.KEYDOWN:  # If a key was pressed
#        if event.key == pygame.K_h:  # Reset to home position
#            reset_camera()
#        elif event.key in [pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d, pygame.K_r, pygame.K_f]:
#            move_camera(event.unicode)
#        elif event.key in [pygame.K_q, pygame.K_e]:
#            rotate_camera(event.unicode)

# Initialize the game engine
pygame.init()
 
# Define the colors we will use in RGB format
BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
BLUE =  (  0,   0, 255)
GREEN = (  0, 255,   0)
RED =   (255,   0,   0)

# Set the height and width of the screen
size = [DISPLAY_WIDTH, DISPLAY_HEIGHT]
screen = pygame.display.set_mode(size)

pygame.display.set_caption("Shape Drawing")
 
#Set needed variables
done = False
clock = pygame.time.Clock()
start = Point(0.0,0.0)
end = Point(0.0,0.0)
linelist = loadHouse()

#Loop until the user clicks the close button.
while not done:
 
    # This limits the while loop to a max of 100 times per second.
    # Leave this out and we will use all CPU we can.
    clock.tick(100)

    # Clear the screen and set the screen background
    screen.fill(BLACK)

    #Controller Code#
    #####################################################################

    for event in pygame.event.get():
        if event.type == pygame.QUIT: # If user clicked close
            done=True
            
    pressed = pygame.key.get_pressed()

    #if pressed[pygame.K_a]:
    #    camera_location[0] -= camera_move_speed

    #if pressed[pygame.K_w]:
    #    camera_location[2] += 0.25

    if pressed[pygame.K_w]:  # Move forward
        camera_location[2] -= camera_move_speed * np.cos(camera_rotation_y)
        camera_location[0] += camera_move_speed * np.sin(camera_rotation_y)
    elif pressed[pygame.K_s]:  # Move backward
        camera_location[2] += camera_move_speed * np.cos(camera_rotation_y)
        camera_location[0] -= camera_move_speed * np.sin(camera_rotation_y)
    elif pressed[pygame.K_a]:  # Move left
        camera_location[0] -= camera_move_speed * np.cos(camera_rotation_y)
        camera_location[2] -= camera_move_speed * np.sin(camera_rotation_y)
    elif pressed[pygame.K_d]:  # Move right
        camera_location[0] += camera_move_speed * np.cos(camera_rotation_y)
        camera_location[2] += camera_move_speed * np.sin(camera_rotation_y)
    elif pressed[pygame.K_r]:  # Move up
        camera_location[1] += camera_move_speed
    elif pressed[pygame.K_f]:  # Move down
        camera_location[1] -= camera_move_speed
    elif pressed[pygame.K_q]:  # Turn left
        camera_rotation_y -= camera_rotation_speed
    elif pressed[pygame.K_e]:  # Turn right
        camera_rotation_y += camera_rotation_speed


    #Viewer Code#
    #####################################################################

    project = buildProjectionMatrix()

    for s in linelist:
        
        pt1_w = np.matrix([[s.start.x],[s.start.y],[s.start.z],[1]])
        pt2_w = np.matrix([[s.end.x],[s.end.y],[s.end.z],[1]])
        
        pt1_c = project*pt1_w
        pt2_c = project*pt2_w
        
        if clipTest(pt1_c,pt2_c):
            pt1_s = toScreen(pt1_c)
            pt2_s = toScreen(pt2_c)
            pygame.draw.line(screen, BLUE, (pt1_s.x, pt1_s.y), (pt2_s.x, pt2_s.y))

    # Go ahead and update the screen with what we've drawn.
    # This MUST happen after all the other drawing commands.
    pygame.display.flip()

# Be IDLE friendly
pygame.quit()



# Extra Credit

'''
def buildProjectionMatrix():
    global perspective_projection
    if perspective_projection:
        perspective = np.array([[zoomx,0,0,0],
                            [0,zoomy,0,0],
                            [0,0,(far_field + near_field)/(far_field-near_field),(-2*near_field*far_field)/(far_field-near_field)],
                            [0,0,1,0]])
    else:
        orthographic = np.array([[1,0,0,0],
                             [0,1,0,0],
                             [0,0,0,0],
                             [0,0,0,1]])
'''

# # Controller Code
# for event in pygame.event.get():
#     if event.type == pygame.QUIT:  # If user clicked close
#         done = True
#     elif event.type == pygame.KEYDOWN:  # If a key was pressed
#         if event.key == pygame.K_h:  # Reset to home position
#             reset_camera()
#         elif event.key in [pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d, pygame.K_r, pygame.K_f]:
#             move_camera(event.unicode)
#         elif event.key in [pygame.K_q, pygame.K_e]:
#             rotate_camera(event.unicode)
#         elif event.key == pygame.K_o:  # Switch to Orthographic Projection
#             perspective_projection = False
#             buildProjectionMatrix()  # Rebuild projection matrix
#         elif event.key == pygame.K_p:  # Switch to Perspective Projection
#             perspective_projection = True
#             buildProjectionMatrix()  # Rebuild projection matrix
