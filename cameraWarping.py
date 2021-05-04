import cv2
import os
import shutil
import numpy as np

if __name__ == '__main__':

    # Read base image (top view)
    src = cv2.imread('./strip.png')
    if src is None:
        print("Could not find/load image!")
        exit(0)

    # Clear any existing images in folder
    if os.path.exists('./warped_images'):
        shutil.rmtree('./warped_images')
        print("Deleted images and folder")

    # Make sure output folder exists
    if not os.path.exists('./warped_images'):
        os.makedirs('./warped_images')

    # Create template image to fill in warped image
    dst = np.ndarray(shape=src.shape, dtype=src.dtype)
    h , w = src.shape[:2]

    # Implementation specifics, simulating airport approach
    rotXdeg = 10
    start_dist = 4000
    min_dist = 500
    step_size = 5
    z_deviation = 0.5
    z_min = 90 - z_deviation
    x_decay = 0.01
    rotZdeg = 90

    for dist in range(start_dist, min_dist, -step_size):

        rotXdeg -= x_decay

        rotZdeg = (2 * z_deviation) * np.random.random_sample() + z_min

        rotX = np.deg2rad(rotXdeg - 90)
        rotZ = np.deg2rad(rotZdeg - 90)

        #Projection 2D -> 3D matrix
        A1= np.matrix([[1, 0, -w/2],
                       [0, 1, -h/2],
                       [0, 0, 0   ],
                       [0, 0, 1   ]])

        # Rotation matrices around the X,Y,Z axis
        RX = np.matrix([[1,           0,            0, 0],
                        [0,np.cos(rotX),-np.sin(rotX), 0],
                        [0,np.sin(rotX),np.cos(rotX) , 0],
                        [0,           0,            0, 1]])

        RZ = np.matrix([[ np.cos(rotZ), -np.sin(rotZ), 0, 0],
                        [ np.sin(rotZ), np.cos(rotZ), 0, 0],
                        [            0,            0, 1, 0],
                        [            0,            0, 0, 1]])

        # Composed rotation matrix with (RX,RY,RZ)
        # No rotation along Y axis, so no need to consider RY
        R = RX * RZ

        #Translation matrix on the Z axis change dist will change the height
        T = np.matrix([[1,0,0,0],
                       [0,1,0,0],
                       [0,0,1,dist],
                       [0,0,0,1]])

        # Camera Intrisecs matrix 3D -> 2D
        # Set default value of f = 500
        f = 500
        A2= np.matrix([[f, 0, w/2,0],
                       [0, f, h/2,0],
                       [0, 0,   1,0]])

        # Final and overall transformation matrix
        H = A2 * (T * (R * A1))

        # Apply matrix transformation
        cv2.warpPerspective(src, H, (w, h), dst, cv2.INTER_CUBIC)

        # Crop out image
        dst_to_show = dst[480:-400]
        
        # Save image for testing by model
        cv2.imwrite("./warped_images/d_%d,z_%d.png" % (dist, rotZdeg), dst_to_show)
