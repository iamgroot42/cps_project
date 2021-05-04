import cv2
import os
import shutil
import numpy as np
import torch as ch
import torchgeometry as tgm


# Implementation specifics, simulating airport approach
rotXdeg = 10
start_dist = 1400
min_dist = 200
step_size = 5
z_deviation = 0.5
z_min = 90 - z_deviation
x_decay = 0.01
rotZdeg = 90


def ch_to_cv2(x):
    # Convert to channels-last
    x_ = x.transpose(1, 2, 0)
    vis = cv2.cvtColor(x_, cv2.COLOR_RGB2BGR)
    return vis


def get_H_matrix(dist, w, h, rotX, rotZ):
     #Projection 2D -> 3D matrix
    A1 = np.matrix([[1, 0, -w/2],
                    [0, 1, -h/2],
                    [0, 0, 0],
                    [0, 0, 1]])

    # Rotation matrices around the X,Y,Z axis
    RX = np.matrix([[1,           0,            0, 0],
                    [0, np.cos(rotX), -np.sin(rotX), 0],
                    [0, np.sin(rotX), np.cos(rotX), 0],
                    [0,           0,            0, 1]])

    RZ = np.matrix([[np.cos(rotZ), -np.sin(rotZ), 0, 0],
                    [np.sin(rotZ), np.cos(rotZ), 0, 0],
                    [0,            0, 1, 0],
                    [0,            0, 0, 1]])

    # Composed rotation matrix with (RX,RY,RZ)
    # No rotation along Y axis, so no need to consider RY
    R = RX * RZ

    #Translation matrix on the Z axis change dist will change the height
    T = np.matrix([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, dist],
                   [0, 0, 0, 1]])

    # Camera Intrisecs matrix 3D -> 2D
    # Set default value of f = 500
    f = 500
    A2 = np.matrix([[f, 0, w/2, 0],
                    [0, f, h/2, 0],
                    [0, 0,   1, 0]])

    # Final and overall transformation matrix
    H = A2 * (T * (R * A1))
    
    return H


def get_warped_images_torch(base_img):
    h, w = base_img.shape[1:]
    imgs = []
    rotXhere = rotXdeg

    distances, degrees, images = [], [], []
    for dist in range(start_dist, min_dist, -step_size):

        rotXhere -= x_decay
        rotZdeg = (2 * z_deviation) * np.random.random_sample() + z_min

        rotX = np.deg2rad(rotXhere - 90)
        rotZ = np.deg2rad(rotZdeg - 90)

        # Final and overall transformation matrix
        H = get_H_matrix(dist, w, h, rotX, rotZ)
        H = ch.from_numpy(H)

        img_warp = tgm.warp_perspective(base_img.unsqueeze(0), H, dsize=(h, w))

        imgs.append(img_warp)
        distances.append(dist)
        degrees.append(rotZdeg)
        
    return np.array(distances), np.array(degrees), ch.cat(imgs, 0).float()


def get_warped_images(base_image):
    # Create template image to fill in warped image
    dst = np.ndarray(shape=base_image.shape, dtype=base_image.dtype)
    h, w = base_image.shape[:2]
    rotXhere = rotXdeg

    distances, degrees, images = [], [], []
    for dist in range(start_dist, min_dist, -step_size):

        rotXhere -= x_decay

        rotZdeg = (2 * z_deviation) * np.random.random_sample() + z_min

        rotX = np.deg2rad(rotXhere - 90)
        rotZ = np.deg2rad(rotZdeg - 90)

        # Final and overall transformation matrix
        H = get_H_matrix(dist, w, h, rotX, rotZ)

        # Apply matrix transformation
        cv2.warpPerspective(base_image, H, (w, h), dst, cv2.INTER_CUBIC)

        distances.append(dist)
        degrees.append(rotZdeg)
        images.append(dst.copy())
    
    return distances, degrees, images


if __name__ == '__main__':

    # Read base image (top view)
    src = cv2.imread('./strip.jpg')
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    if src is None:
        print("Could not find/load image!")
        exit(0)

    # Clear any existing images in folder
    if os.path.exists('./warped_images'):
        shutil.rmtree('./warped_images')

    # Make sure output folder exists
    if not os.path.exists('./warped_images'):
        os.makedirs('./warped_images')

    warped_images = get_warped_images(src)

    for d, z, wi in zip(*warped_images):
        # Crop out image
        # dst_to_show = wi[75:-50]
        dst_to_show = wi[:]
        dst_to_show = cv2.cvtColor(dst_to_show, cv2.COLOR_RGB2BGR)

        # Save image for testing by model
        cv2.imwrite("./warped_images/d_%d,z_%d.png" % (d, z), dst_to_show)
