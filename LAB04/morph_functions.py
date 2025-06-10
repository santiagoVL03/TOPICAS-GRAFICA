import cv2
import os
import numpy as np

def dilatation_image(img_path, kernel, origin = 'center', coor_origin=(0, 0)):
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("Image not found")
            return
        # Get kernel dimensions and calculate the necessary padding sizes
        k_h, k_w = kernel.shape
        pad_h = k_h // 2
        pad_w = k_w // 2

        # Pad the image with zeros on the border
        padded_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
        dilated_img = np.zeros_like(img)

        # Get the positions in the kernel that are active (nonzero)
        active_positions = np.argwhere(kernel)

        # Iterate over each pixel of the original image
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
            # For each pixel, compute the maximum value over the neighborhood defined by the kernel
                values = []
                for pos in active_positions:
                    di, dj = pos
                    # Calculate the offset relative to the kernel's coordinate origin
                    # we assume the origin is the center of the kernel
                    # coor_origin is the top-left corner of the kernel in the padded image
                    # we donot use it at least origin is not 'center'
                    # Calculate the offset relative to the kernel's center
                    if origin != 'center':
                        # If origin is not center, adjust the offsets accordingly
                        offset_i = di - coor_origin[0]
                        offset_j = dj - coor_origin[1]
                    else:
                        offset_i = di - pad_h
                        offset_j = dj - pad_w
                    # Append the corresponding pixel from the padded image
                    values.append(padded_img[i + pad_h + offset_i, j + pad_w + offset_j])
                dilated_img[i, j] = np.max(values)
        return dilated_img
    except:
        print("CV2 error")
        return None
    
def erosion_image(img_path, kernel, origin='center', coor_origin=(0, 0)):
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("Image not found")
            return
        # Get kernel dimensions and calculate the necessary padding sizes
        k_h, k_w = kernel.shape
        pad_h = k_h // 2
        pad_w = k_w // 2

        # Pad the image with zeros on the border
        padded_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=255)
        eroded_img = np.zeros_like(img)

        # Get the positions in the kernel that are active (nonzero)
        active_positions = np.argwhere(kernel)
        # Iterate over each pixel of the original image
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                # For each pixel, compute the minimum value over the neighborhood defined by the kernel
                values = []
                for pos in active_positions:
                    di, dj = pos
                    # Calculate the offset relative to the kernel's coordinate origin
                    # we assume the origin is the center of the kernel
                    # coor_origin is the top-left corner of the kernel in the padded image
                    # we donot use it at least origin is not 'center'
                    # Calculate the offset relative to the kernel's center
                    if origin != 'center':
                        # If origin is not center, adjust the offsets accordingly
                        offset_i = di - coor_origin[0]
                        offset_j = dj - coor_origin[1]
                    else:
                        offset_i = di - pad_h
                        offset_j = dj - pad_w
                    # Append the corresponding pixel from the padded image
                    values.append(padded_img[i + pad_h + offset_i, j + pad_w + offset_j])
                eroded_img[i, j] = np.min(values)
        return eroded_img
    except:
        print("CV2 error")
        return None