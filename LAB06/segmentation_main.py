import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import segmentation_functions as sf

images_canny = [
    '14.jpg', 
    'ab1.png', 
    'Lab03.jpg', 
    'lab05.jpg', 
    '20.jpg',
    'Lab2.jpg',
    'Lab04.jpg']

images_otsu = [
    '14.jpg', 
    'ab1.png', 
    '20.jpg']

def main():
    # Canny Edge Detection
    if not os.path.exists('canny_images'):
        os.makedirs('canny_images')
    
    for img_name in images_canny:
        img_path = os.path.join('../images', img_name)
        edges = sf.canny_segmentation(img_path, low_threshold=100, high_threshold=200)
        plt.figure(figsize=(10, 10))
        images_to_show = [
            ("Original", cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)),
            ("Canny Edges", edges)
        ]
        for title, img in images_to_show:
            if img is not None:
                plt.imshow(img, cmap='gray')
                plt.title(title)
                plt.axis('off')
            else:
                print(f"Image {title} could not be processed.")
            plt.tight_layout()
            base_name = os.path.splitext(img_name)[0]
            plt.savefig(f'canny_images/canny_{base_name}.png')
            plt.close()

    # Otsu's Thresholding
    if not os.path.exists('otsu_images'):
        os.makedirs('otsu_images')
    for img_name in images_otsu:
        img_path = os.path.join('../images', img_name)
        thresholded_img = sf.otsu_thresholding(img_path)
        plt.figure(figsize=(10, 10))
        images_to_show = [
            ("Original", cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)),
            ("Otsu Thresholding", thresholded_img)
        ]
        for title, img in images_to_show:
            if img is not None:
                plt.imshow(img, cmap='gray')
                plt.title(title)
                plt.axis('off')
            else:
                print(f"Image {title} could not be processed.")
            plt.tight_layout()
            base_name = os.path.splitext(img_name)[0]
            plt.savefig(f'otsu_images/otsu_{base_name}.png')
            plt.close()
            
if __name__ == "__main__":
    main()
    print("Segmentation completed and images saved.")