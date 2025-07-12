import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import segmentation_functions as sf

images_canny = [
    'Lab2.jpg',
    'Lab04.jpg',
    '14.jpg',
    'ab1.png',
    'Lab03.jpg',
    'lab05.jpg',
    '20.jpg',
]

def main():
    os.makedirs('images_treshold', exist_ok=True)

    for img_name in images_canny:
        img_path = os.path.join('../Images', img_name)
        image_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image_gray is None:
            print(f"Image {img_name} not found. Please check the path.")
            continue

        # Segmentaciones
        edges = sf.canny_edge_detection(img_path, low_threshold=50, high_threshold=100, sigma=0.5)
        edges_cv2 = sf.canny_cv2(img_path, low_threshold=50, high_threshold=100)
        otsu_simple_cv2 = sf.otsu_simple_cv2(img_path)
        otsu_adaptative_manual_10 = sf.otsu_adaptive_manual(img_path, 10)
        otsu_adaptative_manual_20 = sf.otsu_adaptive_manual(img_path, 20)
        otsu_adaptative_manual_30 = sf.otsu_adaptive_manual(img_path, 30)

        # Crear figura con espacio para 9 imágenes (3x3 grid)
        plt.figure(figsize=(18, 24))

        images_to_show = [
            ("Canny Segmentation", edges),
            ("Canny OpenCV", edges_cv2),
            ("Original", image_gray),
        ]

        # Mostrar imágenes segmentadas
        for i, (title, img) in enumerate(images_to_show):
            plt.subplot(4, 3, i + 1)
            plt.imshow(img, cmap='gray')
            plt.title(title)
            plt.xticks([]), plt.yticks([]) # type: ignore

        # Histograma de la imagen original
        plt.subplot(4, 3, 4)
        plt.hist(image_gray.ravel(), bins=256, range=[0, 256], color='blue') # type: ignore
        plt.title("Histogram - Original")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        
        images_to_show_otsu = [
            ("Otsu OpenCV", otsu_simple_cv2),
            ("Otsu Adaptative Manual size 10", otsu_adaptative_manual_10),
            ("Otsu Adaptative Manual size 20", otsu_adaptative_manual_20),
            ("Otsu Adaptative Manual size 30", otsu_adaptative_manual_30),
        ]
        
        for i, (title, img) in enumerate(images_to_show_otsu):
            if img is not None:
                plt.subplot(4, 3, i + 5)
                plt.imshow(img, cmap='gray')
                plt.title(title)
                plt.xticks([]), plt.yticks([]) # type: ignore

        if otsu_simple_cv2 is not None:
            plt.subplot(4, 3, 9)
            plt.hist(otsu_simple_cv2.ravel(), bins=256, range=[0, 256], color='red') # type: ignore
            plt.title("Histogram - Otsu OpenCV tresholding")
            plt.xlabel("Pixel Intensity")
            plt.ylabel("Frequency")
            
        if otsu_adaptative_manual_10 is not None:
            plt.subplot(4, 3, 10)
            plt.hist(otsu_adaptative_manual_10.ravel(), bins=256, range=[0, 256], color='green') # type: ignore
            plt.title("Histogram - Otsu Adaptative Manual size 10")
            plt.xlabel("Pixel Intensity")
            plt.ylabel("Frequency")
            
        if otsu_adaptative_manual_20 is not None:
            plt.subplot(4, 3, 11)
            plt.hist(otsu_adaptative_manual_20.ravel(), bins=256, range=[0, 256], color='orange') # type: ignore
            plt.title("Histogram - Otsu Adaptative Manual size 20")
            plt.xlabel("Pixel Intensity")
            plt.ylabel("Frequency")
            
        if otsu_adaptative_manual_30 is not None:
            plt.subplot(4, 3, 12)
            plt.hist(otsu_adaptative_manual_30.ravel(), bins=256, range=[0, 256], color='purple') # type: ignore
            plt.title("Histogram - Otsu Adaptative Manual size 30")
            plt.xlabel("Pixel Intensity")
            plt.ylabel("Frequency")
        
        # Guardar figura
        base_name = os.path.splitext(img_name)[0]
        plt.tight_layout()
        plt.savefig(f'images_treshold/{base_name}.png')
        plt.close()

if __name__ == "__main__":
    main()
    print("Segmentation and histograms completed.")
