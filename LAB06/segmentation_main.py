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
    'Lab04.jpg'
]

def main():
    os.makedirs('images_treshold', exist_ok=True)

    for img_name in images_canny:
        img_path = os.path.join('../images', img_name)
        image_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image_gray is None:
            print(f"Image {img_name} not found. Please check the path.")
            continue

        # Segmentaciones
        edges = sf.canny_edge_detection(img_path, low_threshold=50, high_threshold=100, sigma=0.5)
        edges_cv2 = sf.canny_cv2(img_path, low_threshold=50, high_threshold=100)
        otsu_thresholded = sf.otsu_adaptive_thresholding(img_path, 10)
        otsu_thresholded_cv2 = sf.otsu_cv2(img_path)
        otsu_simple = sf.otsu_thresholding(img_path)

        # Crear figura con espacio para 8 imágenes (3x3 grid)
        plt.figure(figsize=(18, 18))

        images_to_show = [
            ("Original", image_gray),
            ("Canny Segmentation", edges),
            ("Canny OpenCV", edges_cv2),
            ("Otsu Adaptive", otsu_thresholded),
            ("Otsu OpenCV", otsu_thresholded_cv2),
            ("Otsu Simple", otsu_simple)
        ]

        # Mostrar imágenes segmentadas
        for i, (title, img) in enumerate(images_to_show):
            plt.subplot(3, 3, i + 1)
            plt.imshow(img, cmap='gray')
            plt.title(title)
            plt.xticks([]), plt.yticks([]) # type: ignore

        # Histograma de la imagen original
        plt.subplot(3, 3, 7)
        plt.hist(image_gray.ravel(), bins=256, range=[0, 256], color='blue') # type: ignore
        plt.title("Histogram - Original")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")

        # Histograma de la imagen Otsu Simple
        if otsu_thresholded_cv2 is not None:
            plt.subplot(3, 3, 8)
            plt.hist(otsu_thresholded_cv2.ravel(), bins=256, range=[0, 256], color='green') # type: ignore
            plt.title("Histogram - Otsu tresholding")
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
