import cv2
import os
import matplotlib.pyplot as plt
import morph_functions as mf
import numpy as np

images = ['figures.jpg', 'figuraV2.jpg', 'figuraV.png', 'gato1.jpg', 'blancoNegro.png']

def main():
    for image_name in images:
        image_path = '../Images/' + image_name
        print(f"Processing {image_name}...")
        image = cv2.imread(image_path)
        if image is None:
            print(f"Image {image_name} not found. Please check the path.")
            return

        if not os.path.exists('morphedImages'):
            os.makedirs('morphedImages')
            
        kernel_cruz = np.array([[0, 1, 0],
                                [1, 1, 1],
                                [0, 1, 0]], np.uint8)
        kernel_diamante_5_x_5 = np.array([[0, 0, 1, 0, 0],
                                     [0, 1, 1, 1, 0],
                                     [1, 1, 1, 1, 1],
                                     [0, 1, 1, 1, 0],
                                     [0, 0, 1, 0, 0]], np.uint8)
        kernel_diamante_7_x_7 = np.array([[0, 0, 0, 1, 0, 0, 0],
                                     [0, 0, 1, 1, 1, 0, 0],
                                     [0, 1, 1, 1, 1, 1, 0],
                                     [1, 1, 1, 1, 1, 1, 1],
                                     [0, 1, 1, 1, 1, 1, 0],
                                     [0, 0, 1, 1, 1, 0, 0],
                                     [0, 0, 0, 1, 0, 0, 0]], np.uint8)
        kernel_barra_vertical = np.array([[0, 1, 0],
                                     [0, 1, 0],
                                     [0, 1, 0]], np.uint8)
        
        kernel_barra_horizontal = np.array([[0, 0, 0],
                                        [1, 1, 1],
                                        [0, 0, 0]], np.uint8)

        eroded_result_cruz = mf.erosion_image(image_path, kernel_cruz)
        dilated_result_cruz = mf.dilatation_image(image_path, kernel_cruz)
        eroded_result_diamante_5_x_5 = mf.erosion_image(image_path, kernel_diamante_5_x_5)
        dilated_result_diamante_5_x_5 = mf.dilatation_image(image_path, kernel_diamante_5_x_5)
        eroded_result_diamante_7_x_7 = mf.erosion_image(image_path, kernel_diamante_7_x_7)
        dilated_result_diamante_7_x_7 = mf.dilatation_image(image_path, kernel_diamante_7_x_7)
        eroded_result_barra_vertical = mf.erosion_image(image_path, kernel_barra_vertical, origin = 'edge', coor_origin=(1, 1))
        dilated_result_barra_vertical = mf.dilatation_image(image_path, kernel_barra_vertical, origin = 'edge', coor_origin=(1, 1))
        eroded_result_barra_horizontal = mf.erosion_image(image_path, kernel_barra_horizontal)
        dilated_result_barra_horizontal = mf.dilatation_image(image_path, kernel_barra_horizontal)
        
        plt.figure(figsize=(20, 10))

        images_to_show = [
            ("Original", image),
            ("Erosion Cruz", eroded_result_cruz),
            ("Dilatation Cruz", dilated_result_cruz),
            ("Erosion Diamante 5x5", eroded_result_diamante_5_x_5),
            ("Dilatation Diamante 5x5", dilated_result_diamante_5_x_5),
            ("Erosion Diamante 7x7", eroded_result_diamante_7_x_7),
            ("Dilatation Diamante 7x7", dilated_result_diamante_7_x_7),
            ("Erosion Barra Vertical", eroded_result_barra_vertical),
            ("Dilatation Barra Vertical", dilated_result_barra_vertical),
            ("Erosion Barra Horizontal", eroded_result_barra_horizontal),
            ("Dilatation Barra Horizontal", dilated_result_barra_horizontal)
        ]

        for i, (title, img) in enumerate(images_to_show):
            plt.subplot(2, 6, i + 1)
            plt.imshow(img, cmap='gray')
            plt.title(title)
            plt.xticks([]), plt.yticks([]) # type: ignore

        plt.tight_layout()
        base_name = os.path.splitext(image_name)[0]
        plt.savefig(f'morphedImages/{base_name}_morphing.png')
        plt.close()

if __name__ == "__main__":
    main()