import cv2
import os
import matplotlib.pyplot as plt
import filter_functions as ff

images = ['mandrill.jpg', 
          'figuraV2.jpg', 
          'figures.jpg', 
          'blancoNegro.png', 
          'cameraman.jpg', 
          'figuraV.png',
          'gato1.jpg',
          'lena.png',
          'ninioB.png',
          'patrones1.jpg',
          'riceB.png',]


def main():
    for image_name in images:
        image_path = '../Images/' + image_name
        print(f"Processing {image_name}...")
        image = cv2.imread(image_path)
        if image is None:
            print(f"Image {image_name} not found. Please check the path.")
            return

        if not os.path.exists('filteredImages'):
            os.makedirs('filteredImages')

        original_image, filtered_image_media = ff.media_filter(image_path, 5)
        _, filtered_image_mediana = ff.mediana_filter(image_path, 5)
        _, filtered_image_roberts = ff.roberts_filter(image_path)
        _, filtered_image_sobel = ff.sobel_filter(image_path)
        _, filtered_image_prewitt = ff.prewitt_filter(image_path)
        _, filtered_image_laplace = ff.laplace_filter(image_path)

        plt.figure(figsize=(18, 10))

        images_to_show = [
            ("Original", original_image),
            ("Media", filtered_image_media),
            ("Mediana", filtered_image_mediana),
            ("Roberts", filtered_image_roberts),
            ("Sobel", filtered_image_sobel),
            ("Prewitt", filtered_image_prewitt),
            ("Laplace", filtered_image_laplace),
        ]

        for i, (title, img) in enumerate(images_to_show):
            plt.subplot(2, 4, i + 1)
            plt.imshow(img, cmap='gray')
            plt.title(title)
            plt.xticks([]), plt.yticks([]) # type: ignore

        plt.tight_layout()
        plt.savefig(f'filteredImages/{image_name}_all_filters.png')
        plt.close()

if __name__ == "__main__":
    main()