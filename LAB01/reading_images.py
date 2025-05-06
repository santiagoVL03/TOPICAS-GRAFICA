import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

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

def get_histogram(image):
    if len(image.shape) == 2:
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        return hist, range(256)
    else:
        color = ('b', 'g', 'r')
        hists = []
        for i, col in enumerate(color):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            hists.append(hist)
        return hists, range(256)

def get_histogram_bw_images(image):
    hist = [0] * 256
    h, w = image.shape
    for i in range(h):
        for j in range(w):
            pixel_value = image[i, j]
            hist[pixel_value] += 1
    hist = np.array(hist)
    hist = hist.reshape(-1)
    return hist, range(256)

def convert_to_bw(image_path):
    image = cv2.imread(image_path)
    bw_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return bw_image

def equalize_histogram(image):
    equalized = image.copy()
        
    hist, _ = get_histogram_bw_images(image)
        
    cdf = np.zeros_like(hist)
    cdf[0] = hist[0]
    for i in range(1, len(hist)):
        cdf[i] = cdf[i-1] + hist[i]

    cdf_normalized = ((cdf - cdf.min()) * 255) / (cdf.max() - cdf.min() + 1e-8)
        
    height, width = image.shape
    for i in range(height):
        for j in range(width):
            equalized[i, j] = cdf_normalized[image[i, j]]
        
    return equalized

def main():
    for image_name in images:
        image = cv2.imread('Images/' + image_name)
        if image is None:
            print("Image not found. Please check the path.")
            return
        if not os.path.exists('equalizedimages'):
            os.makedirs('equalizedimages')
        bw_image = convert_to_bw('Images/' + image_name)
        
        cv2.imwrite(f'equalizedimages/{image_name}_original_.jpg', bw_image)
        get_new_image = equalize_histogram(bw_image)
        
        cv2.imwrite(f'equalizedimages/{image_name}_equalized_.jpg', get_new_image)
        hist, bins = get_histogram_bw_images(bw_image)
        hist_equalized, bins_equalized = get_histogram_bw_images(get_new_image)
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(bins, hist, color='black')
        plt.title('Histogram of Original Image')
        plt.xlim([0, 256])
        plt.subplot(1, 2, 2)
        plt.plot(bins_equalized, hist_equalized, color='black')
        plt.title('Histogram of Equalized Image')
        plt.xlim([0, 256])
        plt.savefig(f'equalizedimages/{image_name}_histogram_.png')
    
if __name__ == "__main__":
    main()