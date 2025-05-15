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

def convolution(image, kernel):
    kernel_height, kernel_width = kernel.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='reflect')
    filtered_image = np.zeros_like(image, dtype=np.float32)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            roi = padded_image[i:i + kernel_height, j:j + kernel_width]
            filtered_image[i, j] = np.sum(roi * kernel)

    return np.clip(filtered_image, 0, 255).astype(np.uint8)

def media_filter(image_path, kernel_size):
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size debe ser impar")

    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("No se pudo leer la imagen")
    
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    filtered_image = convolution(image, kernel)

    return image, filtered_image

def mediana_filter(image_path, kernel_size):
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size debe ser impar")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("No se pudo leer la imagen")

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = image.shape
    pad = kernel_size // 2
    padded_image = np.pad(image, pad, mode='reflect')
    filtered_image = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            roi = padded_image[i:i+kernel_size, j:j+kernel_size]
            filtered_image[i, j] = np.median(roi) # type: ignore
    return image,filtered_image

def roberts_filter(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("No se pudo leer la imagen")

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)

    gradient_x = convolution(image, kernel_x)
    gradient_y = convolution(image, kernel_y)

    filtered_image = np.sqrt(gradient_x**2 + gradient_y**2)
    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)

    return image, filtered_image

def sobel_filter(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("No se pudo leer la imagen")

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)

    gradient_x = convolution(image, kernel_x)
    gradient_y = convolution(image, kernel_y)

    filtered_image = np.sqrt(gradient_x**2 + gradient_y**2)
    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)

    return image, filtered_image

def prewitt_filter(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("No se pudo leer la imagen")

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)

    gradient_x = convolution(image, kernel_x)
    gradient_y = convolution(image, kernel_y)

    filtered_image = np.sqrt(gradient_x**2 + gradient_y**2)
    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)

    return image, filtered_image

def laplace_filter(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("No se pudo leer la imagen")

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)

    filtered_image = cv2.filter2D(image, -1, kernel)
    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)

    return image, filtered_image