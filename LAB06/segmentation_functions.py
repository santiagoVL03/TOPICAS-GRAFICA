import cv2
import numpy as np

import numpy as np
from PIL import Image
import math

def canny_cv2(image_path, low_threshold=100, high_threshold=200):
    """
    Detección de bordes de Canny usando OpenCV.
    
    Parámetros:
    - image_path: ruta de la imagen
    - low_threshold: umbral bajo para la detección de bordes
    - high_threshold: umbral alto para la detección de bordes
    
    Retorna:
    - edges: imagen con bordes detectados
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image not found at {image_path}")
    
    edges = cv2.Canny(image, low_threshold, high_threshold)
    
    return edges

def create_gaussian_filter(kernel_size, std_dev):
    """
    Construye un filtro gaussiano 2D.
    
    Argumentos:
        kernel_size: dimensión del kernel (entero impar)
        std_dev: desviación estándar de la gaussiana
    
    Retorna:
        matriz del filtro gaussiano normalizada
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    filter_matrix = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center_pos = kernel_size // 2
    sigma_squared = std_dev ** 2
    
    for row in range(kernel_size):
        for col in range(kernel_size):
            dist_x = row - center_pos
            dist_y = col - center_pos
            exponent = -(dist_x*dist_x + dist_y*dist_y) / (2 * sigma_squared)
            filter_matrix[row, col] = math.exp(exponent)
    
    # Normalización del kernel
    total_sum = np.sum(filter_matrix)
    return filter_matrix / total_sum

def convolve_2d(input_img, filter_kernel):
    """
    Realiza convolución 2D entre imagen y kernel.
    
    Argumentos:
        input_img: imagen de entrada
        filter_kernel: kernel de convolución
    
    Retorna:
        imagen convolucionada
    """
    img_height, img_width = input_img.shape
    kernel_h, kernel_w = filter_kernel.shape
    
    # Calcular padding necesario
    pad_vertical = kernel_h // 2
    pad_horizontal = kernel_w // 2
    
    # Aplicar padding por extensión de bordes
    padded_img = np.pad(input_img, 
                       ((pad_vertical, pad_vertical), (pad_horizontal, pad_horizontal)), 
                       mode='edge')
    
    output_img = np.zeros((img_height, img_width), dtype=np.float32)
    
    # Proceso de convolución
    for i in range(img_height):
        for j in range(img_width):
            region = padded_img[i:i+kernel_h, j:j+kernel_w]
            output_img[i, j] = np.sum(region * filter_kernel)
    
    return output_img

def compute_image_gradients(image_data):
    """
    Calcula gradientes usando máscaras de Sobel.
    
    Argumentos:
        image_data: imagen en escala de grises
    
    Retorna:
        tupla con (magnitud_gradiente, angulo_gradiente)
    """
    # Máscaras de Sobel para derivadas direccionales
    mask_horizontal = np.array([[-1, 0, 1],
                               [-2, 0, 2], 
                               [-1, 0, 1]], dtype=np.float32)
    
    mask_vertical = np.array([[-1, -2, -1],
                             [ 0,  0,  0],
                             [ 1,  2,  1]], dtype=np.float32)
    
    # Calcular derivadas parciales
    derivative_x = convolve_2d(image_data, mask_horizontal)
    derivative_y = convolve_2d(image_data, mask_vertical)
    
    # Magnitud del gradiente
    gradient_mag = np.sqrt(derivative_x**2 + derivative_y**2)
    
    # Dirección del gradiente en grados
    gradient_angle = np.degrees(np.arctan2(derivative_y, derivative_x))
    gradient_angle[gradient_angle < 0] += 180
    
    return gradient_mag, gradient_angle

def suppress_non_maxima(grad_magnitude, grad_direction):
    """
    Implementa supresión de no-máximos para adelgazar bordes.
    
    Argumentos:
        grad_magnitude: magnitud del gradiente
        grad_direction: dirección del gradiente en grados
    
    Retorna:
        imagen con bordes adelgazados
    """
    rows, cols = grad_magnitude.shape
    thinned_edges = np.zeros_like(grad_magnitude)
    
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            direction = grad_direction[r, c]
            current_magnitude = grad_magnitude[r, c]
            
            # Seleccionar vecinos según dirección del gradiente
            if (0 <= direction < 22.5) or (157.5 <= direction <= 180):
                # Dirección horizontal (0°)
                neighbor1 = grad_magnitude[r, c-1]
                neighbor2 = grad_magnitude[r, c+1]
            elif 22.5 <= direction < 67.5:
                # Dirección diagonal (45°)
                neighbor1 = grad_magnitude[r-1, c-1]
                neighbor2 = grad_magnitude[r+1, c+1]
            elif 67.5 <= direction < 112.5:
                # Dirección vertical (90°)
                neighbor1 = grad_magnitude[r-1, c]
                neighbor2 = grad_magnitude[r+1, c]
            else:  # 112.5° <= direction < 157.5°
                # Dirección diagonal (135°)
                neighbor1 = grad_magnitude[r-1, c+1]
                neighbor2 = grad_magnitude[r+1, c-1]
            
            # Conservar solo máximos locales
            if current_magnitude >= neighbor1 and current_magnitude >= neighbor2:
                thinned_edges[r, c] = current_magnitude
    
    return thinned_edges

def apply_dual_threshold(edge_image, threshold_low, threshold_high):
    """
    Aplica umbralización dual para clasificar píxeles de borde.
    
    Argumentos:
        edge_image: imagen con bordes adelgazados
        threshold_low: umbral inferior
        threshold_high: umbral superior
    
    Retorna:
        imagen umbralizada con bordes fuertes y débiles
    """
    result_image = np.zeros_like(edge_image, dtype=np.uint8)
    
    # Clasificar píxeles
    strong_pixel_mask = edge_image >= threshold_high
    weak_pixel_mask = (edge_image >= threshold_low) & (edge_image < threshold_high)
    
    result_image[strong_pixel_mask] = 255  # Bordes fuertes (blanco)
    result_image[weak_pixel_mask] = 127    # Bordes débiles (gris)
    
    return result_image

def track_edges_hysteresis(threshold_image):
    """
    Conecta bordes débiles con bordes fuertes mediante histéresis.
    
    Argumentos:
        threshold_image: imagen con umbralización dual
    
    Retorna:
        imagen final de bordes conectados
    """
    height, width = threshold_image.shape
    final_result = np.copy(threshold_image)
    
    # Direcciones de vecindario 8-conectado
    neighbor_offsets = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    
    # Iterativamente conectar bordes débiles con fuertes
    changed = True
    while changed:
        changed = False
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                if final_result[i, j] == 127:  # Píxel de borde débil
                    # Revisar vecindario
                    has_strong_neighbor = False
                    for di, dj in neighbor_offsets:
                        ni, nj = i + di, j + dj
                        if final_result[ni, nj] == 255:  # Vecino fuerte encontrado
                            has_strong_neighbor = True
                            break
                    
                    if has_strong_neighbor:
                        final_result[i, j] = 255  # Promover a borde fuerte
                        changed = True
    
    # Eliminar bordes débiles no conectados
    final_result[final_result == 127] = 0
    
    return final_result

def canny_edge_detection(image_path, low_threshold=100, high_threshold=200, 
                        sigma=1.0, kernel_size=5):
    """
    Detector de bordes Canny implementado desde cero.
    
    Argumentos:
        image_path: ruta del archivo de imagen
        low_threshold: umbral bajo para histéresis (default: 100)
        high_threshold: umbral alto para histéresis (default: 200)
        sigma: desviación estándar para suavizado gaussiano (default: 1.0)
        kernel_size: tamaño del kernel gaussiano (default: 5)
    
    Retorna:
        imagen binaria con bordes detectados
    """
    # Paso 1: Cargar imagen y convertir a escala de grises
    try:
        img_pil = Image.open(image_path)
        if img_pil.mode != 'L':
            img_pil = img_pil.convert('L')
        img_array = np.array(img_pil, dtype=np.float32)
    except Exception as e:
        raise FileNotFoundError(f"Error al cargar imagen: {e}")
    
    
    # Paso 2: Suavizado gaussiano para reducir ruido
    gauss_filter = create_gaussian_filter(kernel_size, sigma)
    smoothed_img = convolve_2d(img_array, gauss_filter)
    
    # Paso 3: Cálculo de gradientes
    magnitude, direction = compute_image_gradients(smoothed_img)
    
    # Paso 4: Supresión de no-máximos
    suppressed_img = suppress_non_maxima(magnitude, direction)
    
    # Paso 5: Umbralización dual
    thresholded_img = apply_dual_threshold(suppressed_img, low_threshold, high_threshold)
    
    # Paso 6: Seguimiento por histéresis
    edge_result = track_edges_hysteresis(thresholded_img)
    
    return edge_result

def otsu_simple_cv2(img_path):
    """
    Perform Otsu's thresholding using OpenCV.
    
    Parameters:
    - img_path: Path to the input image.
    
    Returns:
    - thresholded_img: The image after applying Otsu's thresholding.
    """
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:

            return None
        
        # Step 1: Apply Otsu's thresholding
        _, thresholded_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresholded_img
    except Exception as e:
        return None

def otsu_manual(region):
    # Flatten and compute histogram
    hist = np.bincount(region.ravel(), minlength=256)
    total = region.size

    sum_total = np.dot(np.arange(256), hist)
    sum_b, w_b, w_f, var_max, threshold = 0.0, 0, 0, 0.0, 0

    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break

        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f

        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > var_max:
            var_max = var_between
            threshold = t

    return threshold

def otsu_adaptive_manual(img_path, window_size):
    """
    Apply adaptive Otsu thresholding manually on an image (grayscale).
    - img_path: path to image
    - window_size: size of the square window (must be odd)
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Couldn't load image")

    height, width = img.shape
    pad = window_size // 2
    padded_img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    result = np.zeros_like(img)

    for y in range(height):
        for x in range(width):
            y1, y2 = y, y + window_size
            x1, x2 = x, x + window_size
            window = padded_img[y1:y2, x1:x2]
            threshold = otsu_manual(window)
            result[y, x] = 255 if img[y, x] > threshold else 0

    return result

