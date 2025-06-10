import cv2
import numpy as np

def sobel_operator(image, axis):
    """
    Apply Sobel operator manually to compute gradients.
            
    Parameters:
    - image: Input grayscale image as a 2D numpy array.
    - axis: 'x' for horizontal gradient, 'y' for vertical gradient.
            
    Returns:
    - gradient: Gradient image as a 2D numpy array.
    """
    kernel_x = np.array([[-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]])
    kernel_y = np.array([[1, 2, 1],
                [0, 0, 0],
                [-1, -2, -1]])
            
    kernel = kernel_x if axis == 'x' else kernel_y
            
    rows, cols = image.shape
    gradient = np.zeros_like(image, dtype=np.float64)
            
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            region = image[i-1:i+2, j-1:j+2]
            gradient[i, j] = np.sum(region * kernel)
                
    return gradient

def canny_segmentation(img_path, low_threshold=100, high_threshold=200):
    """
    Perform Canny edge detection on the input image.
    
    Parameters:
    - img_path: Path to the input image.
    - low_threshold: Lower threshold for hysteresis.
    - high_threshold: Upper threshold for hysteresis.
    
    Returns:
    - edges: The edges detected in the image.
    """
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("Image not found")
            return None
        # Step 1: Apply Gaussian Blur to reduce noise
        blurred_img = cv2.GaussianBlur(img, (5, 5), 1)

        # Step 2: Compute gradients using Sobel operator manually
        
        grad_x = sobel_operator(blurred_img, 'x')
        grad_y = sobel_operator(blurred_img, 'y')

        # Step 3: Compute gradient magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x) * (180 / np.pi)
        direction[direction < 0] += 180

        # Step 4: Non-maximum suppression
        suppressed = np.zeros_like(magnitude, dtype=np.uint8)
        rows, cols = magnitude.shape
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                angle = direction[i, j]
                q, r = 255, 255
                if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                    q = magnitude[i, j + 1]
                    r = magnitude[i, j - 1]
                elif 22.5 <= angle < 67.5:
                    q = magnitude[i - 1, j + 1]
                    r = magnitude[i + 1, j - 1]
                elif 67.5 <= angle < 112.5:
                    q = magnitude[i - 1, j]
                    r = magnitude[i + 1, j]
                elif 112.5 <= angle < 157.5:
                    q = magnitude[i - 1, j - 1]
                    r = magnitude[i + 1, j + 1]

                if magnitude[i, j] >= q and magnitude[i, j] >= r:
                    suppressed[i, j] = magnitude[i, j]
                else:
                    suppressed[i, j] = 0

        # Step 5: Double thresholding
        strong_edges = suppressed > high_threshold
        weak_edges = (suppressed >= low_threshold) & (suppressed <= high_threshold)

        edges = np.zeros_like(suppressed, dtype=np.uint8)
        edges[strong_edges] = 255
        edges[weak_edges] = 75

        # Step 6: Edge tracking by hysteresis
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if edges[i, j] == 50:  # Weak edge
                    if np.any(edges[i-1:i+2, j-1:j+2] == 255):  # Check neighbors
                        edges[i, j] = 255
                    else:
                        edges[i, j] = 0
        return edges
    except Exception as e:
        print(f"Error in Canny segmentation: {e}")
        return None
    
def canny_cv2 (img_path, low_threshold=100, high_threshold=200):
    """
    Perform Canny edge detection using OpenCV.
    
    Parameters:
    - img_path: Path to the input image.
    - low_threshold: Lower threshold for hysteresis.
    - high_threshold: Upper threshold for hysteresis.
    
    Returns:
    - edges: The edges detected in the image.
    """
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("Image not found")
            return None
        
        # Step 1: Apply Canny edge detection
        edges = cv2.Canny(img, low_threshold, high_threshold)
        
        return edges
    except Exception as e:
        print(f"Error in Canny edge detection: {e}")
        return None

def otsu_thresholding(img_path):
    """
    Perform Otsu's thresholding on the input image.
    
    Parameters:
    - img_path: Path to the input image.
    
    Returns:
    - thresholded_img: The image after applying Otsu's thresholding.
    """
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("Image not found")
            return None
        
        # Step 1: Calculate histogram
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        
        # Step 2: Normalize histogram
        hist = hist.ravel() / hist.sum()
        
        # Step 3: Calculate cumulative distribution function (CDF)
        cdf = hist.cumsum()
        
        # Step 4: Calculate mean level
        mean_level = np.arange(256)
        
        # Step 5: Calculate between-class variance
        sigma_b_squared = (cdf[-1] * (mean_level * cdf).cumsum() - (mean_level * cdf) ** 2) / (cdf[-1] ** 2)
        
        # Step 6: Find the threshold that maximizes the between-class variance
        optimal_threshold = np.argmax(sigma_b_squared)
        optimal_threshold = int(optimal_threshold)
        print(f"Optimal threshold found: {optimal_threshold}")
        
        # Step 7: Apply the threshold to create a binary image
        _, thresholded_img = cv2.threshold(img, optimal_threshold, 255, cv2.THRESH_BINARY)
        
        return thresholded_img
    except Exception as e:
        print(f"Error in Otsu's thresholding: {e}")
        return None