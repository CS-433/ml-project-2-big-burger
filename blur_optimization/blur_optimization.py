import numpy as np
import skimage.metrics as metrics
from scipy.optimize import minimize_scalar
from PIL import Image
import sys
sys.path.append('.')
import matplotlib.pyplot as plt
from utils import plot_2_image, prepare_image, generate_noisy_image, IMAGE_PATH

# compares first real image of dataset with first image of 1 particle simulation

BOUNDS = (1, 100)
METRIC = ['ssim', 'mse/psnr'][0]


def optimize_blur(original_image):
    """
    Find the optimal Gaussian blur amount to match the original image.
    
    Parameters:
    -----------
    original_image : ndarray
        The target image 
    
    Returns:
    --------
    optimal_sigma : float
        The optimal Gaussian blur sigma value
    min_difference : float
        The minimal difference metric between images
    """
    metric_log = []
    def difference_metric(sigma):
        """
        Compute the difference between the original and blurred image.
        Uses Structural Similarity Index (SSIM) as the difference metric.
        
        Parameters:
        -----------
        sigma : float
            Gaussian blur standard deviation
        
        Returns:
        --------
        difference : float
            Lower values indicate more similar images
        """
        blurred = generate_noisy_image(sigma, sigma)

        if METRIC == 'ssim':
            # Compute Structural Similarity Index (SSIM)
            # Note: SSIM ranges from -1 to 1, where 1 is perfect similarity
            # We want to minimize, so we return the negative of SSIM
            ssim = metrics.structural_similarity(original_image, blurred, 
                                                data_range=original_image.max() - original_image.min())
            metric_log.append([sigma, ssim])
            return -ssim  # Minimize negative SSIM (maximize actual SSIM)
        
        elif METRIC == 'mse/psnr':
            # Mean Squared Error (MSE)
            mse = metrics.mean_squared_error(original_image, blurred)
            
            # Peak Signal-to-Noise Ratio (PSNR)
            psnr = metrics.peak_signal_noise_ratio(original_image, blurred)

            difference = mse / (psnr + 1e-10)

            metric_log.append([sigma, difference])

            return difference  # Minimize MSE / PSNR
    
    # Perform optimization
    # Bounds are typically between 0 and 5 for sigma, adjust if needed
    result = minimize_scalar(difference_metric, bounds=BOUNDS, method='bounded')
    
    # Get optimal sigma and corresponding blurred image
    optimal_sigma = result.x
    
    return optimal_sigma, -result.fun, np.array(metric_log)


def plot_metric_log(metric_log):
    iterations = np.arange(1, metric_log.shape[0]+1)
    _, axes = plt.subplots(2, 1, figsize=(15, 8))

    axes[0].scatter(iterations, metric_log[:, 1], label='SSIM')
    axes[0].set_title('SSIM vs Iterations')
    axes[0].set_xlabel('Iterations')
    axes[0].set_ylabel('SSIM')
    axes[1].scatter(metric_log[:, 0], metric_log[:, 1], label='SSIM')
    axes[1].set_title('SSIM vs Gaussian and Poisson Noise')
    axes[1].set_xlabel('Gaussian and Poisson Noise')
    axes[1].set_ylabel('SSIM')
    plt.tight_layout()
    plt.show()

def prepare_image(img):
    # Convert to float
    img_float = np.array(img).astype(np.float64)
    
    # Normalize to 0-1 range
    img_normalized = (img_float - img_float.min()) / (img_float.max() - img_float.min())
    
    return img_normalized

def main(return_results=False, image_path=IMAGE_PATH):

    image = Image.open(image_path)
    #image_array = np.array(image) / 18000 # Normalize by 18000
    image_array = prepare_image(image)
    
    # Optimize blur
    sigma, similarity, metric_log = optimize_blur(image_array)

    if return_results: return sigma
    
    # Print and save results
    print(f"Optimal Blur (sigma): {sigma}")
    print(f"Similarity Score: {similarity}")
    # Generate blurred image
    blurred_image = generate_noisy_image(sigma, sigma)

    similarity = metrics.structural_similarity(image_array, blurred_image, 
                                             data_range=image_array.max() - image_array.min())

    plot_2_image(image_array, blurred_image, title=f"Original vs Generated Image with {sigma} gaussian and {sigma} poisson noise. SSIM: {similarity}")
    plot_metric_log(metric_log)
if __name__ == "__main__":
    main()