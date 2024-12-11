import numpy as np
import skimage.metrics as metrics
from scipy.optimize import minimize_scalar
from PIL import Image
import sys
sys.path.append('.')
import matplotlib.pyplot as plt
from utils import plot_2_image, prepare_image, generate_noisy_image, IMAGE_PATH, metrics_computation

# compares first real image of dataset with first image of 1 particle simulation

BOUNDS = (1, 100)
METRIC = ['SSIM', 'MSE/PSNR'][1]


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

        ssim, mse, psnr = metrics_computation(original_image, blurred, str=False)

        if METRIC == 'SSIM':

            metric_log.append([sigma, ssim])
            return -ssim  # Minimize negative SSIM (maximize actual SSIM)
        
        elif METRIC == 'MSE/PSNR':

            ratio = mse / (psnr + 1e-10)

            metric_log.append([sigma, ratio])

            return ratio  # Minimize MSE / PSNR
    
    # Perform optimization
    # Bounds are typically between 0 and 5 for sigma, adjust if needed
    result = minimize_scalar(difference_metric, bounds=BOUNDS, method='bounded')
    
    # Get optimal sigma and corresponding blurred image
    optimal_sigma = result.x
    
    return optimal_sigma, -result.fun, np.array(metric_log)


def plot_metric_log(metric_log):
    iterations = np.arange(1, metric_log.shape[0]+1)
    _, axes = plt.subplots(2, 1, figsize=(15, 8))

    axes[0].scatter(iterations, metric_log[:, 1], label=METRIC)
    axes[0].set_title(f'{METRIC} vs Iterations')
    axes[0].set_xlabel('Iterations')
    axes[0].set_ylabel(METRIC)
    axes[1].scatter(metric_log[:, 0], metric_log[:, 1], label=METRIC)
    axes[1].set_title(f'{METRIC} vs Gaussian and Poisson Noise')
    axes[1].set_xlabel('Gaussian and Poisson Noise')
    axes[1].set_ylabel(METRIC)
    plt.tight_layout()
    plt.show()

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

    plot_2_image(image_array, blurred_image, title=f"Original vs Generated Image with {sigma:.4f} gaussian and {sigma:.4f} poisson noise\n{metrics_computation(image_array, blurred_image, str=True)}")
    plot_metric_log(metric_log)
if __name__ == "__main__":
    main()