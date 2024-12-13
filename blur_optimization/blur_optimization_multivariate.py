import numpy as np
import skimage.io as io
import skimage.filters as filters
import skimage.metrics as metrics
from scipy.optimize import minimize_scalar, minimize, differential_evolution
from PIL import Image
import matplotlib.pyplot as plt

import sys
sys.path.append('.')
from utils import plot_2_image, prepare_image, generate_noisy_image, IMAGE_PATH, metrics_computation, load_original_image

# compares first real image of dataset with first image of 1 particle simulation
# difference_metric(sigma) can return any metric that compares the two images, please try different ones

# Initial guess for sigmas
initial_guess = [100, 30]

# Bounds for sigmas (adjust as needed)
bounds = [(1, 300), (1, 300)]

TOL = 1e-4 # Convergence tolerance, the lower the better but the slower
MAXITER = 10 # doesnt seem to have a big impact



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

    def difference_metric(sigmas, verbose=True):
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
        gaussian_noise, poisson_noise = abs(sigmas)
        blurred = generate_noisy_image(gaussian_noise=gaussian_noise, poisson_noise=poisson_noise)

        ssim, mse, psnr = metrics_computation(original_image, blurred, str=False)
        
        # Combined metric: minimize negative SSIM and maximize PSNR, minimize MSE
        combined_difference = -ssim + mse / (psnr + 1e-10)

        metric_log.append([ssim, mse, psnr, combined_difference])
        
        if verbose:
            print(f"Sigmas: {gaussian_noise:.4f}, {poisson_noise:.4f} | SSIM: {ssim:.4f} | MSE: {mse:.4f} | PSNR: {psnr:.4f}")
                
        return combined_difference #-ssim #mse

    

    
    # # Perform optimization
    # result = minimize(
    #     difference_metric, 
    #     initial_guess, 
    #     method='L-BFGS-B',  # Allows bound constraints
    #     bounds=bounds
    # )

    # Perform optimization using Differential Evolution
    result = differential_evolution(
        difference_metric,
        bounds=bounds,
        strategy='best1bin',  # Robust crossover strategy
        popsize=15,  # Larger population for more thorough search
        maxiter=MAXITER,  # More iterations
        tol= TOL,#1e-7,  # Convergence tolerance
        recombination=0.7,  # Crossover rate
        mutation=(0.5, 1.5)  # Mutation scaling factor range
    )
    # Get optimal sigmas and corresponding blurred image
    optimal_sigmas = result.x
    
    return optimal_sigmas, -result.fun, np.array(metric_log)




def plot_metric_log(metric_log):
    iterations = np.arange(1, metric_log.shape[0]+1)
    fig, ax = plt.subplots(4, 1, figsize=(10, 20))
    ax[0].scatter(iterations, metric_log[:, 0], label='SSIM')
    #ax[0].set_title('SSIM')
    ax[0].set_xlabel('iterations')
    ax[0].set_ylabel('SSIM')
    ax[1].scatter(iterations, metric_log[:, 1], label='MSE')
    #ax[1].set_title('MSE')
    ax[1].set_xlabel('iterations')
    ax[1].set_ylabel('MSE')
    ax[2].scatter(iterations, metric_log[:, 2], label='PSNR')
    #ax[2].set_title('PSNR')
    ax[2].set_xlabel('iterations')
    ax[2].set_ylabel('PSNR')
    ax[3].scatter(iterations, metric_log[:, 3], label='Combined Difference')
    #ax[3].set_title('Combined Difference')
    ax[3].set_xlabel('iterations')
    ax[3].set_ylabel('Combined Difference')
    plt.tight_layout()
    plt.show()


def main():


    image_array = load_original_image()
    
    # Optimize blur
    (opt_gaussian_noise, opt_poisson_noise), similarity, metric_log = optimize_blur(image_array)
    
    # Print and save results
    print(f"Optimal Blur (gaussian_noise): {opt_gaussian_noise}")
    print(f"Optimal Blur (poisson_noise): {opt_poisson_noise}")
    print(f"Similarity Score: {similarity}")

    # Generate blurred image
    blurred_image = generate_noisy_image(opt_gaussian_noise, opt_poisson_noise)
    similarity = metrics.structural_similarity(image_array, blurred_image, 
                                             data_range=image_array.max() - image_array.min())
    plot_2_image(image_array, blurred_image, title=f"Original vs Generated Image with {opt_gaussian_noise:.4f} gaussian and {opt_poisson_noise:.4f} poisson noise\n{metrics_computation(image_array, blurred_image, str=True)}")
    plot_metric_log(metric_log)


if __name__ == "__main__":
    main()