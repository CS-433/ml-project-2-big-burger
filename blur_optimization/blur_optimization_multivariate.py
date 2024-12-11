import numpy as np
import skimage.io as io
import skimage.filters as filters
import skimage.metrics as metrics
from scipy.optimize import minimize_scalar, minimize, differential_evolution
from PIL import Image
import matplotlib.pyplot as plt

import sys
sys.path.append('.')
from generate_images_fast.generate_images_fast import generateImagesAndEstimateD

# compares first real image of dataset with first image of 1 particle simulation
# difference_metric(sigma) can return any metric that compares the two images, please try different ones

# picking first image in real data
IMAGE_PATH = "real-data/blocks_64x64x16_70_01/block-001-6.658-0.057-456.tif"
# Initial guess for sigmas
initial_guess = [100, 30]

# Bounds for sigmas (adjust as needed)
bounds = [(1, 300), (1, 300)]

TOL = 1e-4 # Convergence tolerance, the lower the better but the slower
MAXITER = 10 # doesnt seem to have a big impact


# Hyperparameters for simulation
nparticles = 1   # Number of particles
nframes = 1    # Number of steps in the simulation
nposframe = 10    # Number of position per frame
dt = 0.01        # Integration time frame in second (time between two frames)
DGen = 20000        # Diffusion coefficient in nm^2 per s (=0.000001 (um^2)/s)
num_steps = nframes*nposframe
# Hyperparameters for image generation
npixel = 64 # number of image pixels
pixelsize = 100 # in nm 
fwhm_psf = 200 # full width half maximum (emulates microscope)
factor_hr = 5 # image high resulution factor
flux = 100 # number of photons per s
poisson_noise = 100 
gaussian_noise = 10
background = 100 # base background value
normalizeValue = 1000 # value by which all samples will be normalized ! Needs to be fixed and the same for all images !
n_val_im = 50
D = 10


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

        # Compute Structural Similarity Index (SSIM)
        # Note: SSIM ranges from -1 to 1, where 1 is perfect similarity
        # We want to minimize, so we return the negative of SSIM
        ssim = metrics.structural_similarity(original_image, blurred, 
                                             data_range=original_image.max() - original_image.min())

        # Mean Squared Error (MSE)
        mse = metrics.mean_squared_error(original_image, blurred)
        
        # Peak Signal-to-Noise Ratio (PSNR)
        psnr = metrics.peak_signal_noise_ratio(original_image, blurred)
        
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

def generate_noisy_image(poisson_noise, gaussian_noise, nthframe=0):
    """
    generate image with chosen amount of noise
    return the first image of the generated images
    """

    images = generateImagesAndEstimateD(nparticles,nframes,npixel,factor_hr,nposframe,D,dt,fwhm_psf,pixelsize,flux,background,poisson_noise, gaussian_noise, silent=True)[0]

    return prepare_image(images[0, nthframe, :, :])


def plot_2_image(image, blurred, title=""):
    """
    Plot the original and blurred images side by side.
    
    Parameters:
    -----------
    image : ndarray
        The original image
    blurred : ndarray
        The blurred image
    title : str
        Title for
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title("Original Image")
    ax[1].imshow(blurred, cmap='gray')
    ax[1].set_title("Generated Image")
    plt.suptitle(title)
    plt.show()

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

def prepare_image(img):
    # Convert to float
    img_float = np.array(img).astype(np.float64)
    
    # Normalize to 0-1 range
    img_normalized = (img_float - img_float.min()) / (img_float.max() - img_float.min())
    
    return img_normalized


def main():


    image = Image.open(IMAGE_PATH)
    #image_array = np.array(image) / 18000 # Normalize by 18000

    image_array = prepare_image(image)
    
    # Optimize blur
    (opt_gaussian_noise, opt_poisson_noise), similarity, metric_log = optimize_blur(image_array)
    
    # Print and save results
    print(f"Optimal Blur (gaussian_noise): {opt_gaussian_noise}")
    print(f"Optimal Blur (poisson_noise): {opt_poisson_noise}")
    print(f"Similarity Score: {similarity}")

    # Generate blurred image
    blurred_image = generate_noisy_image(opt_gaussian_noise, opt_poisson_noise)
    plot_2_image(image_array, blurred_image, title=f"Original vs Generated Image with {opt_gaussian_noise} gaussian and {opt_poisson_noise} poisson noise")
    plot_metric_log(metric_log)


if __name__ == "__main__":
    main()