import numpy as np
import skimage.io as io
import skimage.filters as filters
import skimage.metrics as metrics
from scipy.optimize import minimize_scalar
from PIL import Image
import sys
sys.path.append('.')
from generate_images_fast.generate_images_fast import generateImagesAndEstimateD
import matplotlib.pyplot as plt

# compares first real image of dataset with first image of 1 particle simulation
# picking first image in real data
IMAGE_PATH = "real-data/blocks_64x64x16_70_01/block-001-6.658-0.057-456.tif"
BOUNDS = (1, 100)

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
        # Compute Structural Similarity Index (SSIM)
        # Note: SSIM ranges from -1 to 1, where 1 is perfect similarity
        # We want to minimize, so we return the negative of SSIM
        ssim = metrics.structural_similarity(original_image, blurred, 
                                             data_range=original_image.max() - original_image.min())
        metric_log.append([sigma, ssim])
        return -ssim  # Minimize negative SSIM (maximize actual SSIM)
    
    # Perform optimization
    # Bounds are typically between 0 and 5 for sigma, adjust if needed
    result = minimize_scalar(difference_metric, bounds=BOUNDS, method='bounded')
    
    # Get optimal sigma and corresponding blurred image
    optimal_sigma = result.x
    
    return optimal_sigma, -result.fun, np.array(metric_log)

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

def main():
    image = Image.open(IMAGE_PATH)
    #image_array = np.array(image) / 18000 # Normalize by 18000
    image_array = prepare_image(image)
    
    # Optimize blur
    sigma, similarity, metric_log = optimize_blur(image_array)
    
    # Print and save results
    print(f"Optimal Blur (sigma): {sigma}")
    print(f"Similarity Score: {similarity}")
    # Generate blurred image
    blurred_image = generate_noisy_image(sigma, sigma)
    plot_2_image(image_array, blurred_image, title=f"Original vs Generated Image with {sigma} gaussian and {sigma} poisson noise")
    plot_metric_log(metric_log)
if __name__ == "__main__":
    main()