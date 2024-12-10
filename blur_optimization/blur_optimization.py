import numpy as np
import skimage.io as io
import skimage.filters as filters
import skimage.metrics as metrics
from scipy.optimize import minimize_scalar
from PIL import Image

import sys
sys.path.append('.')
from generate_images_fast.generate_images_fast import generateImagesAndEstimateD

# compares first real image of dataset with first image of 1 particle simulation


# picking first image in real data
IMAGE_PATH = "real-data/blocks_64x64x16_70_01/block-001-6.658-0.057-456.tif"


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
        return -ssim  # Minimize negative SSIM (maximize actual SSIM)
    
    # Perform optimization
    # Bounds are typically between 0 and 5 for sigma, adjust if needed
    result = minimize_scalar(difference_metric, bounds=(0, 5), method='bounded')
    
    # Get optimal sigma and corresponding blurred image
    optimal_sigma = result.x
    
    return optimal_sigma, -result.fun

def generate_noisy_image(poisson_noise, gaussian_noise, nthframe=0):
    """
    generate image with chosen amount of noise
    return the first image of the generated images
    """

    images = generateImagesAndEstimateD(nparticles,nframes,npixel,factor_hr,nposframe,D,dt,fwhm_psf,pixelsize,flux,background,poisson_noise, gaussian_noise)[0]

    return images[0, nthframe, :, :]


def main():


    image = Image.open(IMAGE_PATH)
    image_array = np.array(image) / 18000 # Normalize by 18000

    print(image_array.shape)
    
    # Optimize blur
    sigma, similarity = optimize_blur(image_array)
    
    # Print and save results
    print(f"Optimal Blur (sigma): {sigma}")
    print(f"Similarity Score: {similarity}")


if __name__ == "__main__":
    main()