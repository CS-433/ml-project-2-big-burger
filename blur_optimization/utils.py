import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('.')
from generate_images_fast.generate_images_fast import generateImagesAndEstimateD
from skimage import metrics

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


def prepare_image(img):
    # Convert to float
    img_float = np.array(img).astype(np.float64)
    
    # Normalize to 0-1 range
    img_normalized = (img_float - img_float.min()) / (img_float.max() - img_float.min())
    
    return img_normalized

def generate_noisy_image(poisson_noise, gaussian_noise, nthframe=0):
    """
    generate image with chosen amount of noise
    return the first image of the generated images
    """

    images = generateImagesAndEstimateD(nparticles,nframes,npixel,factor_hr,nposframe,D,dt,fwhm_psf,pixelsize,flux,background,poisson_noise, gaussian_noise, silent=True)[0]

    return prepare_image(images[0, nthframe, :, :])


def metrics_computation(image_array, noisy_image, str=False):

    # Compute Structural Similarity Index (SSIM)
    # Note: SSIM ranges from -1 to 1, where 1 is perfect similarity
    # We want to minimize, so we return the negative of SSIM
    similarity = metrics.structural_similarity(image_array, noisy_image, 
                                             data_range=image_array.max() - image_array.min())
    
    # Mean Squared Error (MSE)
    mse = metrics.mean_squared_error(image_array, noisy_image)
        
    # Peak Signal-to-Noise Ratio (PSNR)
    psnr = metrics.peak_signal_noise_ratio(image_array, noisy_image)

    return f"SSIM: {similarity:.4f} | MSE: {mse:.4f} | PSNR: {psnr:.4f}" if str else (similarity, mse, psnr)