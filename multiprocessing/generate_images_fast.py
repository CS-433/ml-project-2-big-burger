import numpy as np
import multiprocessing as mp
import time
import sys
sys.path.append('.')
from helpers import *
from tqdm import tqdm

def generateImagesAndEstimateD(
    nparticles, nframes, npixel, factor_hr, nposframe, D, dt, fwhm_psf, pixelsize,
    flux, background, poisson_noise, gaussian_noise, normalizeValue=-1):
    """
    Generates the full pipeline of images and estimates the diffusion coefficient (D) for each particle.

    Parameters:
    - nparticles (int): Number of particles.
    - nframes (int): Number of frames to generate per particle.
    - npixel (int): Number of pixels for the image (square grid).
    - factor_hr (int): High-resolution scaling factor.
    - nposframe (int): Number of positions within each frame.
    - D (float): Diffusion coefficient for Brownian motion simulation.
    - dt (float): Time interval between frames.
    - fwhm_psf (float): Full width at half maximum for the PSF.
    - pixelsize (float): Pixel size in nanometers.
    - flux (float): Photon flux of the particles.
    - background (float): Background intensity level.
    - poisson_noise (float): Poisson noise scaling factor.
    - gaussian_noise (float): Gaussian noise standard deviation.

    Returns:
    - image_array (ndarray): Array of shape (nparticles, nframes, npixel, npixel)
                             containing the simulated noisy images.
    - D_estimates (ndarray): Array of size (nparticles) with estimated diffusion coefficients.
    """
    image_array = np.zeros((nparticles, nframes, npixel, npixel))
    D_estimates = np.zeros(nparticles)
    time_range = np.arange(nframes * nposframe) * dt / nposframe

    # Simulate Brownian motion for all particles
    trajectories = brownian_motion(nparticles, nframes, nposframe, D, dt)

    args = [(trajectories[p].copy(), nframes, npixel, factor_hr, nposframe, 
             fwhm_psf, pixelsize, flux, background, poisson_noise, gaussian_noise, 
             time_range, normalizeValue) for p in range(nparticles)]
    

    cpu_count = mp.cpu_count()
    print(f"running programm on each {cpu_count} cpu core of the computer")
    # Multiprocessing
    with mp.Pool(cpu_count) as pool:
        results = list(tqdm(
                pool.imap(generateImageforParticle, args),
                total=nparticles,
                desc="Generating images and estimating D"
                ))    
    
    for p, (frame_noisy, D_estimate) in enumerate(results):
        image_array[p] = frame_noisy
        D_estimates[p] = D_estimate
    
    return image_array, D_estimates


def generateImageforParticle(arg):
    """
    Generates the images for a single particle and estimates the diffusion coefficient (D)
    """

    (trajectory, nframes, npixel, factor_hr, nposframe, 
    fwhm_psf, pixelsize, flux, background, poisson_noise, gaussian_noise, 
    time_range, normalizeValue) = arg

    frame_hr = np.zeros((nframes, npixel * factor_hr, npixel * factor_hr))
    frame_noisy = np.zeros((nframes, npixel, npixel))

    for k in range(nframes):
        start = k * nposframe
        end = (k + 1) * nposframe
        trajectory_segment = trajectory[start:end, :]
        xtraj = trajectory_segment[:, 0]
        ytraj = trajectory_segment[:, 1]

        # Generate frames
        for pos in range(nposframe):
            frame_spot = gaussian_2d(
                xtraj[pos], ytraj[pos], 2.35 * fwhm_psf / pixelsize,
                npixel * factor_hr, flux
            )
            frame_hr[k] += frame_spot

        # Downsample and add noise
        frame_lr = block_reduce(frame_hr[k], block_size=factor_hr, func=np.mean)
        frame_noisy[k] = add_noise_background(frame_lr, background, poisson_noise, gaussian_noise, normalizeValue)

    # Estimate D from the trajectory
    msd = mean_square_displacement(trajectory)
    D_estimate = estimateDfromMSD(msd, time_range)

    return (frame_noisy, D_estimate)



if __name__ == "__main__":
    # Parameters
    nparticles = 100
    nframes = 100
    npixel = 100
    factor_hr = 5
    nposframe = 10
    D = 0.1
    dt = 0.01
    fwhm_psf = 200
    pixelsize = 100
    flux = 1000
    background = 100
    poisson_noise = 0.1
    gaussian_noise = 10

    # Generate images and estimate D
    start = time.time()
    image_array, D_estimates = generateImagesAndEstimateD(
        nparticles, nframes, npixel, factor_hr, nposframe, D, dt, fwhm_psf, pixelsize,
        flux, background, poisson_noise, gaussian_noise
    )
    end = time.time()
    print("Time elapsed:", end - start)