import numpy as np
import multiprocessing as mp
import time
import sys
sys.path.append('.')
from helpers import *
from tqdm import tqdm
from os import path, makedirs

CPU_COUNT = mp.cpu_count()

def generateImagesAndEstimateD(
    nparticles, nframes, npixel, factor_hr, nposframe, D, dt, fwhm_psf, pixelsize,
    flux, background, poisson_noise, gaussian_noise, normalizeValue=-1, save_dir=None, silent=False):
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
    - save_dir (str): Directory to save the images and D estimates.

    Returns:
    - image_array (ndarray): Array of shape (nparticles, nframes, npixel, npixel)
                             containing the simulated noisy images.
    - D_estimates (ndarray): Array of size (nparticles) with estimated diffusion coefficients.
    """
    image_array = np.zeros((nparticles, nframes, npixel, npixel))
    D_estimates = np.zeros(nparticles)
    time_range = np.arange(nframes * nposframe) * dt / nposframe

    if not silent: print(f"running program on each {CPU_COUNT} cpu core of the computer")


    # Simulate Brownian motion for all particles
    trajectories = _brownian_motion(nparticles, nframes, nposframe, D, dt, silent=silent)

    args = [(trajectories[p].copy(), nframes, npixel, factor_hr, nposframe, 
             fwhm_psf, pixelsize, flux, background, poisson_noise, gaussian_noise, 
             time_range, normalizeValue) for p in range(nparticles)]
    
    # Multiprocessing
    with mp.Pool(CPU_COUNT) as pool:
        results = list(tqdm(
                pool.imap(_generateImageforParticle, args),
                total=nparticles,
                desc="Generating images and estimating D",
                disable=silent
                ))    
    
    for p, (frame_noisy, D_estimate) in enumerate(results):
        image_array[p] = frame_noisy
        D_estimates[p] = D_estimate

    if save_dir is not None:

        if not path.isdir(save_dir):
            makedirs(save_dir)
            print(f"Directory {save_dir} didn't exist, it has now been created")

        np.save(path.join(save_dir,"images.npy"), image_array)
        np.save(path.join(save_dir,"D_estimates.npy"), D_estimates)
        print(f"Images and D estimates saved in {save_dir}")
    
    return image_array, D_estimates


def _brownian_motion(nparticles, nframes, nposframe, D, dt, startAtZero=False, silent=False):
    """
    Simulates the Brownian motion of particles over a specified number of frames 
    and interframe positions.

    Parameters:
    - nparticles (int): Number of particles to simulate.
    - nframes (int): Number of frames in the simulation.
    - nposframe (int): Number of interframe positions to calculate per frame.
    - D (float): Diffusion coefficient, influencing the spread of particle movement.
    - dt (float): Time interval between frames, affects particle displacement.
    - startAtZero (bool): If True, initializes the starting position at (0, 0).

    Returns:
    - trajectory (ndarray): Array of shape (nparticles, num_steps, 2) containing 
                            the x, y coordinates of each particle at each time step.
                            `num_steps` is calculated as `nframes * nposframe`.
    """
    num_steps = nframes * nposframe
    positions = np.zeros(2)
    trajectory = np.zeros((nparticles, num_steps, 2))
    
    # the formula for sigma might be wrong ?
    #https://en.wikipedia.org/wiki/Mean_squared_displacement#:~:text=In%20statistical%20mechanics%2C%20the%20mean,a%20reference%20position%20over%20time.
    #https://en.wikipedia.org/wiki/Gaussian_function
    sigma = np.sqrt(2 * D * dt / nposframe)
    #sigma = np.sqrt(4 * D * dt / nposframe)  # Standard deviation of step size based on D and dt

    #for p in range(nparticles):

    with mp.Pool(mp.cpu_count()) as pool:
        trajectory = np.array(list(tqdm(
                pool.imap(_generate_trajectory, [(num_steps, sigma, startAtZero)]*nparticles),
                total=nparticles,
                desc="Generating trajectories",
                disable=silent
                )))
         
    assert trajectory.shape == (nparticles, num_steps, 2), "Trajectory shape is incorrect"

    return trajectory


def _generate_trajectory(args):
    (num_steps, sigma, startAtZero) = args
    # Generate random steps in x and y directions based on normal distribution
    dxy = np.random.randn(num_steps, 2) * sigma
    if startAtZero:
        dxy[0, :] = [0, 0]  # Set starting position at origin for the first step
    # Calculate cumulative sum to get positions from step displacements
    positions = np.cumsum(dxy, axis=0)

    # if the trajectory is out of the frame, we redo the trajectory
    if np.any(np.abs(positions) > 1): 
        return _generate_trajectory(args)

    return positions

def _generateImageforParticle(arg):
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
    nparticles = 10
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
        flux, background, poisson_noise, gaussian_noise, save_dir="multiprocessing/results"
    )
    end = time.time()
    print("Time elapsed:", end - start)