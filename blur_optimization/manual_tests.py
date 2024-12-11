from utils import plot_2_image, prepare_image, generate_noisy_image, IMAGE_PATH
from PIL import Image
from skimage import metrics

GAUSSIAN_NOISE = 40
POISSON_NOISE = 50

def main():
    # Generate noisy image
    noisy_image = generate_noisy_image(POISSON_NOISE, GAUSSIAN_NOISE)


    image = Image.open(IMAGE_PATH)
    #image_array = np.array(image) / 18000 # Normalize by 18000
    image_array = prepare_image(image)

    # Compute Structural Similarity Index (SSIM)
    # Note: SSIM ranges from -1 to 1, where 1 is perfect similarity
    # We want to minimize, so we return the negative of SSIM
    similarity = metrics.structural_similarity(image_array, noisy_image, 
                                             data_range=image_array.max() - image_array.min())
    
    # Plot noisy image
    plot_2_image(image_array, noisy_image, title=f"Original vs Generated Image with {GAUSSIAN_NOISE} gaussian and {POISSON_NOISE} poisson noise. SSIM: {similarity}")

if __name__ == "__main__":
    main()