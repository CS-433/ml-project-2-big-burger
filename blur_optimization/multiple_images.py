from utils import plot_2_image, prepare_image, generate_noisy_image, IMAGE_PATH
import sys
sys.path.append('.')
from blur_optimization import main as blur_optimization_main
from PIL import Image
from skimage import metrics
from tqdm import tqdm

# implements the univariate optimization of sigmas for multiple images

IMAGES_PATHS = [
    "real-data/blocks_64x64x16_70_01/block-001-6.658-0.057-456.tif",
    "real-data/blocks_64x64x16_83_01/block-001-0.092-0.134-0.tif",
    "real-data/blocks_64x64x16_104_01/block-001-1.465-0.061-16.tif"
]

def main():

    sigmas = []

    for image_path in tqdm(IMAGES_PATHS, total= len(IMAGES_PATHS), desc= "Optimizing sigmas"):

        sigma = blur_optimization_main(return_results= True, image_path= image_path)
        sigmas.append(sigma)
    
    print(f"Optimized gaussian and poisson noise: {sigmas}")

    avg_sigma = sum(sigmas) / len(sigmas)

    # generate images with the average sigma
    gen_image = generate_noisy_image(avg_sigma, avg_sigma)

    # import the original image
    image = Image.open(image_path)
    #image_array = np.array(image) / 18000 # Normalize by 18000
    image_array = prepare_image(image)

    # Compute Structural Similarity Index (SSIM)
    # Note: SSIM ranges from -1 to 1, where 1 is perfect similarity
    # We want to minimize, so we return the negative of SSIM
    similarity = metrics.structural_similarity(image_array, gen_image, 
                                            data_range=image_array.max() - image_array.min())
    
    # Plot noisy image
    plot_2_image(image_array, gen_image, title=f"Original vs Generated Image with average gaussian and poisson noise {avg_sigma}. SSIM: {similarity}")


if __name__ == "__main__":
    main()