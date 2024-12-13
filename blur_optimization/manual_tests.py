from utils import plot_2_image, prepare_image, generate_noisy_image, IMAGE_PATH, metrics_computation, load_original_image


GAUSSIAN_NOISE = 40
POISSON_NOISE = 50

def main():
    # Generate noisy image
    noisy_image = generate_noisy_image(POISSON_NOISE, GAUSSIAN_NOISE)
    image_array = load_original_image()

    # Plot noisy image
    plot_2_image(image_array, noisy_image, title=f"Original vs Generated Image with {GAUSSIAN_NOISE} gaussian and {POISSON_NOISE} poisson noise\n{metrics_computation(image_array, noisy_image, str=True)}")

if __name__ == "__main__":
    main()