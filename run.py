from helpers import *
from Models.modelsUtils import *
from Models.simpleCNNModel import *
from Models.ResNetModel2D import *
from Models.ResNetModel3D import *
from Models.paperCNN import *
from Models.paperCNNNoPooling import *
import os.path
import os
import numpy as np
from PIL import Image
import json
import argparse

OUTPUT_DIR = "run_outputs/" # directory with weigths, losses, and plots
REAL_DATA_PATH = "real-data/blocks_64x64x16_70_01"
VALID_EXTENSIONS = [".tif"] # Valid image extensions
# Valid blocks in image names (blocks)
# can add others or define it as "all" to allow all blocks
# e.g. ["block-001", "block-002"] or "all"
VALID_BLOCK_NAMES = ["block-001"] 
REAL_DATA_MODEL = "resNet2D" # see models_params below for available models

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# Models used in a dictionnary comprehension. 
# To add  a new model, simply add it to the list with the wanted train fct

lr = 0.000001
models_params = {
    #"simpleCNN": {"class": SimpleCNN, "train_fct": train_model, "loaded_model": None, "criterion": nn.MSELoss(), "optimizer" : "adam", "lr" : lr},
    "resNet2D": {"class": ResNet2D, "train_fct": train_model, "loaded_model": None, "criterion": nn.MSELoss(), "optimizer" : "adam", "lr" : lr},
    "resNet3D": {"class": ResNet3D, "train_fct": train_model, "loaded_model": None, "criterion": nn.MSELoss(), "optimizer" : "adam", "lr" : lr},
    #"paperCNNAdam": {"class": PaperCnn, "train_fct": train_model, "loaded_model": None, "criterion": nn.MSELoss(), "optimizer" : "adam", "lr" : lr},
    "paperCNNSGD": {"class": PaperCnn, "train_fct": train_model, "loaded_model": None, "criterion": nn.MSELoss(), "optimizer" : "sgd", "lr" :lr},
    #"paperCNNNoPool": {"class": PaperCnn, "train_fct": train_model, "loaded_model": None, "criterion": nn.MSELoss(), "optimizer" : "adam", "lr" :lr}
}

def main():

    print(f"Output Directory: {OUTPUT_DIR}\nReal Data Path: {REAL_DATA_PATH}\nValid Extensions: {VALID_EXTENSIONS}\nValid Block Names: {VALID_BLOCK_NAMES}\nReal Data Model: {REAL_DATA_MODEL}\n")
    print(f"Models: {list(models_params.keys())}")
    print(f"Using device: {device}\n")

    print("Loading losses")
    # Load the losses
    load_loss_history()

    print("Loading existing models")
    # load the models and losses
    load_models()
    
    # print predictions
    print(f"Predicting on real images, only using the {REAL_DATA_MODEL} model:\n(that can be changed inside the load_real_images_and_predict function inside the run.py file)")
    images_paths = find_real_images(REAL_DATA_PATH)
    predict_on_real_images(images_paths=images_paths)
    print("Plotting real images:")
    plot_real_images(images_paths=images_paths)


def load_models() -> dict:
    """Load the models from the modelsData/ folder and store them in the models_params dictionary"""

    for name, params in models_params.items():
        class_ = params["class"]
        # Load the model weights
        loaded_model = class_().to(device)
        filepath = "modelsData/w_" + name + ".pth"
        if os.path.exists(filepath):
            loaded_model = load_model_weights(loaded_model, filepath)
            print(name, "Loaded existing weights")
        else:
            print(name, "Did not find weights, loaded a new model")
            
        params["loaded_model"] = loaded_model  # Update the dictionary with the loaded model
        
        if(isinstance(params["optimizer"],str)):
            if params["optimizer"] == "adam":
                params["optimizer"] = optim.Adam(loaded_model.parameters(), lr=params["lr"])
            elif params["optimizer"] == "sgd":
                params["optimizer"] = optim.SGD(loaded_model.parameters(), lr=params["lr"], momentum=0.9)
            else: 
                params["optimizer"] = optim.Adam(loaded_model.parameters(), lr=params["lr"])

    return models_params

def load_loss_history() -> dict:
    """similar to load_models, prepare the loss history for the models"""

    totalEpochs = 0
    tr_loss_histories = {name: [] for name in models_params.keys()}
    val_loss_histories = {}

    for name, _ in models_params.items():

        filepath = "modelsData/l_" + name + ".npy"
        if os.path.exists(filepath):
            val_loss_histories[name] = np.load(filepath)
            print(name, "Loaded existing losses")
            totalEpochs = len(val_loss_histories[name])
        else:
            print(name, "Did not find losses, loaded an empty array")
            val_loss_histories[name] = np.array([])

    ds = "modelsData/allDs.npy"
    if os.path.exists(ds):
        allGeneratedDs = np.load(ds)
    else:
        allGeneratedDs = np.array([])  

    return totalEpochs, tr_loss_histories, val_loss_histories, allGeneratedDs 


def find_real_images(folder_path = "real-data/blocks_64x64x16_70_01"):
    """Finds all the real images in the folder path"""
    
    print("Loading only images with the extension(s)", VALID_EXTENSIONS, "and that start with", VALID_BLOCK_NAMES)

    images_paths = []

    # Recursively search for images in the folder
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(tuple(VALID_EXTENSIONS)):
                # Check if the file name starts with the valid block names or if all blocks are allowed
                if VALID_BLOCK_NAMES == "all":
                    images_paths.append(os.path.join(root, file))
                elif file.startswith(tuple(VALID_BLOCK_NAMES)):
                    images_paths.append(os.path.join(root, file))

    # sorting the images
    images_paths.sort()

    print("Found", len(images_paths), "images with the specified extension and naming pattern")

    return images_paths

def real_images_normalization(image):
    """used to normalize the real images to the same scale as the generated images"""    
    return np.array(image) / 18000

def real_images_normalization_daniel(image):
    """alternative normalization function, not used in the code, does not work as well as the previous one"""
    image = np.array(image, dtype=np.float32)
    image -= 300
    image /= 10000
    return np.clip(image, 0, 1)

def predict_on_real_images(images_paths: list , output_name="predictions"): 
    """predicts the diffusion coefficients on the real images"""

    # Initialize an empty list for predictions
    predictions = [] 
    results = {}

    params = models_params[REAL_DATA_MODEL]

    # Process each .tif file
    for image_path in images_paths:
        
        # Open the .tif file and load all 16 frames
        with Image.open(image_path) as img:
            frames = []
            for i in range(16):  # Assuming each .tif file has exactly 16 frames
                img.seek(i)  # Access frame i

                frame_array = real_images_normalization(img)
                
                frames.append(frame_array)
        
        # Convert frames to a NumPy array of shape (16, 64, 64)
        val_images = np.stack(frames, axis=0)
        
        # Query the model for predictions
        model = params["loaded_model"]
        model_preds = predict_diffusion_coefficients(model, val_images, device)
        model_preds_cpu = model_preds.cpu().numpy()
        predictions.append(model_preds_cpu)

        filename = os.path.basename(image_path)
        results[filename] = model_preds_cpu.tolist()

    # Convert predictions to a NumPy array for further processing or saving
    predictions = np.array(predictions)

    # Print shape of predictions for verification
    print("Predictions shape:", predictions.shape)
    print(predictions)

    # if it does not exist, create the predictions folder
    predictions_path = os.path.join(OUTPUT_DIR, "predictions")
    if not os.path.exists(predictions_path):
        os.makedirs(predictions_path)
        print(f"Created folder {predictions_path} to save predictions")

    # Save predictions if needed
    if output_name: 
        output_path = os.path.join(predictions_path, REAL_DATA_MODEL + "_" + os.path.splitext(output_name)[0] + "_" + os.path.basename(REAL_DATA_PATH))
        print("Predictions names are in the form of 'predictions/REAL_DATA_MODEL_output_name_REAL_DATA_PATH.{npy,json}'")
        np.save(output_path + ".npy", predictions)
        print(f"Predictions saved to {output_path}.npy")
        json.dump(results, open(output_path + ".json", 'w'), indent=4)
        print(f"Filenames mapped to predictions saved to {output_path}.json")


def plot_real_images(images_paths: list, output_name="real_images.svg"):
    """Plots the real images"""

    # Read all images and determine global min and max intensity
    images = []
    global_min = float("inf")
    global_max = float("-inf")

    for image_path in images_paths:

        #image_path = os.path.join(folder_path, file)
        image = Image.open(image_path)
        image_array = np.array(image)/10000
        images.append(image_array)
        global_min = min(global_min, image_array.min())
        global_max = max(global_max, image_array.max())

    # Display up to 16 images in 2 rows of 8 images each, on the same scale
    num_images = min(16, len(images))  # Ensure we don't exceed 16 images
    rows, cols = 2, 8  # 2 rows, 8 images per row
    plt.figure(figsize=(20, 8))
    plt.axis("off")
    plt.title(f"First {num_images} Real Images from \n{REAL_DATA_PATH}")
    plt.tight_layout()

    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i] , cmap="gray", vmin=global_min, vmax=global_max)
        plt.title(f"Image {i+1}")
        plt.axis("off")

    if output_name: plt.savefig(os.path.join(OUTPUT_DIR, output_name))
    plt.show()

    print(f"Global Min Intensity: {global_min}, Global Max Intensity: {global_max}")


if __name__ == "__main__":

    # Parse the CLI arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-r", "--real_data_path", type=str, default=None, help=f"Path to the real data, default is {REAL_DATA_PATH}")
    argparser.add_argument("-a", "--all", action="store_true", help=f"Run the model on all blocks inside the real data path, default is {VALID_BLOCK_NAMES}")

    args = argparser.parse_args()

    # Update the global variables if the CLI arguments are provided
    if args.real_data_path:
        REAL_DATA_PATH = args.real_data_path

        # input sanitization: check if the path exists
        assert os.path.exists(REAL_DATA_PATH), f"Path {REAL_DATA_PATH} does not exist"

        print(f"Real Data Path changed to {REAL_DATA_PATH}")
    if args.all:
        VALID_BLOCK_NAMES = "all"
        print("Running the model on all blocks in the real data path")

    main()