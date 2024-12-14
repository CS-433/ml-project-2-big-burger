from helpers import *
from modelsUtils import *
from simpleCNNModel import *
from ResNetModel2D import *
from ResNetModel3D import *
from paperCNN import *
from paperCNNNoPooling import *
import os.path
import os
import numpy as np
from PIL import Image
import json

RETRAIN = False
# directory with weigths, losses, and plots
OUTPUT_DIR = "run_outputs/"
REAL_DATA = True
REAL_DATA_PATH = "real-data/blocks_64x64x16_70_01"

# Hyperparameters for simulation
nparticles = 1000   # Number of particles
nframes = 16    # Number of steps in the simulation
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


device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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

    print(f"Retrain: {RETRAIN}\nReal Data: {REAL_DATA}\nReal Data Path: {REAL_DATA_PATH}\nOutput Directory: {OUTPUT_DIR}\n")
    print(f"Models: {list(models_params.keys())}\n")

    print("Loading models and losses")
    # Load the validation images
    val_images, valDs = load_validation_images()
    # Load the losses
    totalEpochs, tr_loss_histories, val_loss_histories, allGeneratedDs = load_loss_history()

    if RETRAIN:
        print("Retraining models")
        # Generate images and train the models
        totalEpochs, val_loss_histories, tr_loss_histories, allGeneratedDs = generate_images_and_train(val_images, valDs, val_loss_histories, tr_loss_histories)
    else:
        print("Loading existing models")
        # load the models and losses
        load_models()
    
    # print predictions
    if REAL_DATA: 
        print("Predicting on real images, only using the resNet2D model:\n(that can be changed inside the load_real_images_and_predict function inside the run.py file)")
        load_real_images_and_predict()
        print("Plotting real images:")
        load_and_plot_real_images()
    
    #print("Plotting results:")
    # Plot the generated D values
    print("Plotting generated D values")
    plot_generated_Ds(allGeneratedDs)
    if RETRAIN:
        # Plot the losses
        print("Plotting losses")
        plot_losses(tr_loss_histories, totalEpochs)
    # Plot the true vs predicted D values
    #print("Plotting true vs predicted D values")

    # DOES NOT WORK
    #coarseD_array, valDs_array =  plot_trueVpredicted_Dvalues(val_images, valDs)

    # Plot the true vs predicted D values with absolute error
    #print("Plotting true vs predicted D values with absolute error")
    #plot_trueVpredicted_Dvalues_absolute_error(coarseD_array, valDs_array, val_images)
    # Save the models, validation losses, and generated D values

    print("Saving models, validation losses, and generated D values")
    save_models_validations_losses_epochs(allGeneratedDs, val_loss_histories)


def load_models() -> dict:
    """
    Load the models from the models_params dictionary"""

    for name, params in models_params.items():
        class_ = params["class"]
        # Load the model weights
        loaded_model = class_().to(device)
        filename = "w_" + name + ".pth"
        filepath = os.path.join(OUTPUT_DIR, filename)
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
    totalEpochs = 0
    tr_loss_histories = {name: [] for name in models_params.keys()}
    val_loss_histories = {}

    for name, params in models_params.items():

        filename = "l_" + name + ".npy"
        filepath = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(filepath):
            val_loss_histories[name] = np.load(filepath)
            print(name, "Loaded existing losses")
            totalEpochs = len(val_loss_histories[name])
        else:
            print(name, "Did not find losses, loaded an empty array")
            val_loss_histories[name] = np.array([])

    ds = "allDs.npy"
    if os.path.exists(ds):
        allGeneratedDs = np.load(ds)
    else:
        allGeneratedDs = np.array([])  

    return totalEpochs, tr_loss_histories, val_loss_histories, allGeneratedDs 


def load_validation_images(trajectories_dir: str = OUTPUT_DIR) -> tuple: # REAL_DATA_PATH
    trajectories_path = os.path.join(trajectories_dir, "validation_trajectories.npy")
    val_trajs = np.load(trajectories_path)
    val_images, valDs = generateImagesAndEstimateDFromTrajs(val_trajs,n_val_im, nframes, npixel, factor_hr, nposframe, DGen, dt, fwhm_psf, pixelsize,flux, background, poisson_noise, gaussian_noise, normalizeValue)
    valDs = torch.tensor(valDs/1000, dtype=torch.float32, device=device)
    return val_images, valDs

def generate_images_and_train(val_images, valDs, val_loss_histories, tr_loss_histories, trajectories_dir=OUTPUT_DIR):
    epochs = 1
    N = 16 # Number of samples per iteration
    verbose = False # print in console
    for i in range(epochs):

        print(f"Generating images for iteration: {i}")

        images, estimatedDs =  generateImagesAndEstimateDMAXD(N, nframes, npixel, factor_hr, nposframe, DGen, dt, fwhm_psf, pixelsize,flux, background, poisson_noise, gaussian_noise, normalizeValue)
        
        # Divide the estimateDs by 10000 to get values in the range 0.5->70, then add them to the list of all Ds
        estimatedDs = estimatedDs / 1000
        allGeneratedDs = np.append(allGeneratedDs,estimatedDs)
        
        # Add channel dimension to images: (N, 16, 64, 64) -> (N, 16, 1, 64, 64)
        images = torch.tensor(images, dtype=torch.float32).unsqueeze(2)
        estimatedDs = torch.tensor(estimatedDs, dtype=torch.float32)

        for name, params in models_params.items():
            model = params["loaded_model"]
            train_fct = params["train_fct"]
            criterion = params["criterion"]
            optimizer = params["optimizer"]
            if(verbose):
                print("Training model:" , name)
            trained_model, tr_loss_history = train_fct(model, images, estimatedDs, device, criterion, optimizer, epochs=1, batch_size=16)
            params["loaded_model"] = trained_model

            # Compute validation loss on fixed set of images
            model_preds = predict_diffusion_coefficients(trained_model, val_images ,device)
            if(name == 'paperCNNSGD'):
                print(model_preds)
            loss = criterion(model_preds, valDs)
            val_loss_histories[name] = np.append(val_loss_histories[name],loss.item())

            # Store the single training epoch loss
            if isinstance(tr_loss_history, list) and len(tr_loss_history) > 0:
                tr_loss_histories[name].append(tr_loss_history[-1])  # Append last loss in the history
            elif isinstance(tr_loss_history, (float, int)):  # If it's a single loss value
                tr_loss_histories[name].append(tr_loss_history)
            else:
                print(f"Unexpected loss format for model {name}: {tr_loss_history}")

    # add up epochs for later use 
    totalEpochs = totalEpochs + epochs

    # Save the losses
    if trajectories_dir:
        for name, loss_history in tr_loss_histories.items():
            np.save(os.path.join(trajectories_dir, "l_" + name + ".npy"), loss_history)

    return totalEpochs, val_loss_histories, tr_loss_histories, allGeneratedDs

def plot_generated_Ds(allGeneratedDs, output_name="all_generated_Ds.svg"):

    output_path = os.path.join(OUTPUT_DIR, output_name) if output_name else None

    print(np.min(allGeneratedDs))
    print(np.mean(allGeneratedDs))
    print(np.max(allGeneratedDs))

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(np.clip(allGeneratedDs,0,80), bins=20, color='blue', edgecolor='black', alpha=0.7)

    # Add labels, title, and grid
    plt.xlabel('Estimated D Values', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Histogram of All Generated Estimated D Values', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show the plot
    plt.tight_layout()
    if output_path: plt.savefig(output_path)
    plt.show()

def load_real_images_and_predict(folder_path = "real-data/blocks_64x64x16_70_01", output_name="resNet2D_predictions"): #"predictions.npy"
    
    # Get a list of all files in the folder
    file_list = sorted(os.listdir(folder_path))  # Sorted lexicographically

    # Filter only files with valid image extensions and specific naming pattern
    valid_extensions = (".tif")
    image_files = [f for f in file_list if f.endswith(valid_extensions) and f.startswith("block-001")]

    # Initialize an empty list for predictions
    predictions = []
    results = {}

    params = models_params["resNet2D"]

    # Process each .tif file
    for file in image_files:
        image_path = os.path.join(folder_path, file)
        
        # Open the .tif file and load all 16 frames
        with Image.open(image_path) as img:
            frames = []
            for i in range(16):  # Assuming each .tif file has exactly 16 frames
                img.seek(i)  # Access frame i
                frame_array = np.array(img) / 18000  # Normalize by 18000
                frames.append(frame_array)
        
        # Convert frames to a NumPy array of shape (16, 64, 64)
        val_images = np.stack(frames, axis=0)
        
        # Query the model for predictions
        model = params["loaded_model"]
        model_preds = predict_diffusion_coefficients(model, val_images, device)
        model_preds_cpu = model_preds.cpu().numpy()
        predictions.append(model_preds_cpu)
        #print(f"Predictions for {file}:", model_preds_cpu)
        results[file] = model_preds_cpu.tolist()

    # Convert predictions to a NumPy array for further processing or saving
    predictions = np.array(predictions)

    # Print shape of predictions for verification
    print("Predictions shape:", predictions.shape)
    print(predictions)

    # Save predictions if needed
    if output_name: 
        output_path = os.path.join(OUTPUT_DIR, os.path.splitext(output_name)[0])
        np.save(output_path + ".npy", predictions)
        print(f"Predictions saved to {output_path}.npy")
        json.dump(results, open(output_path + ".json", 'w'), indent=4)
        print(f"Filenames mapped to predictions saved to {output_path}.json")


def load_and_plot_real_images(folder_path = "real-data/blocks_64x64x16_70_01", output_name="real_images.svg"):

    # Get a list of all files in the folder
    file_list = sorted(os.listdir(folder_path))  # Sorted lexicographically

    # Filter only files with valid image extensions (e.g., .tif, .jpg, .png)
    valid_extensions = (".tif")
    image_files = [f for f in file_list if f.endswith(valid_extensions) and f.startswith("block-001")]

    # Read all images and determine global min and max intensity
    images = []
    global_min = float("inf")
    global_max = float("-inf")

    for file in image_files:

        image_path = os.path.join(folder_path, file)
        image = Image.open(image_path)
        image_array = np.array(image)/10000
        images.append(image_array)
        global_min = min(global_min, image_array.min())
        global_max = max(global_max, image_array.max())

    # Display up to 16 images in 2 rows of 8 images each, on the same scale
    num_images = min(16, len(images))  # Ensure we don't exceed 16 images
    rows, cols = 2, 8  # 2 rows, 8 images per row
    plt.figure(figsize=(20, 8))

    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i] , cmap="gray", vmin=global_min, vmax=global_max)
        plt.title(f"Image {i+1}")
        plt.axis("off")

    plt.tight_layout()
    if output_name: plt.savefig(os.path.join(OUTPUT_DIR, output_name))
    plt.show()

    print(f"Global Min Intensity: {global_min}, Global Max Intensity: {global_max}")



def plot_loss_and_trueVpredicted(val_loss_histories, totalEpochs, val_images, valDs, output_names=["losses.svg", "true_vs_predicted.svg"]):

    # Output paths for the plots
    loss_output_path, pred_output_path = [os.path.join(OUTPUT_DIR, name) for name in output_names] if output_names else [None, None]

    plt.figure(figsize=(10, 6))
    for model_name, losses in val_loss_histories.items():
        plt.plot(range(0, totalEpochs ), np.clip(losses,0,100), label=model_name)

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss on validation set per Iteration for Each Model')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if loss_output_path: plt.savefig(loss_output_path)
    plt.show()

    plt.figure(figsize=(10, 8))

    # Iterate over all models to plot predictions vs true values
    for name, params in models_params.items():
        # Predictions and true values
        model = params["loaded_model"]
        model_preds = predict_diffusion_coefficients(model, val_images, device)
        valDs_tensor = torch.tensor(valDs, dtype=model_preds.dtype, device=model_preds.device)
        
        # Ensure predictions and true values are on the same device and flattened
        model_preds = model_preds.view(-1).cpu().numpy()  # Convert to numpy for plotting
        valDs_array = valDs_tensor.view(-1).cpu().numpy()  # Convert to numpy for plotting

        # Scatter plot for the current model
        plt.scatter(valDs_array, model_preds, alpha=0.7, label=f'{name} Predictions')

    # Plot the ideal line
    min_val, max_val = min(valDs_array), max(valDs_array)
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal (y = x)')

    # Add plot details
    plt.title("True vs Predicted D Values (All Models)")
    plt.xlabel("True D Values")
    plt.ylabel("Predicted D Values")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if pred_output_path: plt.savefig(pred_output_path)

    # Show the plot
    plt.show()


def plot_trueVpredicted_Dvalues(val_images, valDs, output_name="true_vs_predicted2.svg"):

    output_path = os.path.join(OUTPUT_DIR, output_name) if output_name else None

    # Initialize a vector to store the average predictions across all models
    #average_predictions = torch.zeros_like(valDs_tensor, dtype=torch.float32, device=device)
    #average_predictionsRot = torch.zeros_like(valDs_tensor, dtype=torch.float32, device=device)
    # Count the number of models contributing to the average
    model_count = 0

    # COARSE D doesnt WORK !!!!


    coarseD_tensor = torch.tensor(compute_coarseD_for_batch(val_images, dt), dtype=torch.float32, device=device) / 40
    #coarseD_tensor = torch.clip(coarseD_tensor,0,60)
    #lossCoarseD = criterion(coarseD_tensor, valDs_tensor)  # Loss for coarseD predictions
    coarseD_array = coarseD_tensor.view(-1).cpu().numpy()  # Convert to numpy for plotting
    plt.scatter(valDs, coarseD_array, color='purple', alpha=0.7, label='Coarse D Predictions', marker='^')
    #print("Coase D Loss:", lossCoarseD.item())


    # Iterate over all models to plot predictions vs true values
    for name, params in models_params.items():
        # Predictions and true values
        model = params["loaded_model"]
        criterion = params["criterion"]
        model_preds = predict_diffusion_coefficients(model, val_images, device)
        model_predsRot, individualPreds = predict_with_rotations(model, val_images, device)
        valDs_tensor = torch.tensor(valDs, dtype=model_preds.dtype, device=model_preds.device)

        # Add predictions to the average vector
        #average_predictions += model_preds
        #average_predictionsRot += model_predsRot
        model_count += 1



        # Compute losses
        loss = criterion(model_preds, valDs)
        lossRot = criterion(model_predsRot, valDs)
        # Ensure predictions and true values are on the same device and flattened
        model_preds = model_preds.view(-1).cpu().numpy()  # Convert to numpy for plotting
        model_predsRot = model_predsRot.view(-1).cpu().numpy()  # Convert to numpy for plotting
        valDs_array = valDs_tensor.view(-1).cpu().numpy()  # Convert to numpy for plotting
        # Scatter plot for the current model
        print("Model:", name, "Loss without rotation:", loss.item(), "Loss with rotation", lossRot.item())
        plt.scatter(valDs_array, model_predsRot, alpha=0.7, label=f'{name} Predictions')




    """
    # Calculate the average predictions
    average_predictions /= model_count
    average_predictionsRot /= model_count

    loss = criterion(average_predictions, valDs)
    lossRot = criterion(average_predictionsRot, valDs)

    print("Model:", "Average", "Loss without rotation:", loss.item(), "Loss with rotation", lossRot.item())

    # Convert average predictions to numpy for plotting
    average_predictions_np = average_predictions.view(-1).cpu().numpy()
    average_predictions_npRot = average_predictionsRot.view(-1).cpu().numpy()

    # Scatter plot for the average predictions
    plt.scatter(valDs_array, average_predictions_np, color='orange', alpha=0.9, label='Average Predictions', marker='x')
    plt.scatter(valDs_array, average_predictions_npRot, color='yellow', alpha=0.9, label='Average Predictions', marker='x')
    """


    # Plot the ideal line
    min_val, max_val = min(valDs_array), max(valDs_array)
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal (y = x)')

    # Add plot details
    plt.title("True vs Predicted D Values (All Models)")
    plt.xlabel("True D Values")
    plt.ylabel("Predicted D Values")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if output_path: plt.savefig(output_path)

    # Show the plot
    plt.show()

    return coarseD_array, valDs_array

def plot_trueVpredicted_Dvalues_absolute_error(coarseD_array, valDs_array, val_images, output_name="true_vs_predicted_absolute_error.svg"):

    output_path = os.path.join(OUTPUT_DIR, output_name) if output_name else None

    # Absolute error for coarseD
    absolute_error_coarseD = np.abs(coarseD_array - valDs_array)

    # Number of trajectory sets
    num_trajectories = len(valDs_array)

    # Define bar width and positions
    bar_width = 0.2
    x_indices = np.arange(1, num_trajectories + 1)  # Trajectory indices
    offset = 0  # Offset for each bar group

    # Initialize the figure
    plt.figure(figsize=(14, 8))

    # Plot absolute error for coarseD
    plt.bar(
        x_indices + offset,
        absolute_error_coarseD,
        width=bar_width,
        label="Absolute Error CoarseD",
        alpha=0.7,
        color='orange'
    )
    offset += bar_width  # Update offset for the next group

    # Iterate over all models to plot absolute errors
    for name, params in models_params.items():
        # Predictions from the model
        model = params["loaded_model"]
        model_predsRot, individualPreds = predict_with_rotations(model, val_images, device)
        model_predsRot = model_predsRot.view(-1).cpu().numpy()  # Convert to numpy for plotting

        # Calculate absolute error for the model
        absolute_error_model = np.abs(model_predsRot - valDs_array)

        # Plot the bar for the model
        plt.bar(
            x_indices + offset,
            absolute_error_model,
            width=bar_width,
            label=f"Absolute Error {name}",
            alpha=0.7
        )
        offset += bar_width  # Update offset for the next group

    # Add labels, title, and legend
    plt.xlabel("Trajectory Set Index")
    plt.ylabel("Absolute Error (nm^2/s)")
    plt.title("Absolute Error between Predicted and True Diffusion Coefficients")
    plt.legend()
    plt.grid(True)

    # Adjust x-axis ticks to center bar groups
    plt.xticks(x_indices + (offset - bar_width) / 2 - bar_width, x_indices)

    # Show the plot
    plt.tight_layout()
    if output_path: plt.savefig(output_path)
    plt.show()

def plot_losses(tr_loss_histories, totalEpochs, output_name="losses2.svg"):
    output_path = os.path.join(OUTPUT_DIR, output_name) if output_name else None

    plt.figure(figsize=(10, 6))
    for model_name, losses in tr_loss_histories.items():
        plt.plot(range(1, totalEpochs + 1), losses, label=model_name)

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss per Iteration for Each Model (1 Epoch per Iteration)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if output_path: plt.savefig(output_path)
    plt.show()

def save_models_validations_losses_epochs(allGeneratedDs, val_loss_histories, output_dir=OUTPUT_DIR):
    for name, params in models_params.items():
        model = params["loaded_model"]
        filename = "w_"+name+".pth"
        output_path = os.path.join(output_dir, filename)
        save_model_weights(model, output_path)

    np.save(os.path.join(OUTPUT_DIR,"allDs.npy"),allGeneratedDs)
    for model_name, losses in val_loss_histories.items():
        filename = "l_"+model_name+".npy"
        output_path = os.path.join(output_dir, filename)
        np.save(output_path,losses)

def plots(output_name="image_frames.svg"):

    output_path = os.path.join(OUTPUT_DIR, output_name) if output_name else None
    # Uncomment these 3 lines to generate a new reference image
    #singleIm, singleestimatedDs = generateImagesAndEstimateD(1, nframes, npixel, factor_hr, nposframe, DGen, dt, fwhm_psf, pixelsize,flux, background, poisson_noise, gaussian_noise, normalizeValue)
    #im, estD = singleIm[0,:], singleestimatedDs[0]
    #save_image(im,"refImageBig.npy") 


    # We created 3 reference images, with different D values, to observe what our algorithm predicts
    dict = {"refImageSmall": 1.935, "refImage": 13.875, "refImageBig": 26.092}

    for iname, estD in dict.items():
        im = load_image(iname + ".npy")
        for mname, params in models_params.items():
            model = params["loaded_model"]

            modelEstimation = predict_diffusion_coefficients(model, im ,device)
            print(f"Real D value for image {iname} of model: {mname} : {estD:.4f}. Model estimated value: {modelEstimation:.4f}")

        plot_image_frames(im,f"Image with D={estD}", output_path=output_path)

if __name__ == "__main__":
    main()