# Github Repository of group Big Burger in ML4Science project

This is the GitHub repository for group Big Burger for Project 2 in CS-433.  
The project was thanks to ML4Science initiative in the EPFL Center for Imaging, and supervised by Daniel Sage and Thanh-An Pham. 

## Team members: 
  Mathis Solal Krause - mathis.krause@epfl.ch  
  Anoush Azar-Pey - anoush.azar-pey@epfl.ch  
  Emilien Silly - emilien.silly@epfl.ch  

## ML for science project description (by Daniel Sage):
Single-Molecule Localization Microscopy (Nobel Prize 2014) is a powerful super-resolution technique for imaging live cell compartments with a resolution of up to 15 nm. In these images, a single moving molecule appears as a bright "blob," primarily due to motion blur. By analyzing the shape of this blob, we can gain insights into cellular dynamics, specifically by estimating the diffusion coefficient (D) of the molecule.
The goal of this project is to train a deep learning model to predict the diffusion coefficient (D) from a sequence of simulated images. These simulations can faithfully mimic real conditions, thanks to a well-known physical model that incorporates the Brownian motion of the molecule, the microscopy point-spread function, and the noise, allowing us to generate large datasets. We will investigate different neural network architectures to determine which one is most effective for estimating the diffusion coefficient from the sequence of images. Ultimately, we aim to replicate microscopy experimental conditions to apply this model to real images.

This project was done thanks to the EPFL Center for Imaging, which gave us reference papers and prior research done on this subject. All of those can be found in folder references/

## Project structure:

The project was divided in 3 parts: Simulations, Model Training and Real Data Prediction. The first focuses on generating images that will later be used during training. The second comprises the whole training pipeline, from creating/loading models to training and predicting on the validation data.

- `blur_optimization/` contains the code used to optimize the similarity between real data and simulated data by changing the blur of the generated images
- `Models/` contains the models architecture
- `modelsData/` contains the data saved during training, such as weights and losses
- `real-data/` contains the real data on which we want to predict the diffusion coefficient
- `references/` contains the related papers and research done on this subject
- `run_outputs/` contains the results of the real data prediction on the real data done by run.py
- `tests/` contains various tests used to explore the data and the models
  - `centering_images/` contains code that center the images around the particle
  - `CoarseD/` TODO
  - `local_optimum_fix` contains code that tries to fix the local optimum problem with a different weight initialization technique
- `ModelTrainPipeline.ipynb` is the notebook used to train the models using the simulated data
- `RealDataPrediction.ipynb` is the notebook used to predict the diffusion coefficient on real data
- `helpers.py` is an improved version of code given to us that generates tajectories and images
- `run.py` is a script that can be used to predict the diffusion coefficient on real data with easy to change parameters
- `run_full.py` is an augmented version of run.py that also can retrain the models on the simulated data and plot the results, currently not fully working

## Simulation
### Parameters:
For the simulation, multiple hyperparameters were selected to best match real data distribution. These were carefully selected using notebooks tests/image_creation_test.ipynb and test/brownian-msd.ipynb.
Each of these parameters influences the final image, here are the most important ones with their physical explanation:  
nframes = 16    # Number of frames generated for each particle  
nposframe = 10    # Number of sub-positions per frame  
dt = 0.01        # Integration time frame in seconds (time between two frames)  
DGen = 15000        # Diffusion coefficient in (nm)^2 per s (=0.000001 (um^2)/s)  
num_steps = nframes*nposframe  # total number of positions simulated 
npixel = 64 # number of image pixels  
pixelsize = 100 # in nm  
fwhm_psf = 200 # full width half maximum (emulates microscope)  
factor_hr = 5 # image high resulution factor  
flux = 100 # number of photons per s  
poisson_noise = 100   
gaussian_noise = 10  
background = 100 # base background value  
normalizeValue = 1000 # value by which all samples will be normalized; Fixed for all images 


These parameters are imported in every notebook, and are used when creating images. The notebook tests/image_creation_test.ipynb has an explanation of the algorithm used to generate images.   
The code used for image generation and plots is in helpers.py and reused in other parts. 

## Training 

The training part is done in notebook ModelTrainPipeline. The notebook first loads the existing weights if present, than can be run for a wanted number of epochs for a certain number of images. Everything is parametrized. During training, the models are queried on a validation set of 250 images, to assess performance and observe the validation loss.  
All models used are in folder Models/ and the data saved/loaded in the notebook such as weights or losses_histories are stored in ModelsData/

## Real Data Prediction

Finally, the models can be used to predict the Diffusion Coefficient of Real data, not seen during Training. This is done in notebook RealDataPrediction. This can be used to perform real data prediction on new samples stored in folder real-data/, but requires knowing the data distribution and the Normalization factor needed. 

A folder blur_optimization in which we try to optimize the similarity between real data and simulated data is also present.


## Reproducibility

To perform real data prediction on high number of files simultaneously and not need to use a jupyter notebook, we also a have a run.py and run_full.py which can be run from the terminal. They will use the specified data folder and filenames, and put the results in run_outputs. More details can be found by doing run.py -help.  

In order to make this project reproducible, we included a `requirements.txt` file regrouping all python libraries we used, along with their versions. To install them, it is recommended to do so in a virtual environment; to create one, `python3 -m venv venv_name` is a useful command. After activating it, type `pip install -r requirements.txt` to install the libraries.  
If this doesn't work, we also included a `Dockerfile`, which can be used to create a Docker Image to containerize the application. It is longer to set up but should work on all computers.
