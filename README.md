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
 

## Project structure:

The project was divided in 3 parts: Simulations, Model Training and Real Data Prediction. The first focuses on generating images that will later be used during training. The second comprises the whole training pipeline, from creating/loading models to training and predicting on the validation data.


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
