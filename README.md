# Models

## CO_BC
**CO_BC** stands for the **C**atastrophic **O**utler **B**inary **C**lassification model that takes photometric redshifts and redshift probability distributions of galaxies as input.
The model outputs a value for each galaxy in between 0 and 1, including, that can be interpreted as the probability of a galaxy being a catastrophic outlier.

## PhotZ_LR
**PhotZ_LR** stands for the photometric redshift (**photo-z**) neural network **L**inear **R**egression, used as a baseline photometric redshift prediction method.
This simple implementation of neural network linear regression maps 5-band magnitudes to photometric redshifts for each galaxy.

## PDF_MC
**PDF_MC** stands for the **P**robability **D**ensity **F**unction **M**ulticlass **C**lassifier, used as one of the methods to generate the photometric density distribution. This simple implementation of multiclass classifier algorithm maps 5-band magnitudes to photometric redshift distributions in custom bins for each galaxy.

# Files

## `/data`
All the pre-trained models are stored in this directory.

## `CO_BC.py`
Python implementation of the CO_BC model.

## `CO_BC_train_demo.ipynb`
This Jupyter notebook shows the instructions to train the CO_BC model. 

## `CO_BC_run_demo.ipynb`
This Jupyter notebook shows the instructions to run a pre-trained model of the CO_BC model.

## `PDF_MC.py`
Python implementation of the PDF_MC model.

## `PDF_MC_run_demo.ipynb`
This Jupyter notebook shows the instructions to run a pre-trained model of the PDF_MC model.

## `PhotZ_LR.py`
Python implementation of the PhotZ_LR model.

## `PhotZ_LR_run_demo.ipynb`
This Jupyter notebook shows the instructions to run a pre-trained model of the PhotZ_LR model.

## `model_config.json`
The JSON file contains default configurations for training the models.

## `co_util.py`
The Python file contains various functions facilitating preprocessing, such as rebalancing, and classification.

## `plotting_routine.py`
The Python file contains various functions to plot the dataset properties and results. 