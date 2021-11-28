# flame_ann

Code for Flame ann experiments.
## Startup guide
### Preprare & activate environments
First clone this repository, and open the flame_ann folder.
Then, run the following codes in command line. Note: If conda is not installed yet, please install [miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) first.
```shell
# Create flame-ann-env
conda env create -f environment.yml

# Acvitave environment
conda activate flame-ann-env
```

### Run jupyter-lab to view procedures
Run the following code to open jupyterlab. A browser tab will be open by default. Then you can explore the notebooks in the browser. 
```shell
jupyter-lab
```

## Code explanations

### Notebooks for data1          
- flame_ann1_CO_CO2.ipynb   For network 1, i.e. CO & CO2 rate, also contains the data preprocessing
- flame_ann2_W.ipynb        For network 2, i.e. masses
- flame_ann3_T.ipynb        For network 3, i.e. temperature

The cmp file is for parameter comparison.

### Notebooks for data2
- flame2_preprocess_data.ipynb  For data preprocessing
- flame2_ann1_CO_CO2.ipynb      For network 1, i.e. CO & CO2 rate  
- flame2_ann2_W.ipynb           For network 2, i.e. masses
- flame2_ann3_T.ipynb           For network 3, i.e. temperature

The cmp file is for parameter comparison.

### Fine tune models
- tune_model  Parallel network architecture fine-tuning

### Run jobs for long epochs
- running_jobs/*.py               Codes to run jobs for long epochs, support checkpoint saving and logging
- running_jobs/run_training.sh    Shell scripts to initiate jobs.
