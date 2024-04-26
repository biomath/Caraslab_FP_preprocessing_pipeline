# Fiber photometry analysis pipeline

## Install instructions
1. Download and install [ana/miniconda](https://docs.anaconda.com/free/miniconda/index.html)
2. Download and install [python](https://www.python.org/downloads/) >3.11 
3. Clone this repository in your computer
4. Open terminal window
5. Change terminal directory to location where files were downloaded
For example:
```
cd C:/username/Documents/Caraslab_FP_analysis_pipeline
```
6. Create conda environment
```
conda create -n caraslab_fp
```
5. Activate conda environment
```
conda activate caraslab_fp
```
6. Install requirements
```
conda install --yes --file requirements.txt
```
Pipeline is installed after this. If you want to run right after installation, proceed to step 2 below

## Run instructions
1. Activate conda environment
```
conda activate caraslab_fp
```
2. Run Jupyter on your browser
```
jupyter notebook
```
3. Double-click on the notebook file: Caraslab_FP_analysis_pipeline.ipynb
