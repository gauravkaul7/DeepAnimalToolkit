DeepAnimalToolkit: A toolkit for studying animal behavior
==========================================================================================

## High Preformance Computing Setup Instructions

If you would like to use the Toolkit on a computing cluster with slurm, 
We recommend using Anaconda or Miniconda, you can get it from [Conda download site](https://conda.io/docs/user-guide/install/download.html).

Then, Clone the repository, create a conda environment and install all dependencies:

```bash

git clone https://github.com/gauravkaul7/DAT
conda create -n DAT python=3.9
conda activate DAT
python -m pip install -r requirements.txt

```
