# Face Generation

In this project, I built a generative adversarial networks(GAN) to generate new images of faces. The project is part of fulfillment for a nanodegree at Udacity.

# Installation

For best experience with managing dependency I advise you install [Anconda](https://docs.anaconda.com/anaconda/install/) or [miniconda](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/download.html).

Create a virtual environment with conda
```
conda create --name deep-learning python=3
```
Activate environment.
```
source activate deep-learning
```

Install dependencies.

```
pip install -r requirements.txt
```

Download or clone this face-generation. Launch the app with jupyter-notebook.
```
jupyter-notebook dlnd_face_generation.ipynb
```

# Usage

### Setup
Download the [celeba dataset](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/November/5be7eb6f_processed-celeba-small/processed-celeba-small.zip). Unzip the folder and place it in this project's home directory, at the location `processed_celeba_small/`.

This is a zip file that you'll need to extract in the home directory of this notebook for further loading and processing. After extracting the data, you should be left with a directory of data  `processed_celeba_small/`.

Note: If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder.

### Run
Run all code cells in the notebook (This will take a very very long time to run on a CPU, preferably you should run on a GPU).

Notice that the generated faced though low-resolution are human faces.




