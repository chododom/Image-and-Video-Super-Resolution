# Image and Video Super-Resolution Using Neural Networks
Author: Dominik Chodounský

The aim of this project was to use deep learning techniques to upscale the resolution of images and subsequently videos via synthetic generation.

More specifically, the project explores the usage of the U-Net and GAN architectures to achieve this goal.

All details regarding the data, used methods and evaluation of results is included in the [report](./report.pdf).

## Data

In order to be able to train and test the models, you will need to import the contents of a zip file from the septuplet test version of Vimeo 90k dataset (http://data.csail.mit.edu/tofu/testset/vimeo_super_resolution_test.zip or 'The test set for video super-resolution: zip (6GB)' at https://github.com/anchen1011/toflow) in the folder /data/vimeo/.
The same folder already includes the text files which specify which video sequences are used for trianing and which for testing. The code for setting up the data is already in the included [notebook](./src/Image_super_resolution.ipynb), so all you need to do is download and unzip the Vimeo images.

## Code

All experiments were run in the provided [Jupyter Notebook](./src/Image_super_resolution.ipynb) and all the source codes that it uses are located in folder [src](./src).

All of the codes run with Python 3.8, the list of packages used in my development environment is shown in the [requirements file](requirements.txt).


