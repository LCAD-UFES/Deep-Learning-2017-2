# Running YOLO with IARA's data


 - **Discipline**: Deep Learning 2017/2
 - **Students**: Lucas Grigoleto Scart and Ranik Guidolini
 - **Teacher**: Alberto Ferreira de Souza

### Introduction

This tool runs darknet models and show the predictions. 

### Dependencies

OpenCV is used to load the images and display the results on screen. 
The code was tested only with OpenCV 3, but both versions should work. 

Detection with darknet can be performed on the CPU, but you can gain an enormous speed up by using an NVIDIA GPU with cuda and cuDNN. 
Currently cuDNN 5 or 6 and CUDA >=7.5 are supported.

Python is an optional dependence to use the script that creates the list of images required to run this tool.
This script runs in both python 2 or 3.

### Compilation

First you need to download darknet. For your convenience, it's already present as an submodule of this project, you just need to fetch all the files.
To do so, open an terminal and type:

```bash
git submodule init
git submodule update
```

Now the folder `darknet` inside this directory is not empty anymore. The next step is to compile the framework that you have just downloaded.
Navigate to it and edit some parameters in the makefile:

```bash
cd darknet/
gedit Makefile
```
Change the flags to `OPENCV=1`, `OPENMP=1` and `LIBSO=1`. 
If you have an NVIDIA card with cuda installed, also change `GPU=1` and `CUDNN=1` to have an speed up in detection speed.
Close the file and type `make` to compile darknet. Check if an file named `darknet.so` was created.

Go one folder up and type `make` to compile the tool. The file `yolo` should be created there.

### Running the tool

In order to run the tool, you first need to download the yolo weights. 
To do so, run the script:

```bash
sh download_weights.sh
```

While this download is occurring, you can prepare your data. 
Put the pictures you want to detect at the `images` folder, and run the script to generate the image list:

```bash
python generate_image_list.py
```

Each time you change the pictures in the folder you have to run this script before using the tool.
Now, if the weights have finished downloading, you can run the tool by typing:

```bash
./yolo imagelist.txt
```

To change to the next picture, just hit any key on your keyboard. For improved detection rates, open the file `darknet/cfg/yolo.cfg` and change the `height` and `width` attributes to 832 or any multiple of 32. If an out of memory error occurs then increase `subdivisions` to 16, 32 or 64.

### Results

The `images` directory contains some samples extracted from IARA's data. 
Images were recorded with an Zed stereo camera, placed in the windshield of the car.

Results from the complete recording which the samples were extracted are presented in [this](https://youtu.be/sPyi6hcI16w) video.