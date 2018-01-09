# DEEP LEARNING - 2017 2, Maison M.

GAN - Generative Adversarial Network
Maison Melotti

MAIN RESOURCES


# GAN (ORIGINAL) PAPER -->

* The original [paper](https://arxiv.org/pdf/1406.2661.pdf) written by Ian Goodfellow in 2014. 

# DualGAN Paper -->

 # ICCV paper of DualGAN
<a href="https://arxiv.org/abs/1704.02510">DualGAN: unsupervised dual learning for image-to-image translation</a>




Codes are built on the top of pix2pix-tensorflow and DCGAN-tensorflow. Thanks for their precedent contributions!



# architecture of DualGAN

![architecture](https://github.com/duxingren14/DualGAN/blob/master/0.png)

# How to setup

## Prerequisites

* Python (2.7 or later)

* numpy

* scipy

* NVIDIA GPU + CUDA 8.0 + CuDNN v5.1

* TensorFlow 1.0 or later

* unzip


# Getting Started
## steps

* clone this repo:

```
git clone https://github.com/duxingren14/DualGAN.git

cd DualGAN
```

* download datasets (e.g., sketch-photo), run:

```
bash ./datasets/download_dataset.sh sketch-photo
```

* train the model:

```
python main.py --phase train --dataset_name sketch-photo --image_size 256 --epoch 45 --lambda_A 20.0 --lambda_B 20.0 --A_channels 1 --B_channels 1
```

* test the model:

```
python main.py --phase test --dataset_name sketch-photo --image_size 256 --epoch 45 --lambda_A 20.0 --lambda_B 20.0 --A_channels 1 --B_channels 1
```

## optional

Similarly, run experiments on facades dataset with the following commands:

```
bash ./datasets/download_dataset.sh facades

python main.py --phase train --dataset_name facades --image_size 256 --epoch 45 --lambda_A 20.0 --lambda_B 20.0 --A_channels 3 --B_channels 3

python main.py --phase test --dataset_name facades --image_size 256 --epoch 45 --lambda_A 20.0 --lambda_B 20.0 --A_channels 3 --B_channels 3
```

# datasets

some datasets can also be downloaded manually from the website. Please cite their papers if you use the data. 
Using maps dataset requires add maps on "datasets.sh" file! Not all datasets are default!

facades: http://cmp.felk.cvut.cz/~tylecr1/facade/

sketch: http://mmlab.ie.cuhk.edu.hk/archive/cufsf/

maps: https://mega.nz/#!r8xwCBCD!lNBrY_2QO6pyUJziGj7ikPheUL_yXA8xGXFlM3GPL3c

oil-chinese:  http://www.cs.mun.ca/~yz7241/, jump to http://www.cs.mun.ca/~yz7241/dataset/

day-night: http://www.cs.mun.ca/~yz7241/dataset/


# Experimental results:

![day2night](https://github.com/duxingren14/DualGAN/blob/master/6.PNG)
![da2ni](https://github.com/duxingren14/DualGAN/blob/master/da2ni.png)
![la2ph](https://github.com/duxingren14/DualGAN/blob/master/la2ph.png)
![ph2la](https://github.com/duxingren14/DualGAN/blob/master/ph2la.png)
![sk2ph](https://github.com/duxingren14/DualGAN/blob/master/sk2ph.png)
![ph2sk](https://github.com/duxingren14/DualGAN/blob/master/ph2sk.png)


# Experimental results LCAD-UFES--> IARA data:

Here the



# More GAN Resources

* The original [paper](https://arxiv.org/pdf/1406.2661.pdf) written by Ian Goodfellow in 2014. 
* Siraj Raval's [video tutorial](https://www.youtube.com/watch?v=deyOX6Mt_As) on GANs (Really fun video)
* Ian Godfellow's [keynote](https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Generative-Adversarial-Networks) on GANs (More of a technical video)
* Brandon Amos's image completion [blog post](https://bamos.github.io/2016/08/09/deep-completion/)
* [Blog post](https://medium.com/@ageitgey/abusing-generative-adversarial-networks-to-make-8-bit-pixel-art-e45d9b96cee7) on using GANs in video games. 
* Andrej Karpathy's [blog post](http://cs.stanford.edu/people/karpathy/gan/) with GAN visualizations.
