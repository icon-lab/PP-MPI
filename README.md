# PP-MPI
Official repository for Plug-n-Play MPI Reconstruction (PP-MPI)

B. Askin, A. Güngör, D. A. Soydan, E. U. Saritas, C. B. Top and T. Çukur, "PP-MPI: A Deep Plug-and-Play Prior for Magnetic Particle Imaging Reconstruction," in MLMIR, 2022, pp. 105–114.

# Demo
You can use the following links to download training, validation, test datasets. 

# Dataset

Download the folders datasets and OpenMPI in the same folder as the code:

https://drive.google.com/drive/folders/1_rJxCCIepB8x9jI7vUXJCcGJJ-xPfpuy?usp=sharing


# Pretrained Networks:
Pre-trained network can be found under training/denoiser/ folder.

# Training

Generic training code code:

```python train_ppmpi.py --useGPUno 0```


```useGPUno: Selected GPU
wd: weight decay, default is 0
lr: learning rate
saveModelEpoch: save model every X epoch
valEpoch: Compute validation per X epoch
fixedNsStdFlag: 0: randomly generate noise std for each image, 1: fix noise std.
minNoiseStd: For non-fixed noise, minimum noise std.
maxNoiseStdList: For non-fixed noise: maximum noise std., For fixed noise: noise std. Multiple inputs are separated by comma, each input trains a different network consecutively.
batch_size_train: batch size
epoch_nb: number of epochs
wandbFlag: use Wandb for loss tracking (0 / 1)
wandbName: Experiment name for wandb
reScaleBetween: "x,y" rescale images in the dataset between x and y
dims: 2 / 3, number of dimensions of the image ("3" for using a 3D dataset, default 2)

nb_of_featuresList: Number of features of RDN, separate with comma for training of multiple different networks
nb_of_blocks: Number of blocks of RDN
layer_in_each_block: Layer in each block of RDN
growth_rate: growth rate of RDN
```

# Inference for Open MPI dataset

```python openMPItest.py```

Settings should be changed from within the file. A jupyter notebook might be more helpful since it also helps better visualize and manipulate reconstructed images.

**************************************************************************************************************************************
# Citation
You are encouraged to modify/distribute this code. However, please acknowledge this code and cite the paper appropriately.
```
@InProceedings{ppmpi,
author="Askin, Baris
and G{\"u}ng{\"o}r, Alper
and Alptekin Soydan, Damla
and Saritas, Emine Ulku
and Top, Can Bar{\i}{\c{s}}
and Cukur, Tolga",
editor="Haq, Nandinee
and Johnson, Patricia
and Maier, Andreas
and Qin, Chen
and W{\"u}rfl, Tobias
and Yoo, Jaejun",
title="PP-MPI: A Deep Plug-and-Play Prior for Magnetic Particle Imaging Reconstruction",
booktitle="Machine Learning for Medical Image Reconstruction",
year="2022",
publisher="Springer International Publishing",
address="Cham",
pages="105--114",
isbn="978-3-031-17247-2"
}

@misc{deqmpi,
      title={DEQ-MPI: A Deep Equilibrium Reconstruction with Learned Consistency for Magnetic Particle Imaging}, 
      author={Alper Güngör and Baris Askin and Damla Alptekin Soydan and Can Barış Top and Emine Ulku Saritas and Tolga Çukur},
      year={2022},
      eprint={2212.13233},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}

```
(c) ICON Lab 2023

# Prerequisites

- Python 3.6
- CuDNN 8.2.1
- PyTorch 1.10.0

# Acknowledgements

For questions/comments please send an email to: alperg@ee.bilkent.edu.tr
