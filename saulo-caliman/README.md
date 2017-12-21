# Trabalho da Disciplina Deep Learning 2017-2
Saulo Caliman Gomes
Professor: Alberto Ferreira De Souza

<<<<<<< HEAD
## Índice
=======
## Index
>>>>>>> 39d322f7c10db9f2727ce77066fa45b1b26b1c66

- [1 - Install Anaconda 5.1](#1-install-anaconda)
- [2 - Linux Ubuntu 16](#2-linux-ubuntu-16)
- [3 - Python para Windows](#3-python-para-windows)
- [4 - Instalando o Cuda e Cudnn com suporte a GPU](#4-install-cuda-cudnn-support-gpu)
- [5 - CUDA 8.0 GA2](#5-cuda-8-ga2)
- [6 - PATH](#6-path)
- [7 - Instalando os pacotes](#7-instalando-pacotes)
- [8 - Iniciando o jupyter notebook](#8-iniciando-jupyter)
- [9 - Montando a Rede VGG 16](#9-montando-rede-vgg-16)

<<<<<<< HEAD
# Pré Requisitos
É de fundamental importância que o CUDA, cuDNN, OpenCV e Atlas estejam instalados na máquina. Também deve ser levado em consideração que o OpenPose fica aproximadamente 10% mais rápido com a instalação do cuDNN 5.1 em comparação com o cuDNN 6.

1. [CUDA](https://developer.nvidia.com/cuda-80-ga2-download-archive) Deve ser instalado. Você deve reiniciar sua máquina depois de instalar o CUDA.
2. [cuDNN](https://developer.nvidia.com/cudnn): Depois de baixá-lo, basta descompactá-lo e copiar (mesclar) o conteúdo na pasta CUDA, e.g. `/usr/local/cuda-8.0/`. Nota: O OpenPose trabalha mais rápido ~10% com o cuDNN 5.1 comparado ao cuDNN 6.
3. OpenCV pode ser instalado utilizando `apt-get install libopencv-dev`. Se você compilou por sí mesmo o OpenCV, siga [Manual Compilation](#manual-compilation). Depois que os dois arquivos Makefile.config foram gerados, os edite e descomente a linha `# OPENCV_VERSION := 3`.
4. Atlas pode ser instalado com `sudo apt-get install libatlas-base-dev`. Em vez de Atlas, você pode usar OpenBLAS ou Intel MKL modificando a linha `BLAS := atlas` da mesma maneira que mencionado anteriormente para a seleção da versão do OpenCV.

# Instalação do OpenPose por Script
O script abaixo faz a compilação do Caffe e da biblioteca OpenPose. Também efetua o download dos modelos para para o Ubuntu 14.04 ou 16.04 (detectado automaticamente para o script) e CUDA 8:
```
bash ./ubuntu/install_caffe_and_openpose_if_cuda8.sh
```
**Importante:** Esse script funciona apenas com o CUDA 8 e o Ubuntu 14 ou 16. Veja a instalação manual caso utilizando outra versão do Ubuntu.

# Instalação do OpenPose manualmente
1. Instalar os pré requisitos.

2. Compilar o Caffe e o OpenPose utilizando as linhas abaixo:

```cd 3rdparty/caffe/
```

## Selecionar o Makefile (execute apenas um dos próximos 4 comandos)
```cp Makefile.config.Ubuntu14_cuda7.example Makefile.config # Ubuntu 14, cuda 7
```
```cp Makefile.config.Ubuntu14_cuda8.example Makefile.config # Ubuntu 14, cuda 8
```
```cp Makefile.config.Ubuntu16_cuda7.example Makefile.config # Ubuntu 16, cuda 7
```
```cp Makefile.config.Ubuntu16_cuda8.example Makefile.config # Ubuntu 16, cuda 8
```

## Compilar o Caffe
```
make all -j${números_de_cpu} && make distribute -j${números_de_cpu}
```

### Instalar o OpenPose ###
```
cd ../../models/
bash ./getModels.sh # Apenas faz download dos modelos treinados. Os modelos ficam na pasta /models
cd ..
````
## O arquivo deve ser a mesma versão do CUDA e sistema operacional
````
cp ubuntu/Makefile.config.Ubuntu14_cuda7.example Makefile.config
````

## Modificar qualquer flag personalizado de Makefile.config resultante (e.g. OpenCV 3, Atlas/OpenBLAS/MKL, etc.)
```
make all -j${números_de_cpu}
```

## Instalação e execução do código
Copie e cole a pasta OpenPose desse repositório para a pasta OpenPose instalada no computador

Utilize o comando abaixo na pasta raiz do openpose para compilar o código.
```
make all
```
O wrapper do OpenPose permite com que todos os scripts que estão no diretório openpose/examples/user_code seja compilado sem precisar ser chamado no header do programa.

Execute o comando:
```
./runRectanglesTest.sh
```

O script ```runRectanglesTest.sh``` executa o binário da compilação e usa a flag ```-models ``` para selecionar o diretório de treinamento do OpenPose.

Vale ressaltar que o atual experimento utiliza a entrada de vídeo primária ao ser executado sem nenhuma flag. Para executar com a saída de vídeo modifique o arquivo ```runRectanglesTest.sh``` e adicione a flag ```-video {caminho/do/vídeo}```

# Experimento
Criar bounding box ao redor de cada pessoa identificada.

No wrapper do OpenPose é possível utilizar a estrutura ```Datum``` que possui uma propriedade que é o vetor da posição de cada pessoa identificada em cada frame da exibição. Sabendo o máximo e mínimo de X e de Y foi possível, utilizando o OpenCV, criar um bounding box em cada pessoa identificada.

# Resultado
=======
#Pré Requisitos
É de fundamental importância que o CUDA, cuDNN, OpenCV e Atlas estejam instalados na máquina. Também deve ser levado em consideração que o OpenPose fica aproximadamente 10% mais rápido com a instalação do cuDNN 5.1 em comparação com o cuDNN 6.
'''
1. [CUDA](https://developer.nvidia.com/cuda-80-ga2-download-archive) must be installed. You should reboot your machine after installing CUDA.
2. [cuDNN](https://developer.nvidia.com/cudnn): Once you have downloaded it, just unzip it and copy (merge) the contents on the CUDA folder, e.g. `/usr/local/cuda-8.0/`. Note: We found OpenPose working ~10% faster with cuDNN 5.1 compared to cuDNN 6. Otherwise, check [Compiling without cuDNN](#compiling-without-cudnn).
3. OpenCV can be installed with `apt-get install libopencv-dev`. If you have compiled OpenCV 3 by your own, follow [Manual Compilation](#manual-compilation). After both Makefile.config files have been generated, edit them and uncomment the line `# OPENCV_VERSION := 3`. You might alternatively modify all `Makefile.config.UbuntuXX` files and then run the scripts in step 2.
4. In addition, OpenCV 3 does not incorporate the `opencv_contrib` module by default. Assuming you have OpenCV 3 compiled with the contrib module and you want to use it, append `opencv_contrib` at the end of the line `LIBRARIES += opencv_core opencv_highgui opencv_imgproc` in the `Makefile` file.
5. Atlas can be installed with `sudo apt-get install libatlas-base-dev`. Instead of Atlas, you can use OpenBLAS or Intel MKL by modifying the line `BLAS := atlas` in the same way as previosuly mentioned for the OpenCV version selection.
'''

#Instalação do OpenPose por Script
O script abaixo faz a compilação do Caffe e da biblioteca Openpose. Também efetua o download dos modelos para para o Ubuntu 14.04 ou 16.04 (detectado automaticamente para o script) e CUDA 8:
'''
bash ./ubuntu/install_caffe_and_openpose_if_cuda8.sh
'''
**Importante:** Esse script funciona apenas com o CUDA 8 e o Ubuntu 14 ou 16. Veja a instalação manual caso utilizando outra versão do Ubuntu.

#Instalação do OpenPose manualmente
1. Instalar os pré requisitos.

2. Compilar o Caffe e o OpenPose utilizando as linhas abaixo:
'''
cd 3rdparty/caffe/
# Select your desired Makefile file (run only one of the next 4 commands)
cp Makefile.config.Ubuntu14_cuda7.example Makefile.config # Ubuntu 14, cuda 7
cp Makefile.config.Ubuntu14_cuda8.example Makefile.config # Ubuntu 14, cuda 8
cp Makefile.config.Ubuntu16_cuda7.example Makefile.config # Ubuntu 16, cuda 7
cp Makefile.config.Ubuntu16_cuda8.example Makefile.config # Ubuntu 16, cuda 8
# Change any custom flag from the resulting Makefile.config (e.g. OpenCV 3, Atlas/OpenBLAS/MKL, etc.)
# Compile Caffe
make all -j${number_of_cpus} && make distribute -j${number_of_cpus}

### Install OpenPose ###
cd ../../models/
bash ./getModels.sh # It just downloads the Caffe trained models
cd ..
# Same file cp command as the one used for Caffe
cp ubuntu/Makefile.config.Ubuntu14_cuda7.example Makefile.config
# Change any custom flag from the resulting Makefile.config (e.g. OpenCV 3, Atlas/OpenBLAS/MKL, etc.)
make all -j${number_of_cpus}
'''

#Instalação e execução
Copie e cole a pasta OpenPose desse repositório para a pasta OpenPose instalada no computador

Utilize o comando make all na pasta raiz do openpose para compilar

Execute o comando
./runRectanglesTest.sh

#Resultado
>>>>>>> 39d322f7c10db9f2727ce77066fa45b1b26b1c66
Aplicando o OpenPose no log do LCAD, foi possível reconhecer 12 de 13 pedestres que caminharam na faixa de pedestres.
