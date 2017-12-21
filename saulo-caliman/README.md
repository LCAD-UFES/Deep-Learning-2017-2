# Trabalho da Disciplina Deep Learning 2017-2
Saulo Caliman Gomes
Professor: Alberto Ferreira De Souza

## Índice

- [Pré Requisitos](#pre-requisitos)
- [Instalação do OpenPose por Script](#instalacao-do-openpose-por-script)
- [Instalação do OpenPose manualmente](#instalacao-do-openpose-manualmente)
  - [Selecionar o Makefile (execute apenas um dos próximos 4 comandos)](#selecionar-o-makefile-execute-apenas-um-dos-proximos-4-comandos)
  - [Compilar o Caffe](#compilar-o-caffe)
          - [Instalar o OpenPose](#Instalar-o-openpose)
  - [O arquivo deve ser a mesma versão do CUDA e sistema operacional](#o-arquivo-deve-ser-a-mesma-versao-do-cuda-e-sistema-operacional)
  - [Modificar qualquer flag personalizado de Makefile.config resultante (e.g. OpenCV 3, Atlas/OpenBLAS/MKL, etc.)](#modificar-qualquer-flag-personalizado-de-makefileconfig-resultante-eg-opencv-3-atlasopenblasmkl-etc)
  - [Instalação e execução do código](#instalacao-e-execucao-do-codigo)
- [Experimento](#experimento)
- [Resultado](#resultado)

# Pré Requisitos
É de fundamental importância que o CUDA, cuDNN, OpenCV e Atlas estejam instalados na máquina. Também deve ser levado em consideração que o OpenPose fica aproximadamente 10% mais rápido com a instalação do cuDNN 5.1 em comparação com o cuDNN 6.

1. [CUDA](https://developer.nvidia.com/cuda-80-ga2-download-archive) Deve ser instalado. Você deve reiniciar sua máquina depois de instalar o CUDA.
2. [cuDNN](https://developer.nvidia.com/cudnn): Depois de baixá-lo, basta descompactá-lo e copiar (mesclar) o conteúdo na pasta CUDA, e.g. `/usr/local/cuda-8.0/`. Nota: O OpenPose trabalha mais rápido ~10% com o cuDNN 5.1 comparado ao cuDNN 6.
3. OpenCV pode ser instalado utilizando `apt-get install libopencv-dev`. Se você compilou por sí mesmo o OpenCV, siga [Manual Compilation](#manual-compilation). Depois que os dois arquivos Makefile.config foram gerados, os edite e descomente a linha `# OPENCV_VERSION := 3`.
4. Atlas pode ser instalado com `sudo apt-get install libatlas-base-dev`. Em vez de Atlas, você pode usar OpenBLAS ou Intel MKL modificando a linha `BLAS := atlas` da mesma maneira que mencionado anteriormente para a seleção da versão do OpenCV.

# Instalação do OpenPose por Script
O script abaixo faz a compilação do Caffe e da biblioteca OpenPose. Também efetua o download dos modelos para para o Ubuntu 14.04 ou 16.04 (detectado automaticamente para o script) e CUDA 8:
`bash ./ubuntu/install_caffe_and_openpose_if_cuda8.sh`

**Importante:** Esse script funciona apenas com o CUDA 8 e o Ubuntu 14 ou 16. Veja a instalação manual caso utilizando outra versão do Ubuntu.

# Instalação do OpenPose manualmente
1. Instalar os pré requisitos.

2. Compilar o Caffe e o OpenPose utilizando as linhas abaixo:

cd 3rdparty/caffe/

## Selecionar o Makefile (execute apenas um dos próximos 4 comandos)
```
cp Makefile.config.Ubuntu14_cuda7.example Makefile.config # Ubuntu 14, cuda 7
cp Makefile.config.Ubuntu14_cuda8.example Makefile.config # Ubuntu 14, cuda 8
cp Makefile.config.Ubuntu16_cuda7.example Makefile.config # Ubuntu 16, cuda 7
cp Makefile.config.Ubuntu16_cuda8.example Makefile.config # Ubuntu 16, cuda 8
```

## Compilar o Caffe
`make all -j${números_de_cpu} && make distribute -j${números_de_cpu}`

### Instalar o OpenPose
```
cd ../../models/
bash ./getModels.sh # Apenas faz download dos modelos treinados. Os modelos ficam na pasta /models
cd ..
````
## O arquivo deve ser a mesma versão do CUDA e sistema operacional
`cp ubuntu/Makefile.config.Ubuntu14_cuda7.example Makefile.config`

## Modificar qualquer flag personalizado de Makefile.config resultante (e.g. OpenCV 3, Atlas/OpenBLAS/MKL, etc.)
`make all -j${números_de_cpu}`

## Instalação e execução do código
Copie e cole a pasta OpenPose desse repositório para a pasta OpenPose instalada no computador

Utilize o comando abaixo na pasta raiz do openpose para compilar o código.
`make all`

O wrapper do OpenPose permite com que todos os scripts que estão no diretório openpose/examples/user_code seja compilado sem precisar ser chamado no header do programa.

Execute o comando:
`./runRectanglesTest.sh`

O script `runRectanglesTest.sh` executa o binário da compilação e usa a flag `-models` para selecionar o diretório de treinamento do OpenPose.

Vale ressaltar que o atual experimento utiliza a entrada de vídeo primária ao ser executado sem nenhuma flag. Para executar com a saída de vídeo modifique o arquivo `runRectanglesTest.sh` e adicione a flag `-video {caminho/do/vídeo}`

# Experimento
Criar bounding box ao redor de cada pessoa identificada.

No wrapper do OpenPose é possível utilizar a estrutura `Datum` que possui uma propriedade que é o vetor da posição de cada pessoa identificada em cada frame da exibição. Sabendo o máximo e mínimo de X e de Y foi possível, utilizando o OpenCV, criar um bounding box em cada pessoa identificada.

# Resultado
Aplicando o OpenPose no log do LCAD, foi possível reconhecer 12 de 13 pedestres que caminharam na faixa de pedestres.

[![Experimento](https://raw.githubusercontent.com/LCAD-UFES/Deep-Learning-2017-2/master/saulo-caliman/doc/output.webm)]
