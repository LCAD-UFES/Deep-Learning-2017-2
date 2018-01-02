# Redes de Tensores Lógicos para Interpretação Semântica de Imagens de Veículos
 - **Autores**: Lucas Martinuzzo Batista e Pedro Reisen Zanotti
 - **Instituição de Ensino**: Universidade Federal do Espírito Santo (UFES)
 - **Disciplina**: Deep Learning 2017/2
 - **Professor**: Alberto Ferreira de Souza
 
 O material deste repositório é uma adaptação da implementação do artigo *Logic Tensor Networks for Semantic Image Interpretation* (Ivan Donadello, Luciano Serafini e Arthur d'Avila Garcez, 2017).
 
 Siga as instruções abaixo para baixar e rodar o programa.

## Download do Repositório
 Faça o download do repositório completo através do seguinte comando:
```sh
$ git clone https://Martinuzzo@bitbucket.org/Martinuzzo/deeplearningltn.git
```
 Em seguida, extraia os arquivos de ***data.zip***. Verifique se o diretório ***code*** contém diretamente os diretórios ***training*** e ***test*** e os arquivos ***classes.csv*** e ***pascalPartOntology.csv***. Caso eles estejam em um diretório mais interno, traga-os para o diretório ***data***.

## Estrutura do Repositório
- **pascalpart_dataset.tar.gz**: Contém as anotações do conjunto de dados PASCAL-Part no estilo PASCAL VOC. Este diretório é necessário se você quiser treinar a *Fast-RCNN* (https://github.com/rbgirshick/fast-rcnn) nesse conjunto de dados para computar o *grounding*/vetor de características de cada *bounding box*;
    - **Annotations**: Contém as anotações no formato *.xml*. Para ver as *bounding boxes* nas imagens, use o PASCAL VOC Devkit (http://host.robots.ox.ac.uk/pascal/VOC/index.html);
    - **ImageSets**: Contém a divisão do conjunto de dados em conjunto de treinamento e conjunto de teste de acordo com cada predicado ou classe unária. Para mais informações, veja PASCAL VOC Devkit, mencionado acima;
    - **JPEGImages**: Este diretório está vazio, mas é possível baixar as imagens originais de http://host.robots.ox.ac.uk/pascal/VOC/voc2010/.
- **code**: Contém os dados, o diretório de saída e o código fonte da LTN.
    - **data**: Contém o conjunto de treinamento, o conjunto de teste e a ontologia que define os axiomas mereológicos;
    - **models**: Contém a avaliação da *baseline* e das *grounded theories*. Caso esse diretório não tenha vindo com o repositório, ele deve ser criada.
    - **results**: Contém as *grounded theories* treinadas. Caso esse diretório não tenha vindo com o repositório, ele deve ser criada.

## Instalação do Anaconda
 Acesse o endereço https://www.anaconda.com/download/ e faça o *download* da versão compatível com o seu sistema operacional e com a versão do Python de sua preferência.
 
 **Linux**
 
 Abra o terminal, acesse o local em que o arquivo foi baixado e utilize o seguinte comando:
```sh
$ bash ~/Anaconda2-5.0.1-Linux-x86_64.sh
```
 Os números *2-5.0.1* variam de acordo com a versão baixada. Siga os passos indicados no terminal e, ao finalizar a instalação, feche o mesmo e abra outro terminal.

 **Windows**
 
 Clique duas vezes no instalador *.exe* e siga os passos.

## Instalação do Tensorflow (Linux)
 Após a instação do Anaconda, abra o terminal e digite:
```sh
$ conda create -n tensorflow python=2.7 #Ou python=3.4, etc.
```
 Em seguida, ative o ambiente Conda:
```sh
$ source activate tensorflow
(tensorflow)$  # Sua linha de comando deve mudar.
```
## Instalação do Tensorflow (Ambiente Conda):
 Utilize o seguinte comando:
```sh
(tensorflow)$ pip install --ignore-installed --upgrade tfBinaryURL
```
 O parâmetro *tfBinaryURL* é:
 
 Python 2.7 (somente CPU):
```
https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.4.0-cp27-none-linux_x86_64.whl
```
 Python 2.7 (suporte à GPU):
```
https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.0-cp27-none-linux_x86_64.whl
```
 Python 3.4 (somente CPU):
```
https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.4.0-cp34-cp34m-linux_x86_64.whl
```
 Python 3.4 (suporte à GPU):
```
https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.0-cp34-cp34m-linux_x86_64.whl
```
Caso tenha escolhido alguma versão de suporte à GPU, certifique-se que a máquina possua a CUDA instalada. Caso contrário, a instalação está concluída.

## Ativação do ambiente virtual
 Antes de executar o programa, sempre verifique se o ambiente virtual está ativo, isto é, se aparece `(tensorflow) $` no terminal. Caso não apareça, ative-o através do seguinte comando:

```sh
$ source activate tensorflow
```
## Treinamento
 Para iniciar o treinamento das *grounded theories*, vá para o diretório ***code*** e execute o treinamento através de:
```sh
$ cd ./LTN/code
$ python train.py
```
 As *grounded theories* treinadas podem ser encontradas no diretório ***models***.

## Teste
 A avaliação de *grounded theories* e *baselines* pode ser executada da seguinte forma:
```sh
$ python evaluate.py
```
 Os resultados podem ser encontrados no diretório ***results***. Resultados mais detalhados podem ser encontrados em ***results/report.csv***.
