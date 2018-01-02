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
 Em seguida, extraia os arquivos de **data.zip**. Verifique se o diretório **code** contém diretamente os diretórios **training** e **test** e os arquivos **classes.csv** e **pascalPartOntology.csv**. Caso eles estejam em um diretório mais interno, traga-os para o diretório **data**.

## Estrutura do Repositório
- **pascalpart_dataset.tar.gz**: Contém as anotações do conjunto de dados PASCAL-Part no estilo PASCAL VOC. Este diretório é necessário se você quiser treinar a *Fast-RCNN* (https://github.com/rbgirshick/fast-rcnn) nesse conjunto de dados para computar o *grounding*/vetor de características de cada *bounding box*.
    - **Annotations**: as anotações no formato **.xml**. Para ver as bounding boxes nas imagens, use o pascalvok devkit  http://host.robots.ox.ac.uk/pascal/VOC/index.html.
    - **ImageSets**: A divisão do conjunto de dados em conjunto de treino e teste de acordo com cada predicado/classe unário. Para mais informações, veja o formato pascalvoc no devkit citado acima.
    - **JPEGImages**: Esta pasta está vazia, mas é possível baixar as imagens originais de http://host.robots.ox.ac.uk/pascal/VOC/voc2010/.
- **code**: Contem os dados, a pasta de saída e o código fonte do LTN.
    - **data**: Contém o conjunto de treino, de teste e a ontologia que define os axiomas mereológicos.
    - **models**: Contem a avaliação de baseline e das grounded theories. Caso esta pasta não tenha vindo com o repositório, ela deve ser criada.
    - **results**: Contem as grounded theories treinadas. Caso esta pasta não tenha vindo com o repositório, ela deve ser criada.

## Instalação do Anaconda
Acesse o endereço https://www.anaconda.com/download/ e faça o download da versão compatível com o sistema operacional e a versão do Python de preferência.
Para Linux:
Abra um terminal, acesse o local onde o arquivo foi baixado e entre com o seguinte comando no terminal:
```sh
$ bash ~/Anaconda2-5.0.1-Linux-x86_64.sh
```
Os números *2-5.0.1* mudam de acordo com a versão baixada.
Siga os passos indicados no terminal e, ao terminar a instalação, feche o mesmo e abra outro terminal.

Para Windows:
Clique duas vezes no instalador .exe e siga os passos.

# Os próximos passos serão indicados de acordo com o sistema operacional utilizado na execução do trabalho.

## Instalação do Tensorflow no Linux
1 - Instalado o Anaconda, abra o terminal e digite:
```sh
$ conda create -n tensorflow python=2.7 #ou python=3.4, etc.
```
2 - Ative o ambiente conda:
```sh
$ source activate tensorflow
(tensorflow)$  # Sua linha de comando deve mudar.
```
## Instalação do Tensorflow no ambiente conda:
```sh
(tensorflow)$ pip install --ignore-installed --upgrade tfBinaryURL
```
Onde tfBinaryURL é:
Python 2.7 somente CPU:
```
https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.4.0-cp27-none-linux_x86_64.whl
```
Python 2.7 suporte a GPU:
```
https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.0-cp27-none-linux_x86_64.whl
```
Python 3.4 somente CPU:
```
https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.4.0-cp34-cp34m-linux_x86_64.whl
```
Python 3.4 suporte a GPU:
```
https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.0-cp34-cp34m-linux_x86_64.whl
```

## Considerações da instalação
Caso tenha escolhido a versão de suporte a GPU, certifique-se que a máquina possua o CUDA instalado. Caso contrário, a instalação está concluída.

## Ativação do ambiente virtual
Sempre, antes de executar o programa, verifique se o ambiente virtual está ativo, isto é, se aparece `(tensorflow) $` no terminal, caso não apareça, ative-o com o comando:

```sh
$ source activate tensorflow
```
## Treinamento de uma grounded theory
Para iniciar o treinamento vá para o diretório **code** e execute o treinamento.
```sh
$ cd ./LTN/code
$ python train.py
```
Grounded theories treinadas estão no diretório **models**.

## Avaliação de grounded theories e baselines
```sh
$ python evaluate.py
```
Os resultados estão no diretório **results**.
Resultados mais detalhados estão em **results/report.csv**.
