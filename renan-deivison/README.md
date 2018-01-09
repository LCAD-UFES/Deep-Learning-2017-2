# Criar CNN - Convolutional Neural Network para identificação de carros usando Keras e TensorFlow
 - Passo a passo de como criar uma CNN (Convolutional Neural Network) para identificação de carro usando Keras e Tensor Flow baseado nas redes AlexNet e ZFNet usando Windows 10. Este arquivo contempla a instalação do Anaconda, treino e predição de imagens após treino ( baseado na ALEXNET e ZFNET)
 
# Autores: Renan Mantuanelli de Aquino e Deivison Augusto da Vitória

# Trabalho disciplina DeepLearning UFES 2017-2

## Index

1.    Instalar anaconda phyton
- https://www.continuum.io/downloads)

2.    Instalar spyder
- Abrir anaconda navegator e instalar spyder

3.    Instalar theano
- Abrir anaconda prompt e instalar theano:
```c
pip install theano
```

4.    Instalar tensorflow
- Abrir anaconda prompt e instalar theano:
```c
pip install tensorflow
```

5.    Instalar keras
- Abrir anaconda prompt e instalar keras:
```c
pip install keras
```

6.    Atualizar todos os programas recém instalados e outros pacotes do Anaconda
- Abrir anaconda prompt e atualizar:
```c
conda update –all
```

7.    Instalar o Vstudio 2015

8.    Instalar cuda_8.0.61_win10

9.    Instalar cuda_8.0.61.2_windows

10.    Incluir caminho no Path:
- edit system variables
- Environment variables
- System variables
- Path Edit
- New (inserir na última linha caminho que está instalado o cuda ex.: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\bin)

11.    Instalar GPU para utilização no Tensorflow
- https://www.tensorflow.org/install/install_windows

- Criar ambiente conda com nome tensorflow (abrir anaconda prompt)
```c
conda create -n tensorflow python=3.5
```
- Ativar ambiente conda (abrir anaconda prompt)
```c
activate tensorflow
```
- Atualizar tensorflow (abrir anaconda prompt)
```c
pip install --ignore-installed --upgrade tensorflow
```
- Instalar GPU versão do tensorflow
```c
pip install --ignore-installed --upgrade tensorflow-gpu
```
- Testar sucesso instalação digitando os comandos abaixo no spyder (phynton)
```c
import tensorflow as tf
```
```c
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
```
- Se a instalação foi concluída com sucesso a saída (Console Phyton - Spyder) irá retornar
```c
Hello, TensorFlow!
```

12.    Criar a rede CNN baseado na ZFnet

"""
Spyder Editor
"""
# Part 1 – Construindo a rede Aquino&VitóriaNet (CNN) baseado na ZFNET

- Importando o tensorflow, keras, pacotes e bibliotecas
```c
import tensorflow as tf
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Convolution2D
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import MaxPool2D
from keras.layers import BatchNormalization
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.utils import multi_gpu_model
from keras.preprocessing.image import ImageDataGenerator
import json
```

- Definindo as variáveis e funções de otimizações para melhorar aprendizado no treinamento
```c
sgd = SGD(lr=0.01, decay=1e-2, momentum=0.9, nesterov=True)

rms = RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0)
```

- Inicializando a rede Aquino&VitóriaFNET CNN

- O tipo da rede será sequencial que é uma característica da CNN
```c
classifier = Sequential()
```

- Step 1 – Convolution – 96 camadas, kernel 7x7, imagem entrada 32x32x3(RGB), função de ativação ReLU
```c
classifier.add(Convolution2D(96, 7, 7, border_mode='same', input_shape=(32, 32, 3), activation = 'relu'))
```

- Step 2 - Max Pooling – tamanho 3x3, passo2
```c
classifier.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same', data_format=None))
```

- Step 3 - Contrast Image – aplicando operação de contraste local de normalização na saída do mapa de caracteríscica (feature map)
```c
classifier.add(BatchNormalization())
classifier.compile(optimizer = 'RMSprop', loss = 'categorical_crossentropy')
```

- Step 4 – Convolution - 256 camadas, kernel 5x5, imagem entrada 16x16x256 função de ativação ReLU
```c
classifier.add(Convolution2D(256, 5, 5, border_mode='same', activation = 'relu'))
```


- Step 5 - Max Pooling - tamanho 3x3, passo2
```c
classifier.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same', data_format=None))
```

- Step 6 - Contrast Image - aplicando operação de contraste local de normalização na saída do mapa de caracteríscica (feature map)
```c
classifier.add(BatchNormalization())
classifier.compile(optimizer = 'RMSprop', loss = 'categorical_crossentropy')
```

- Step 7 – Convolution - 512 camadas, kernel 3x3, imagem entrada 8x8x512, função de ativação ReLU
```c
classifier.add(Convolution2D(512, 3, 3, border_mode='same', activation = 'relu'))
```

- Step 8 - Convolution - 1024 camadas, kernel 3x3, imagem entrada 8x8x1024, função de ativação ReLU
```c
classifier.add(Convolution2D(1024, 3, 3, border_mode='same', activation = 'relu'))
```

- Step 9 - Convolution - 512 camadas, kernel 3x3, imagem entrada 8x8x512, função de ativação ReLU
```c
classifier.add(Convolution2D(512, 3, 3, border_mode='same', activation = 'relu'))
```

- Step 10 - Max Pooling - tamanho 3x3, passo2
```c
classifier.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same', data_format=None))
```

- Step 11 – Flattening – preparação da saída da última etapa de convolução conforme necessário para entrada na camada primeira totalmente conectada com 8192 saídas (dimensões).
```c
classifier.add(Flatten())
```

- Step 12  - Full Connection – primeira camada totalmente conectada com 4096 dimensões, 33.558.528  parâmetros e função de ativação ReLU, desligamento de 50% dos neurônios (dropout). A segunda camada totalmente conectada com 4096 dimensões, 16.781.312  parâmetros e função de ativação ReLU, desligamento de 50% dos neurônios (dropout). A última camada classificadora do tipo softmax com três saídas (carro, gato ou cachorro) com 12.291 parâmetros. No caso da ZFNet a saída da última camada seria 1000, em funções da possibilidade de classificação de 1000 classes diferentes.
```c
classifier.add(Dense(output_dim = 4096, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(output_dim = 4096, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(output_dim = 3, activation = 'softmax'))
```

- Compiling the CNN – Compilando a rede neural para classificação final (acuracidade) e verificação da arquitetura da rede
```c
classifier.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics = ['accuracy'])

classifier.summary()
```
 
# Part 2 – Fazendo a ligação da rede neural com as imagens para treino, teste/validação
```c
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
                                    rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=(32, 32),
                                                batch_size=12,
                                                class_mode='categorical')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(32, 32),
                                            batch_size=12,
                                            class_mode='categorical')

classifier.fit_generator(training_set,
                        steps_per_epoch=12000,
                        epochs=30,
                        validation_data=test_set,
                        validation_steps=3000)
```

# Part 3 – Salvando os pesos após treino (aprendizado “memória” da rede)
```c
classifier.save('TesteFinal.h5')
print('Classifier Saved')
```
# Part 4 – Fazendo novas predições com a rede já treinada
```c
import numpy as np
from keras.preprocessing import image

test_image = image.load_img('dataset/single_prediction/cat_or_dog_or_car_3.jpg', target_size = (32, 32))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
print(result)
training_set.class_indices

test_image = image.load_img('dataset/single_prediction/cat_or_dog_or_car_2.jpg', target_size = (32, 32))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
print(result)
training_set.class_indices

test_image = image.load_img('dataset/single_prediction/cat_or_dog_or_car_1.jpg', target_size = (32, 32))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
print(result)
training_set.class_indices

test_image = image.load_img('dataset/single_prediction/cat_or_dog_or_car_4.jpg', target_size = (32, 32))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
print(result)
training_set.class_indices
```

# Part 5 – Salvando a estrutura do modelo
```c
classifier_json = classifier.to_json()
with open("Aquino&VitótiaNetStructure.json", "w") as json_file:
    json_file.write(classifier_json)
```
