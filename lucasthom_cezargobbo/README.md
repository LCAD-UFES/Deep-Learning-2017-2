## XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks.
Implementado usando Torch 7.0

### Requisitos
[Torch com CUDA](http://torch.ch/docs/getting-started.html#_)
[Dataset Imagenet. 1000 classes e 1.2M imagens.](http://image-net.org/download-images)
[Baixar estes arquivos e extrair nesta pasta](https://s3-us-west-2.amazonaws.com/ai2-vision/xnornet/cache.tar)
```bash
tar -xvf ./cache.tar
```

### Treinamento
```bash
th main.lua -data [path to ImageNet dataset] -nGPU 4 -batchSize 800 -netType alexnetxnor -binaryWeight -optimType adam -epochSize 1500
```

### Modelo Pretreinado
[XNOR-Network(pesos)](https://s3-us-west-2.amazonaws.com/ai2-vision/xnornet/alexnet_XNOR.t7)
Preprocessamento da imagem fazendo o seguinte:
```bash
find . -name "*.JPEG" | xargs -I {} convert {} -resize "256^>" {}
``` 
