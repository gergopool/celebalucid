# CelebA & Lucid

![cover_image](https://i.imgur.com/RVzTfhv.png)

This package is a quick access tool of a research project of Alfred Renyi Institute of Mathematics. In our project we use lucid visualizations to get a better understanding how transfer learning manipulates a neural network. As a baseline we took Clarity's InceptionV1 network trained on imagenet and we further trained the network on the CelebA dataset.

In our project we also revisit and rethink the comparison of activations and features.

# Networks

We've trained numerous neural networks. At the first stage, we trained three: one learning 100 epochs with SGD optimizer, one learning 10 epochs with Adam and one learning 100 epochs with Adam. We reference these trainings later as **sgd, adam and adam_overfit**, while the baseline will be referenced as **imagenet**. Then, we took a permuation of some sampling parameters. We defined 3 different gradient noise (the order of batches the neural network is fed) and 3+1 networks. The 3 networks were initialized with 3 different, randomly sampled weights, while the +1 is the network which were pretrained on imagenet.

| | GradNoise1 (gn1) | GradNoise2 (gn2) | GradNoise3 (gn3) |
| --- | :---: | :---: | :---: |
| **Imagenet (img)** | &#9745; | &#9745; | &#9745; |
| **InitNoise1 (in1)** | &#9745; | &#9745; | &#9745; |
| **InitNoise2 (in2)** | &#9745; | &#9745; | &#9745; |
| **InitNoise3 (in3)** | &#9745; | &#9745; | &#9745; |


Details of the trainings:

| Name | Epochs | Train loss | Train Acc | Val loss | Val acc |
| --- | --- | --- | --- | --- | --- |
| sgd  | 100 | 0.16 | 92.7% | 0.18 | 91.9% |
| adam  | 10 | 0.16 | 92.9% | 0.18 | 91.9% |
| adam_overfit  | 100 | 0.01 | 99.3% | 0.93 | 90.8% |
| img-gn1 | 20 | 0.11 | 94.90% | 0.21 | 91.63% |
| img-gn2 | 20 | 0.11 | 94.89% | 0.21 | 91.60% |
| img-gn3 | 20 | 0.11 | 94.88% | 0.21 | 91.49% |
| in1-gn1 | 20 | 0.17 | 92.33% | 0.18 | 91.70% |
| in1-gn2 | 20 | 0.17 | 92.32% | 0.19 | 91.56% |
| in1-gn3 | 20 | 0.17 | 92.29% | 0.19 | 91.53% |
| in2-gn1 | 20 | 0.17 | 92.36% | 0.18 | 91.73% |
| in2-gn2 | 20 | 0.17 | 92.35% | 0.19 | 91.60% |
| in2-gn3 | 20 | 0.17 | 92.33% | 0.19 | 91.56% |
| in3-gn1 | 20 | 0.17 | 92.32% | 0.18 | 91.69% |
| in3-gn2 | 20 | 0.17 | 92.32% | 0.19 | 91.59% |
| in3-gn3 | 20 | 0.17 | 92.32% | 0.19 | 91.60% |

# Setup

## Requirements

Python 3.6+

## Install

```bash
pip install celebalucid
```

# Usage

## Load

First, load either of the networks as
```python
from celebalucid import load_model

# Choose from 
# 'imagenet', 'sgd', 'adam', 'adam_overfit'
# 'img-gn1', 'img-gn2', 'img-gn3'
# 'in1-gn1', 'in1-gn2', 'in1-gn3'
# 'in2-gn1', 'in2-gn2', 'in2-gn3'
# 'in3-gn1', 'in3-gn2', 'in3-gn3'
model = load_model('imagenet') 
```

## Change weights
In order to avoid the redefinition of models and unnecessary memory usage, you can update the state dictionary of a model.
```python
model.switch_to('sgd') 
```



## Layer information

Get accessable layers as
```python
model.layer_info
```
Calling this you will see the list of available layers along with the available number of channels (neurons).
```bash
Out: [['conv2d0', 64],
      ['conv2d1', 64],
      ['conv2d2', 192],
      ['mixed3a_1x1', 64],
      ['mixed3a_3x3_bottleneck', 96],
      ...]
```

## Lucid

Visualize a neuron with lucid as

```python
# Default args: size=224, thresholds=[512], progress=False
img = model.lucid('mixed5a_5x5:7', progress=True)
plt.imshow(img)
```
![result_image](https://i.imgur.com/PBBuG1r.png)

## Weights

You can access the weights of a network as

```python
# Method 1
weights = model.weights.mixed3a_3x3_bottleneck.w 
bias = model.weights.mixed3a_3x3_bottleneck.b

# Method 2
weights = model.weights['mixed3a_3x3_bottleneck'].w 
bias = model.weights['mixed3a_3x3_bottleneck'].b
```

## Activations

In order to see activations, you first need to download the test data (~300MB)

```python
from celebalucid import build_generator

# The test data will be downloaded to this folder
DOWNLOAD_DIR = 'res/images'

# Choose dataset 'celeba' or 'imagenet'
DATASET = 'imagenet'

# This is a torch.utils.data.DataLoader object
# If you already have the dataset downloaded
# to this folder, it will skip the download
generator = build_generator(DOWNLOAD_DIR, DATASET)
```

Now, utilize the data generator to load images from disk, feed the input with the neural network and then you're free to analyse the activations (aka. neurons.)

```python
for x, y in generator:
    model.stream(x)
    # Now access neurons the same way as on weights
    some_operations(model.neurons.conv2d0)
```

## CKA similarity

The code provides a centered kernel alignment (CKA) comparison between models. In each epoch, the script runs the models on `n_iters * batch_size` images and then calculates the CKA value for a given layer. It does the same for each epoch and it returns with the mean cka over the epochs.

Define your CKA analyser as

```python
from celebalucid import CKA

# Args: working_dir, dataset, models_to_compare
# Default kwargs: n_epochs=10, n_iters=64, batch_size=32
cka = CKA('res/images', 'celeba', ['imagenet', 'sgd'])
```
Then run your comparison on a specific layer
```python
# Optional: set verbose=False for no progress bar
cka('mixed4a_3x3') 
```
```bash
Out: 0.95517987
```

If you want to choose models, instead of creating a new CKA object set new models in order to be GPU RAM efficient:

```python
cka.set_models(['imagenet', 'adam']) 
```

# Weight manipulation

You have the possibility to override a network's weights with constant or another model's weights. 

```python
imagenet = load_model('imagenet')
sgd = load_model('sgd')

# Change an entire layer to imagenet
sgd.set_weights('mixed4a_3x3', imagenet)

# Change a single neuron
sgd.set_weights('mixed4a_3x3:0', imagenet)

# Change to a constant
sgd.set_weights('mixed4a_3x3', 0)
```
It's good to know that by default both weights an biases change. If you know you need only one of these, you can save a bit of computation.
```python
# Default behaviour
sgd.set_weights('mixed4a_3x3', 0, mode='both')

# Change weights only
sgd.set_weights('mixed4a_3x3', 0, mode='weight')

# Change bias only
sgd.set_weights('mixed4a_3x3', 0, mode='bias')
```




# Credits

```bash
https://github.com/tensorflow/lucid
https://github.com/greentfrapp/lucent
```
# References
```
@inproceedings{pmlr-v97-kornblith19a,
  title = {Similarity of Neural Network Representations Revisited},
  author = {Kornblith, Simon and Norouzi, Mohammad and Lee, Honglak and Hinton, Geoffrey},
  booktitle = {Proceedings of the 36th International Conference on Machine Learning},
  pages = {3519--3529},
  year = {2019},
  volume = {97},
  month = {09--15 Jun},
  publisher = {PMLR}
}
```