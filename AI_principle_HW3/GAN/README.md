# GAN

It is the model train on flowers dataset


The goal us to generate flowers

## How to train

```bash
python train.py
```
### hyperparameters
```python
parser.add_argument('--latent_dim', type = int, default = 256)
parser.add_argument('--figsize', type = int, default = 32)
parser.add_argument('--batch_size', type = int, default = 16)
parser.add_argument('--epochs', type = int, default = 100)
parser.add_argument('--resume', type = str, default = '')
parser.add_argument('--save', type = str, default = './checkpoint')
parser.add_argument('--pics_dir', type = str, default = './pic')
parser.add_argument('--dataset_dir', type = str, default = './dataset_flowers')
parser.add_argument('--lr', type = float, default = 1e-4)
```

## directories
- train.py
- model.py
- gen_gif.py
- pics
    - flower_big.gif
    - flower_small.gif
    - flower_small_256.gif


## train results
train on RTX4050 (6 GB)

generator input
noise = tf.random.normal(shape = [1, latent_dim], seed = 1)

**latent_dim = 10**

epochs = 100, batch_size = 16, time = 40 sec/epoch
![Flower small](pics/flower_small.gif)



**latent_dim = 256**

epochs = 100, batch_size = 16, time = 40 sec/epoch
![Flower small](pics/flower_small_256.gif)



**atent_dim = 1024**

epochs = 500, batch_size = 32, time = 30 sec/epoch
![Flower Big](pics/flower_big.gif)