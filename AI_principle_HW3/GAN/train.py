import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import model
from utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--latent_dim', type = int, default = 10)
parser.add_argument('--figsize', type = int, default = 32)
parser.add_argument('--batch_size', type = int, default = 16)
parser.add_argument('--epochs', type = int, default = 1000)
parser.add_argument('--resume', type = str, default = '')
parser.add_argument('--save', type = str, default = './checkpoint')
parser.add_argument('--pics_dir', type = str, default = './pic')
parser.add_argument('--dataset_dir', type = str, default = './dataset_flowers/sunflower')
parser.add_argument('--lr', type = float, default = 1e-5)
args = parser.parse_args()

latent_dim = args.latent_dim
figsize = args.figsize
batch_size = args.batch_size
epochs = args.epochs
checkpoint_dir = args.save
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
resume_dir = args.resume
pics_dir = args.pics_dir
dataset_dir = args.dataset_dir
lr = args.lr

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

if not os.path.exists(pics_dir):
    os.makedirs(pics_dir)


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def train_step(images, noise):
    noise = noise

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return gen_loss, disc_loss

def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    plt.cla()
    predictions = model(test_input, training=False)
    plt.imshow((predictions[0, :, :, :] / 2.0) + 0.5)
    plt.savefig(os.path.join(pics_dir, "image_at_epoch_{:04d}.jpg".format(epoch)))
    return

def train(dataset, epochs):
    ### input the noise
    
    for epoch in range(epochs):
        noise = tf.random.normal(shape = [1, latent_dim], seed = 1)
        gen_loss = 0
        disc_loss = 0
        print("Epoch: ", epoch)
        for image_batch in tqdm(dataset):
            std_gaussian = get_stddev(epoch)
            gaussian_noise = tf.random.normal(shape = [batch_size, latent_dim], mean = 0.0, 
                                              stddev = std_gaussian, dtype=tf.float32)
            image_batch += gaussian_noise
            g_loss, d_loss = train_step(image_batch, noise)
            gen_loss += (g_loss)
            disc_loss += (d_loss)
            
        print("Generator loss: ", gen_loss / len(dataset))
        print("Discriminator loss: ", disc_loss / len(dataset))
        gen_losses.append(gen_loss/len(dataset))
        disc_losses.append(disc_loss/len(dataset))
        plot_losses(gen_losses, disc_losses)
        # if (epoch + 1) % 20 == 0:
        #     checkpoint.save(file_prefix = checkpoint_prefix)
        generate_and_save_images(generator, epoch + 1, noise)


    # Generate after the final epoch
    generate_and_save_images(generator, epochs, noise)


def get_stddev(epoch, initial_stddev=1.0, decay_rate=0.002):
    return initial_stddev / (1 + decay_rate * epoch)

if __name__ == '__main__':
    ### Load dataset from folder
    dataset = keras.utils.image_dataset_from_directory(
        "./dataset_flowers", label_mode=None, seed=123, image_size=(figsize, figsize), batch_size=batch_size,
    )

    dataset = dataset.map(lambda x: tf.clip_by_value((x - 127.5) / 127.5, -1, 1)) # - 127.0
    

    ### Build the model
    generator = model.make_generator_model()
    discriminator = model.make_discriminator_model()
    
    ### Define the loss and optimizers
    gen_losses = []
    disc_losses = []
    generator_optimizer = tf.keras.optimizers.Adam(lr)
    discriminator_optimizer = tf.keras.optimizers.Adam(lr)
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator,
                                 history=dict(gen_losses=gen_losses, disc_losses=disc_losses)
                                 )
    ### train the model
    train(dataset, epochs)

