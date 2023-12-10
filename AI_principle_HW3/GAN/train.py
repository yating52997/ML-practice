import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import model
from IPython import display


figsize = 128
latent_dim = 1000
batch_size = 8
epochs = 1000
checkpoint_dir = './training_checkpoints_1'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")


### The code that can plot like figure 1
def plot_latent_space(vae, n=5):
    # display an n*n 2D manifold of digits
    digit_size = figsize
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n, 3))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            rest_latent = np.ones((1,latent_dim-2))*yi
            z_sample = np.array([[xi, yi]])
            z_sample = np.concatenate((z_sample, rest_latent), axis=1)
            x_decoded = vae.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size, 3)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure)
    # plt.show()
    plt.savefig('out.png')
    return

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
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    # for i in range(predictions.shape[0]):
    #     plt.subplot(4, 4, i+1)
    plt.imshow((predictions[0, :, :, :] * 0.5 ) +0.5) #+ 127.0 
        # plt.axis('off')

    plt.savefig(os.path.join("./pic_3", "image_at_epoch_{:04d}.jpg".format(epoch)))
    return

def train(dataset, epochs):
    ### input the noise
    
    # test_input = tf.random.normal([1, latent_dim], seed = 1)
    for epoch in range(epochs):
        noise = tf.random.normal(mean = 0,stddev= 1 , shape=[1, latent_dim],seed=1)
        # noise = tf.random.normal([batch_size, 8, 8, 3], seed = 1)
        gen_loss = []
        disc_loss = []
        print("Epoch: ", epoch)
        for image_batch in tqdm(dataset):
            g_loss, d_loss = train_step(image_batch, noise)
            gen_loss.append(g_loss)
            disc_loss.append(d_loss)
            # Save the model every 15 epochs
            # if (epoch + 1) % 15 == 0:
            #     checkpoint.save(file_prefix = checkpoint_prefix)
        print("Generator loss: ", np.mean(gen_loss))
        print("Discriminator loss: ", np.mean(disc_loss))
        # Produce images for the GIF as you go
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
        if (epoch + 1) % 1 == 0:
            display.clear_output(wait=True)
            generate_and_save_images(generator, epoch + 1, noise)


    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, noise)

if __name__ == '__main__':
    ### Load dataset from folder
    dataset = keras.utils.image_dataset_from_directory(
        "./dataset_flowers", label_mode=None, seed=123, image_size=(figsize, figsize), batch_size=batch_size,
    )
    # for images in dataset.take(1):
    #     plt.imshow(images[0].numpy().astype("uint8"))
    #     plt.savefig('./dataset_flowers.png')
    #     break
    # mean = np.mean(dataset)
    # std = np.std(dataset)
    dataset = dataset.map(lambda x: (x - 127.0)/127.0) # - 127.0
    
    ### Show the figure
    # for x in dataset:
    #     plt.axis("off")
    #     plt.imshow((x.numpy() * 255).astype("int32")[0])
    #     plt.savefig('./pic/dataset_flowers.png')
    #     break

    ### Build the model
    generator = model.make_generator_model()
    discriminator = model.make_discriminator_model()
    

    
    
    ### Define the loss and optimizers
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
    ### train the model
    train(dataset, epochs)

