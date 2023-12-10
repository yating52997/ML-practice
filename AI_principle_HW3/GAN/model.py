import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
# Bring in the sequential api for the generator and discriminator
from tensorflow.keras.models import Sequential
# Bring in the layers for the neural network
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Reshape, LeakyReLU, Dropout, UpSampling2D, BatchNormalization
from train import latent_dim, figsize

# def make_generator_model():
#     model = tf.keras.Sequential()
#     model.add(layers.Dense(8*8*256, use_bias=False, input_shape=(latent_dim,)))
#     model.add(layers.BatchNormalization())
#     model.add(layers.LeakyReLU())

#     model.add(layers.Reshape((8, 8, 256)))

#     model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
#     # output_shape = (batch_size, 8, 8, 128)
#     model.add(layers.BatchNormalization())
#     model.add(layers.LeakyReLU())

#     model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
#     # output_shape = (batch_size, 16, 16, 64)
#     model.add(layers.BatchNormalization())
#     model.add(layers.LeakyReLU())

#     model.add(layers.Conv2DTranspose(32, (5, 5), strides=(1, 1), padding='same', use_bias=False))
#     # output_shape = (batch_size, 16, 16, 32)
#     model.add(layers.BatchNormalization())
#     model.add(layers.LeakyReLU())

#     model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
#     # output_shape = (batch_size, 32, 32, 3)
#     return model

def make_generator_model():
    model = Sequential()
    
    # Takes in random values and reshapes it to 7x7x128
    # Beginnings of a generated image
    # model.add(BatchNormalization())
    model.add(Dense(32*32*128))
    model.add(BatchNormalization())
    # model.add(LeakyReLU(0.2))
    model.add(Reshape((32,32,128)))
    
    # model.add(Conv2D(128, 5, padding='same'))
    # model.add(BatchNormalization())
    # Upsampling block 1 
    model.add(UpSampling2D())
    model.add(Conv2D(128, 5, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    
    # Upsampling block 2 
    model.add(UpSampling2D())
    model.add(Conv2D(128, 5, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    
    # Convolutional block 1
    model.add(Conv2D(64, 3, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    
    # Convolutional block 2
    model.add(Conv2D(64, 3, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(64, 3, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(64, 3, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    
    # Conv layer to get to one channel
    model.add(Conv2D(3, 3, padding='same'))
    model.add(Activation('tanh'))
    
    return model

from tensorflow.keras.layers import Input, Dense, Reshape, Conv2DTranspose, BatchNormalization, LeakyReLU, Activation, MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
# def make_generator_model():
#     model = tf.keras.Sequential()
#     model.add(layers.Dense(64*64*3, use_bias=False, input_shape=(latent_dim,)))
#     model.add(layers.Reshape((64,64,3)))
#     # downsampling
#     model.add(Conv2D(128,4, strides=1, padding='same',kernel_initializer='he_normal', use_bias=False))
#     model.add(Conv2D(128,4, strides=2, padding='same',kernel_initializer='he_normal', use_bias=False))
#     model.add(BatchNormalization())
#     model.add(LeakyReLU())
#     model.add(Conv2D(256,4, strides=1, padding='same',kernel_initializer='he_normal', use_bias=False))
#     model.add(Conv2D(256,4, strides=2, padding='same',kernel_initializer='he_normal', use_bias=False))
#     model.add(BatchNormalization())
#     model.add(LeakyReLU())
#     model.add(Conv2DTranspose(512, 4, strides=1,padding='same',kernel_initializer='he_normal',use_bias=False))
#     model.add(Conv2D(512,4, strides=2, padding='same',kernel_initializer='he_normal', use_bias=False))
    
#     model.add( LeakyReLU())
#     #upsampling
#     model.add( Conv2DTranspose(512, 4, strides=1,padding='same',kernel_initializer='he_normal',use_bias=False))
#     model.add( Conv2DTranspose(512, 4, strides=2,padding='same',kernel_initializer='he_normal',use_bias=False))
#     model.add( BatchNormalization())
#     model.add( LeakyReLU())
#     model.add( Conv2DTranspose(256, 4, strides=1,padding='same',kernel_initializer='he_normal',use_bias=False))
#     model.add( Conv2DTranspose(256, 4, strides=2,padding='same',kernel_initializer='he_normal',use_bias=False))
#     model.add( BatchNormalization())
    
#     model.add( Conv2DTranspose(128, 4, strides=1,padding='same',kernel_initializer='he_normal',use_bias=False))
#     model.add( Conv2DTranspose(128, 4, strides=1,padding='same',kernel_initializer='he_normal',use_bias=False))
#     model.add( BatchNormalization())
#     model.add( Conv2DTranspose(3,4,strides = 1, padding = 'same',activation = 'tanh'))
    

#     return model

# def make_generator_model():
#     model = Sequential()

#     # Initial Dense layer for a modest upscale
#     model.add(Dense(128 * 2 * 2, input_dim=latent_dim, activation='relu'))
#     model.add(Reshape((2, 2, 128)))

#     # Gradual upsampling and refinement
#     model.add(Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same'))
#     model.add(BatchNormalization())
#     model.add(LeakyReLU(alpha=0.2))

#     model.add(Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same'))
#     model.add(BatchNormalization())
#     model.add(LeakyReLU(alpha=0.2))

#     # Adjusting the number of upsampling layers based on the final image size
#     # Adding more layers for larger figsize
#     upsampling_layers = [
#         (128, (3, 3)),
#         (64, (3, 3)),
#         # Add more layers or adjust strides for larger output sizes
#     ]
#     current_size = 8  # Adjust based on the initial dense layer and strides above
#     for filters, kernel_size in upsampling_layers:
#         if current_size < figsize:
#             model.add(Conv2DTranspose(filters, kernel_size, strides=(2, 2), padding='same'))
#             model.add(BatchNormalization())
#             model.add(LeakyReLU(alpha=0.2))
#             current_size *= 2

#     # Ensure the final layer has the correct size
#     if current_size != figsize:
#         final_stride = (figsize // current_size, figsize // current_size)
#         model.add(Conv2DTranspose(64, (3, 3), strides=final_stride, padding='same'))

#     # Output layer
#     model.add(Conv2DTranspose(3, (3, 3), activation='tanh', padding='same'))

#     return model
# def make_generator_model():
#     model = Sequential()

#     # Foundation for 4x4 feature maps
#     n_nodes = 256 * 4 * 4
#     model.add(Dense(n_nodes, input_dim=latent_dim))
#     model.add(LeakyReLU(alpha=0.2))
#     model.add(Reshape((4, 4, 256)))

#     # Upsample to 8x8
#     model.add(Conv2DTranspose(128, (5,5), strides=(2,2), padding='same'))
#     model.add(BatchNormalization())
#     model.add(LeakyReLU(alpha=0.2))

#     # Upsample to 16x16
#     model.add(Conv2DTranspose(128, (5,5), strides=(2,2), padding='same'))
#     model.add(BatchNormalization())
#     model.add(LeakyReLU(alpha=0.2))

#     # Upsample to 32x32
#     model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
#     model.add(BatchNormalization())
#     model.add(LeakyReLU(alpha=0.2))

#     # Upsample to figsize x figsize
#     final_stride = figsize // 32
#     model.add(Conv2D(128, (4,4), strides=(final_stride,final_stride), padding='same'))
#     model.add(BatchNormalization())
#     model.add(LeakyReLU(alpha=0.2))

#     model.add(Conv2D(64, (3,3), strides=(1,1), padding='same'))
#     model.add(BatchNormalization())
#     model.add(LeakyReLU(alpha=0.2))
    
    
#     model.add(Conv2D(32, (3,3), strides=(1,1), padding='same'))
#     model.add(BatchNormalization())
#     model.add(LeakyReLU(alpha=0.2))
#     # Output layer
#     model.add(Conv2D(3, (3,3), padding='same'))
    
#     model.add(Activation('tanh'))

#     return model


# def make_discriminator_model():
#     model = tf.keras.Sequential()
#     model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
#                                      input_shape=[figsize, figsize, 3]))
#     model.add(layers.LeakyReLU())
#     model.add(layers.Dropout(0.3))

#     model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
#     model.add(layers.LeakyReLU())
#     model.add(layers.Dropout(0.3))

#     model.add(layers.Flatten())
#     model.add(layers.Dense(1))

#     return model

def make_discriminator_model(): 
    model = Sequential()
    
    # First Conv Block
    model.add(Conv2D(16, 5, input_shape = (figsize, figsize, 3)))
    model.add(BatchNormalization())
    model.add(MaxPool2D())
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))
    
    # Second Conv Block
    model.add(Conv2D(32, 5))
    model.add(BatchNormalization())
    model.add(MaxPool2D())
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))
    
    # Third Conv Block
    model.add(Conv2D(64, 5))
    model.add(BatchNormalization())
    model.add(MaxPool2D())
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))
    
    # Fourth Conv Block
    # model.add(Conv2D(256, 5))
    # model.add(BatchNormalization())
    # model.add(LeakyReLU(0.2))
    # model.add(Dropout(0.4))
    
    # Flatten then pass to dense layer
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1))
    
    return model 

if __name__ == '__main__':
    generator = make_generator_model()
    discriminator = make_discriminator_model()
    generator.summary()
    discriminator.summary()