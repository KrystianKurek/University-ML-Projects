import os
from datetime import datetime

import tensorflow as tf
from tqdm import trange
import numpy as np


def return_callbacks(best_model_filepath, patience):
    '''
    Trains a machine learning model based on the specified architecture.

    Arguments:
        * `best_model_filepath`: The path to save the best model.
        * `patience`: The number of epochs with no improvement after which training will be stopped.

    Returns:
        * A list of callbacks to be used during model training, including EarlyStopping to prevent overfitting, 
        ModelCheckpoint to save the best model, and ReduceLROnPlateau to reduce the learning rate if the model 
        stops improving.
    '''
    best_model_kwargs = {'monitor': 'val_loss', 'verbose': 1, 'mode': 'min',
                         'save_best_only': True, 'filename': 'best_model.ckpt', 'save_format': "tf",
                         "save_weights_only": True, "min_delta": 0.0001}

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, min_delta=0.0001,
                                                  verbose=1, restore_best_weights=True),
                 tf.keras.callbacks.ModelCheckpoint(os.path.join(best_model_filepath, best_model_kwargs['filename']),
                                                    **best_model_kwargs),
                 tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=max(patience//2, 1),
                                                      verbose=1)]
    return callbacks


def return_simple_autoencoder(latent_dim, x_shape, num_filters=32, use_maxpooling=True,
                              n_blocks=2, use_upsampling=True):
    '''
    Returns a simple autoencoder with the given parameters.

    Arguments:
        * latent_dim: The size of the latent space.
        * x_shape: The shape of the input data.
        * num_filters (optional): The number of filters in the convolutional layers. Default is 32.
        * use_maxpooling (optional): Whether to use maxpooling layers. Default is True.
        * n_blocks (optional): The number of encoder/decoder blocks to use. Default is 2.
        * use_upsampling (optional): Whether to use upsampling layers. Default is True.

    Returns:
        * A tf.keras.Model object representing the autoencoder.
    '''
    autoencoder = SimpleAutoencoder(latent_dim=latent_dim,
                              x_shape=x_shape,
                              num_filters=num_filters,
                              n_blocks=n_blocks,
                              use_maxpooling=use_maxpooling,
                              use_upsampling=use_upsampling)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    autoencoder.compile(optimizer=optimizer, loss=masked_mse)
    return autoencoder


def return_autoencoder(latent_dim, x_shape, num_filters, use_maxpooling,
                       n_blocks, use_upsampling, use_true_labels, all_classes, now, batch_size, patience, best_model_filepath):
    '''
    Returns a DLMM model based on the given autoencoder architecture.

    Arguments:
        * `latent_dim`: The dimensionality of the latent space.
        * `x_shape`: The shape of the input data.
        * `num_filters`: The number of filters in the convolutional layers.
        * `use_maxpooling`: Whether to use maxpooling layers.
        * `n_blocks`: The number of encoder/decoder blocks to use.
        * `use_upsampling`: Whether to use upsampling layers.
        * `use_true_labels`: Whether to use true labels when training the DLMM.
        * `all_classes`: A list of all possible classes.
        * `now`: The current date/time, used for naming files.
        * `batch_size`: The batch size for training the DLMM.
        * `patience`: The number of epochs to wait for improvement in validation loss before early stopping.

    Returns:
        * A DLMM object based on the given autoencoder architecture.
    '''
    autoencoder = Autoencoder(latent_dim=latent_dim,
                              x_shape=x_shape,
                              num_filters=num_filters,
                              n_blocks=n_blocks,
                              use_maxpooling=use_maxpooling,
                              use_upsampling=use_upsampling)
    autoencoder.compile(tf.keras.optimizers.Adam(learning_rate=0.1))

    return DLMM(autoencoder, batch_size, patience, max(patience//2,1),
            all_classes=all_classes, use_true_labels=use_true_labels, now=now, best_model_filepath=best_model_filepath)


def masked_mse(y_true, y_pred, gamma=0.1):
    '''
    This function calculates the masked mean squared error (MSE) between the true and predicted values.

    Arguments:
        * y_true: The true values of the data.
        * y_pred: The predicted values of the data.
        * gamma (optional): The regularization parameter. Default is 0.1.

    Returns:
        * The masked MSE between y_true and y_pred.
        '''
    mask = tf.cast(tf.equal(y_true, 0), tf.float32)
    mse = tf.reduce_mean(tf.square(y_true - y_pred) * (1 - mask) + gamma * tf.square(y_true - y_pred) * mask)
    return mse


def encoder_block(inputs, num_filters, use_maxpooling=True, block_idx=None):
    '''
    Returns an encoder block that consists of two 2D convolutional layers, batch normalization, activation and (optionally) max pooling.

    Arguments:
        * inputs: The input tensor.
        * num_filters: The number of filters in the convolutional layers.
        * use_maxpooling (optional): Whether to use max pooling layer. Default is True.
        * block_idx (optional): The index of the block. Default is None.

    Returns:
        * A tensor representing the output of the encoder block.
    '''
    encoder = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='valid', name=f"{block_idx}_conv2d_1")(inputs)
    encoder = tf.keras.layers.BatchNormalization(name=f"{block_idx}_bn_1")(encoder)
    encoder = tf.keras.layers.Activation('relu', name=f"{block_idx}_relu_1")(encoder)

    encoder = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='valid', name=f"{block_idx}_conv2d_2")(encoder)
    encoder = tf.keras.layers.BatchNormalization(name=f"{block_idx}_bn_2")(encoder)
    encoder = tf.keras.layers.LeakyReLU(name=f"{block_idx}_relu_2")(encoder)

    if use_maxpooling:
        encoder = tf.keras.layers.MaxPooling2D((2, 2), name=f"{block_idx}_max_pooling")(encoder)

    return encoder


def decoder_block(inputs, num_filters, use_maxpooling, block_idx, strides_kernels_and_padding, use_upsampling=True):
    '''
    Returns a decoder block of a convolutional neural network with the given parameters.

    Arguments:
        * inputs: The input tensor.
        * num_filters: The number of filters in the convolutional layers.
        * use_maxpooling (optional): Whether to use maxpooling layers. Default is True.
        * block_idx: The index of the decoder block.
        * strides_kernels_and_padding (optional): A tuple of 3 elements representing the stride, kernel size, and padding
                                                of the convolutional transpose layer. Default is None.
        * use_upsampling (optional): Whether to use upsampling layers. Default is True.

    Returns:
        * A tensor representing the output of the decoder block.
    '''
    if use_maxpooling:
        if not use_upsampling:
            decoder = tf.keras.layers.Conv2DTranspose(num_filters, kernel_size=strides_kernels_and_padding[1],
                                                      strides=strides_kernels_and_padding[0],
                                                      output_padding=strides_kernels_and_padding[2],
                                                      padding='valid', name=f"{block_idx}_conv2d_1_d")(inputs)
        else:
            decoder = tf.keras.layers.UpSampling2D(size=(2, 2), name=f"{block_idx}_upsampling")(inputs)
            if strides_kernels_and_padding and (0, 0) not in strides_kernels_and_padding[:-1]:
                decoder = tf.keras.layers.Conv2DTranspose(num_filters, kernel_size=strides_kernels_and_padding[1],
                                                          strides=strides_kernels_and_padding[0],
                                                          output_padding=strides_kernels_and_padding[2],
                                                          padding='valid', name=f"{block_idx}_conv2d_1_d")(decoder)

    else:
        decoder = inputs
    decoder = tf.keras.layers.Conv2DTranspose(num_filters, (3, 3), padding='valid', name=f"{block_idx}_conv2d_2_d")(
        decoder)
    decoder = tf.keras.layers.BatchNormalization(name=f"{block_idx}_bn_1_d")(decoder)
    decoder = tf.keras.layers.LeakyReLU(name=f"{block_idx}_relu_1_d")(decoder)

    decoder = tf.keras.layers.Conv2DTranspose(num_filters, (3, 3), padding='valid', name=f"{block_idx}_conv2d_3_d")(
        decoder)
    decoder = tf.keras.layers.BatchNormalization(name=f"{block_idx}_bn_2_d")(decoder)
    decoder = tf.keras.layers.LeakyReLU(name=f"{block_idx}_relu_2_d")(decoder)

    return decoder


def return_encoder(input_shape, latent_dim, num_filters, use_maxpooling, n_blocks):
    '''
    Returns an encoder model with the given parameters.

    Arguments:
        * input_shape: The shape of the input data.
        * latent_dim: The size of the latent space.
        * num_filters (optional): The number of filters in the convolutional layers. Default is 32.
        * use_maxpooling (optional): Whether to use maxpooling layers. Default is True.
        * n_blocks (optional): The number of encoder blocks to use. Default is 2.

    Returns:
        * A tf.keras.Model object representing the encoder.
    '''
    input_ = tf.keras.layers.Input(input_shape)
    encoder = input_
    for i in range(n_blocks):
        encoder = encoder_block(encoder, num_filters, use_maxpooling=use_maxpooling, block_idx=i)
    encoder = tf.keras.layers.Flatten()(encoder)
    output = tf.keras.layers.Dense(latent_dim)(encoder)
    encoder = tf.keras.Model(inputs=input_, outputs=output)
    return encoder


def return_decoder(latent_dim, encoder, num_filters, use_maxpooling, n_blocks, strides_kernels_and_padding,
                   use_upsampling):
    '''
    Constructs a decoder network that takes as input a tensor of shape `(latent_dim,)` and outputs an image tensor of shape `(height, width, 1)`. 
    The decoder is constructed by performing the inverse operations of the encoder. 

    Arguments:
        * latent_dim: integer, dimensionality of the latent space.
        * encoder: a tf.keras.Model object representing the encoder.
        * num_filters: integer, number of filters to use in the decoder blocks.
        * use_maxpooling: boolean, indicating whether to use max pooling in the decoder blocks.
        * n_blocks: integer, number of decoder blocks to use.
        * strides_kernels_and_padding: dictionary, mapping block index to a tuple of (stride, kernel size, padding)
        * use_upsampling: boolean, indicating whether to use upsampling in the decoder blocks.

    Returns:
        * decoder: a tf.keras.Model object representing the decoder.
    '''
    input_ = tf.keras.layers.Input(latent_dim)
    decoder = tf.keras.layers.Dense(encoder.layers[-2].output.shape[-1])(input_)

    decoder = tf.keras.layers.Reshape(encoder.layers[-3].output.shape[1:])(decoder)
    for block_idx in range(n_blocks):
        decoder = decoder_block(decoder, num_filters, use_maxpooling, block_idx,
                                strides_kernels_and_padding.get(block_idx, ()), use_upsampling)
    output = tf.keras.layers.Conv2DTranspose(1, (1, 1), padding='valid', activation='sigmoid')(decoder)
    decoder = tf.keras.Model(inputs=input_, outputs=output)
    return decoder


def get_strides_and_kernel_size_and_padding(x, y, w, z):
    '''
    This function takes as input four integers x, y, w, and z representing the input size and output size of a 
    deconvolutional layer. It calculates the stride, kernel size, and padding required for the deconvolutional layer such that 
    the input tensor is transformed into the output tensor.

    Arguments:
        * x: integer, representing the input size of the deconvolutional layer in the x direction.
        * y: integer, representing the input size of the deconvolutional layer in the y direction.
        * w: integer, representing the output size of the deconvolutional layer in the x direction.
        * z: integer, representing the output size of the deconvolutional layer in the y direction.

    Returns:
        * A tuple of three tuples representing the stride, kernel size, and padding required for the deconvolutional layer 
        respectively.
    '''
    s1 = max(1, (w - x) // max(x - 1, 1))
    k1 = w - (x - 1) * s1

    s2 = max(1, (z - y) // max(y - 1, 1))
    k2 = z - (y - 1) * s2

    # Calculate output padding
    p1 = max(0, (x - 1) * s1 + k1 - w)
    p2 = max(0, (y - 1) * s2 + k2 - z)

    return (s1, s2), (k1, k2), (p1, p2)


class SimpleAutoencoder(tf.keras.Model):
    '''
    Simple Autoencoder class
    '''
    def __init__(self, latent_dim, x_shape, num_filters=32, use_maxpooling=True, n_blocks=2, use_upsampling=True):
        '''
        Creates an instance of the SimpleAutoencoder class.

        Arguments:
        * latent_dim: an integer representing the dimensionality of the latent space of the autoencoder.
        * x_shape: a tuple of integers representing the shape of the input data (excluding the batch size).
        * num_filters: an integer representing the number of filters to use in the convolutional layers of the autoencoder.
        * use_maxpooling: a boolean indicating whether to use max pooling in the encoder.
        * n_blocks: an integer representing the number of convolutional blocks to use in the autoencoder.
        * use_upsampling: a boolean indicating whether to use upsampling in the decoder.
        '''
        super(SimpleAutoencoder, self).__init__()
        encoder = return_encoder(x_shape,
                                 latent_dim=latent_dim,
                                 num_filters=num_filters,
                                 use_maxpooling=use_maxpooling,
                                 n_blocks=n_blocks)

        strides_kernels_and_padding = {}
        if use_maxpooling:
            n_layers = len(encoder.layers)
            k = (n_layers - 1) // 7
            for block_idx in range(1, k + 1):
                relu_output_shape = get_layer_output_shape(encoder.layers[block_idx * 7 - 1])[:-1]
                max_pooling_output_shape = get_layer_output_shape(encoder.layers[block_idx * 7])[:-1]
                if use_upsampling:
                    max_pooling_output_shape = np.array(max_pooling_output_shape) * 2
                strides, kernel_size, output_padding = get_strides_and_kernel_size_and_padding(
                    *max_pooling_output_shape, *relu_output_shape)
                strides_kernels_and_padding[k - block_idx] = strides, kernel_size, output_padding

        self.decoder = return_decoder(latent_dim=latent_dim,
                                      encoder=encoder,
                                      num_filters=num_filters,
                                      use_maxpooling=use_maxpooling,
                                      n_blocks=n_blocks,
                                      strides_kernels_and_padding=strides_kernels_and_padding,
                                      use_upsampling=use_upsampling)
        self.encoder = encoder
        self.encoder.summary()
        self.decoder.summary()

    def get_embeddings(self, x, training=False):
        '''
        Returns emebeddings

        Arguments:
            * x: a tensor representing the input data to be embedded.
            * training: a boolean indicating whether the model is currently in training mode (default is False).

        Returns:
            * A tensor representing the embedded input data in the latent space of the autoencoder.
        '''
        return self.encoder(x, training=training)

    def call(self, x):
        '''
        Applies the encoder and decoder components of the autoencoder

        Arguments:
            * x: a tensor representing the input data to be fed into the autoencoder.

        Returns:
            * A tensor representing the reconstructed data output by the autoencoder.
        '''
        encoded = self.encoder(x)
        y_pred = self.decoder(encoded)
        return y_pred


class Autoencoder(SimpleAutoencoder):
    def call(self, x):
        '''
        Applies the encoder and decoder components of the autoencoder

        Arguments:
            * x: a tensor representing the input data to be fed into the autoencoder.

        Returns:
            * A tensor representing the reconstructed data output by the autoencoder.
        '''
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def get_layer_output_shape(layer):
    ''' 
    gets layer output
    
    Arguments:
        * layer: layer from whcich te output is returned

    Returns:
        * layer's output
    '''
    return layer.output.shape[1:]


class DLMM:
    def __init__(self, model, batch_size, es_patience, rop_patience, all_classes, rop_factor=0.9,
                 best_model_filepath='../models', use_true_labels=True, gamma=0.1, now=None):
        '''
        Creates an instance of the DLMM class with specified parameters.

        Arguments:
            * model: a TensorFlow model.
            * batch_size: an integer representing the number of samples in each batch for training and evaluation.
            * es_patience: an integer representing the number of epochs to wait before early stopping if there is no improvement in validation loss.
            * rop_patience: an integer representing the number of epochs to wait before reducing the learning rate if there is no improvement in validation loss.
            * all_classes: a list of all possible classes in the dataset.
            * rop_factor: a float representing the factor to reduce the learning rate by when the validation loss stops improving.
            * best_model_filepath: a string representing the path to the directory where the best model will be saved.
            * use_true_labels: a boolean indicating whether to use the true labels or predicted labels for the loss calculation.
            * gamma: a float representing the weight given to the classification loss.
            * now: a string representing the current time in the format of "YYYY-MM-DD_HH-MM-SS". If None, the current time will be used.
        
        Returns:
            * An instance of the DLMM class.
        '''
        self.batch_history = None
        self.model, self.batch_size, self.es_patience, self.rop_patience = model, batch_size, es_patience, rop_patience
        self.rop_factor = rop_factor
        self.all_classes = all_classes
        self.use_true_labels = use_true_labels
        self.gamma = gamma
        self.history = {
            'loss': [],
            'val_loss': [],
            'MSE': [],
            'val_MSE': [],
            'cat_CE': [],
            'val_cat_CE': []
        }
        if now is None:
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        else:
            current_time = now
        self.best_model_filename = os.path.join(best_model_filepath, current_time)
        train_log_dir = 'logs/' + current_time + '/train'
        val_log_dir = 'logs/' + current_time + '/val'
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    def log_to_tensorboard(self, epoch):
        '''
        Logs the training and validation metrics to TensorBoard.

        Arguments:
            * epoch: an integer representing the current epoch number.

        Returns:
            * None
        '''
        # Inside your training loop, after each epoch or batch, log the values
        with self.train_summary_writer.as_default():
            for metric, values in self.history.items():
                if 'val' not in metric:
                    tf.summary.scalar(metric, values[-1], step=epoch)

        with self.val_summary_writer.as_default():
            for metric, values in self.history.items():
                if 'val' in metric:
                    tf.summary.scalar(metric.replace('val_', ''), values[-1], step=epoch)

    def check_earlystopping(self):
        '''
        Checks for early stopping by comparing the validation loss of the current epoch with the previous best validation loss.

        Returns:
            * is_stop: a boolean indicating whether early stopping criteria has been met.
        '''
        best_epoch_loss = 999999
        best_num_epoch = 0
        for num_epoch, epoch_loss in enumerate(self.history['val_loss']):
            if epoch_loss < best_epoch_loss - 0.001:
                best_epoch_loss = epoch_loss
                best_num_epoch = num_epoch
        epochs_without_improvement = len(self.history['val_loss']) - 1 - best_num_epoch

        is_stop = epochs_without_improvement == self.es_patience
        if is_stop:
            print("Earlystopping")
        else:
            print(f'{self.es_patience - epochs_without_improvement} epochs without val_loss improvement left to ES.')
        return is_stop

    def check_reduce_lr_on_plateau(self):
        '''
        Checks if there has been no improvement in validation loss for a number of epochs equal to rop_patience.
        If so, reduces the learning rate of the optimizer by rop_factor and prints a message.
        If not, prints the number of epochs left before reducing the learning rate.
        '''
        epochs_without_improvement = len(self.history['val_loss']) - 1 - np.argmin(self.history['val_loss'])
        reduce = epochs_without_improvement == self.rop_patience
        if reduce:
            print("Reducing lr")
            self.model.optimizer.lr.assign(self.model.optimizer.lr * self.rop_factor)
        else:
            print(
                f'{self.rop_patience - epochs_without_improvement} epochs without val_loss improvement left to reduce LR.')

    def check_best_model(self):
        ''' 
        Check if the current model has the best validation loss, and if so, save it to disk.
        '''
        loss_reversed = self.history['val_loss'][::-1]
        idx = len(loss_reversed) - np.argmin(loss_reversed) - 1

        if len(self.history['val_loss']) == idx + 1:
            print(f"Saved best model in {self.best_model_filename}")
            self.model.encoder.save(os.path.join(self.best_model_filename, 'my_model_en.h5'))
            self.model.decoder.save(os.path.join(self.best_model_filename, 'my_model_de.h5'))

    def load_best_model(self):
        '''
        Loads the best saved model from the specified file path.
        '''
        self.model.encoder = tf.keras.models.load_model(os.path.join(self.best_model_filename, 'my_model_en.h5'))
        self.model.decoder = tf.keras.models.load_model(os.path.join(self.best_model_filename, 'my_model_de.h5'))

    def fit(self, X_train, Y_train, X_val, Y_val, epochs):
        '''
        Trains the model for a specified number of epochs on the training data and evaluates on the validation data.

        Arguments:
            * X_train: a numpy array of training samples.
            * Y_train: a numpy array of training labels.
            * X_val: a numpy array of validation samples.
            * Y_val: a numpy array of validation labels.
            * epochs: an integer representing the number of epochs to train the model for.

        '''
        self.num_steps = len(X_train) // self.batch_size

        for epoch_num in range(epochs):
            self.batch_history = {
                'loss': tf.zeros(self.num_steps),
                'val_loss': tf.zeros(self.num_steps),
                'MSE': tf.zeros(self.num_steps),
                'val_MSE': tf.zeros(self.num_steps),
                'cat_CE': tf.zeros(self.num_steps),
                'val_cat_CE': tf.zeros(self.num_steps),
            }

            for step in trange(self.num_steps):
                self.train_step(X_train, Y_train, X_val, Y_val, step)

            for metric in self.history:
                self.history[metric].append(tf.math.reduce_mean(self.batch_history[metric]).numpy())

            self.log_to_tensorboard(epoch_num)
            print(f'loss: {round(self.history["loss"][-1], 4)}, val_loss: {round(self.history["val_loss"][-1], 4)}')

            self.batch_history = {
                'loss': tf.zeros(self.num_steps),
                'val_loss': tf.zeros(self.num_steps),
                'MSE': tf.zeros(self.num_steps),
                'val_MSE': tf.zeros(self.num_steps),
                'cat_CE': tf.zeros(self.num_steps),
                'val_cat_CE': tf.zeros(self.num_steps),
            }

            if self.check_earlystopping():
                self.load_best_model()
                break
            self.check_reduce_lr_on_plateau()
            self.check_best_model()

    @staticmethod
    def normalize_L2(x):
        """
        Normalizes a given tensor by its L2 norm.

        Arguments:
            * x: a TensorFlow tensor.

        Returns:
            * The normalized tensor.
        """
        norm = tf.linalg.norm(x, ord=2, axis=1, keepdims=True)
        return x / norm

    @staticmethod
    def return_similarities(E_1, E_2):
        '''
        Returns the softmax similarity matrix between the embeddings E_1 and E_2.

        Arguments:
            * E_1: a tensor containing embeddings of shape (num_samples_1, embedding_dim).
            * E_2: a tensor containing embeddings of shape (num_samples_2, embedding_dim).

        Returns:
            * A tensor of shape (num_samples_1, num_samples_2) containing the softmax similarity scores between the embeddings in E_1 and E_2.
        '''

        E_1 = DLMM.normalize_L2(E_1)
        E_2 = DLMM.normalize_L2(E_2)

        similarity_matrix = tf.einsum("ae,pe->ap", E_1, E_2)
        softmax_similarity_matrix = tf.nn.softmax(similarity_matrix, axis=1)

        return softmax_similarity_matrix

    @tf.function
    def return_anchors(self, X_train, Y_train):
        '''
        Returns a tuple of anchor samples and their corresponding positive samples.

        Arguments:
            * X_train: a tensor of shape (num_samples, sample_dim) representing the training set samples.
            * Y_train: a tensor of shape (num_samples,) representing the corresponding training set labels.

        Returns:
            * anchor_samples: a tensor of shape (batch_size, sample_dim) containing the randomly selected anchor samples.
            * anchor2_samples: a tensor of shape (batch_size, sample_dim) containing the positive samples corresponding to the selected anchor samples. 
            The positive samples are chosen from the same class as the anchor samples.
        '''
        anchor_indices = tf.cast(tf.random.uniform((self.batch_size,), 0, len(Y_train)), dtype=tf.int32)
        anchor_classes = tf.cast(tf.gather(Y_train, anchor_indices), dtype=tf.int32)
        anchor_samples = tf.gather(X_train, anchor_indices)

        anchor2_samples = tf.zeros_like(anchor_samples)
        for class_ in self.all_classes:
            mask = anchor_classes == class_
            mask_sum = tf.cast(tf.math.reduce_sum(tf.cast(mask, dtype=tf.uint8)), dtype=tf.int32)
            indices_of_class = tf.where(tf.cast(Y_train, dtype=tf.int32) == class_)

            indices_of_indices_of_samples = tf.cast(tf.random.uniform((mask_sum,), 0.0, indices_of_class.shape[0]),
                                                    dtype=tf.int32)
            indices_of_samples = tf.reshape(tf.gather(indices_of_class, indices_of_indices_of_samples), (-1,))
            samples = tf.gather(X_train, indices_of_samples)

            indices_to_replace = tf.where(mask)
            anchor2_samples = tf.tensor_scatter_nd_update(anchor2_samples, indices_to_replace, samples)
        return anchor_samples, anchor2_samples

    @tf.function
    def return_anchors_without_classes(self, X_train):
        '''
        Returns a pair of anchor samples randomly selected from the training set.

        Arguments:
            * X_train: a tensor containing training samples of shape (num_samples, input_shape).

        Returns:
            * A tuple containing two tensors of shape (batch_size, input_shape), each containing randomly selected anchor samples from X_train.
        '''
        indices = tf.cast(tf.random.uniform((self.batch_size,), 0, len(X_train)), dtype=tf.int32)
        samples = tf.gather(X_train, indices)
        return samples, samples

    def calculate_losses(self, X, Y, similarities):
        '''
        Calculates the mean squared error and cross-entropy losses for the current batch.

        Arguments:
            * X: a tensor containing the original data of shape (batch_size, num_features).
            * Y: a tensor containing the reconstructed data of shape (batch_size, num_features).
            * similarities: a tensor of shape (batch_size, batch_size) containing the similarity scores between pairs of samples in X and Y.

        Returns:
            * A list of two tensors: the mean squared error loss and the cross-entropy loss.

        '''
        mask = tf.cast(tf.equal(X, 0), tf.float32)
        # Calculate the mean squared error, applying the mask and multiplying by gamma

        X = tf.cast(X, tf.float32)
        mse = tf.reduce_mean(tf.square(X - Y) * (1 - mask) + self.gamma * tf.square(X - Y) * mask)

        ones = tf.range(len(similarities))

        cc = tf.keras.metrics.sparse_categorical_crossentropy(ones, similarities)
        cc = tf.reduce_mean(cc)

        return [mse, cc]

    def update_metric(self, metric_name, metric_value, step):
        '''
        Update the metric history with the given metric name, value, and step.

        Arguments:
            * metric_name: a string representing the name of the metric.
            * metric_value: a tensor representing the value of the metric.
            * step: an integer representing the current step.
        '''
        self.batch_history[metric_name] = tf.tensor_scatter_nd_update(self.batch_history[metric_name],
                                                                      tf.where(
                                                                          tf.range(self.num_steps + 1) == step),
                                                                      tf.convert_to_tensor([metric_value]))

    def train_step(self, X_train, Y_train, X_val, Y_val, step):
        '''
        Performs a single training step on the model using the given batch of training data.

        rguments:
            * X_train: a tensor containing the input training data of shape (batch_size, num_features).
            * Y_train: a tensor containing the target training data of shape (batch_size, num_features).
            * X_val: a tensor containing the input validation data of shape (batch_size, num_features).
            * Y_val: a tensor containing the target validation data of shape (batch_size, num_features).
            * step: the current training step.
        '''
        if self.use_true_labels:
            anchors, positives = self.return_anchors(X_train, Y_train)
        else:
            anchors, positives = self.return_anchors_without_classes(X_train)

        with tf.GradientTape() as tape:
            anchors_embeddings, anchors_pred = self.model(anchors)
            positives_embeddings = self.model.get_embeddings(positives, training=True)
            similarities = DLMM.return_similarities(anchors_embeddings, positives_embeddings)
            losses = self.calculate_losses(anchors, anchors_pred, similarities)
            loss = losses[0] + losses[1]

        # Calculate gradients and apply via optimizer.
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Update and return metrics (specifically the one for the loss value).

        self.update_metric('loss', loss, step)
        self.update_metric('MSE', losses[0], step)
        self.update_metric('cat_CE', losses[1], step)

        if self.use_true_labels:
            anchors, positives = self.return_anchors(X_val, Y_val)
        else:
            anchors, positives = self.return_anchors_without_classes(X_val)

        anchors_embeddings, anchors_pred = self.model(anchors)
        positives_embeddings = self.model.get_embeddings(positives, training=False)
        similarities = DLMM.return_similarities(anchors_embeddings, positives_embeddings)

        losses = self.calculate_losses(anchors, anchors_pred, similarities)
        loss = losses[0] + losses[1]

        self.update_metric('val_loss', loss, step)
        self.update_metric('val_MSE', losses[0], step)
        self.update_metric('val_cat_CE', losses[1], step)

    def get_embeddings(self, x):
        '''
        Returns the embeddings of the given input tensor computed by the trained model.

        Arguments:
             x: a tensor of shape (batch_size, num_features) containing the input data.

        Returns:
            * A tensor of shape (batch_size, embedding_dim) containing the embeddings of the input tensor computed by the trained model.
        '''
        return self.model.get_embeddings(x)

