import tensorflow as tf
from collections.abc import Iterable


def return_architecture(architecture_type, architecture_kwargs):
    if architecture_type == 'simpleNN':
        return simple_model(**architecture_kwargs)
    elif architecture_type == 'residualNN':
        return residual_model(**architecture_kwargs)
    elif architecture_type == 'VGG':
        return VGG_model(**architecture_kwargs)
    else:
        raise ValueError(f'Unknown architecture type: {architecture_type}')


def simple_model(input_shape, n_filters, kernel_size, dropout_rate, use_batch_normalization):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=n_filters,
                                     kernel_size=(kernel_size, kernel_size),
                                     activation='relu',
                                     input_shape=input_shape))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    if use_batch_normalization:
        model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(filters=n_filters,
                                     kernel_size=(kernel_size, kernel_size),
                                     activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    if use_batch_normalization:
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Conv2D(filters=n_filters,
                                     kernel_size=(kernel_size, kernel_size),
                                     activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    if use_batch_normalization:
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    if use_batch_normalization:
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(64, activation='relu'))

    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    return model


def residual_model(input_shape, n_blocks, n_filters, kernel_size, dropout_rate, use_batch_normalization, conv_head):
    if not isinstance(dropout_rate, Iterable): 
        dropout_rate = [dropout_rate]*3
    if not isinstance(use_batch_normalization, Iterable): 
        use_batch_normalization = [use_batch_normalization]*3
        
    def residual_block(input_, i):
        conv_2d = tf.keras.layers.Conv2D(n_filters, (kernel_size, kernel_size), padding='same',
                                         name=f'residual_block_conv2d_{i}_1')(input_)
        bn = tf.keras.layers.BatchNormalization(name=f'residual_block_bn_{i}_1')(conv_2d)
        relu = tf.keras.layers.ReLU(name=f'residual_block_relu_{i}_1')(bn)
        conv_2d = tf.keras.layers.Conv2D(n_filters, (kernel_size, kernel_size), padding='same',
                                         name=f'residual_block_conv2d_{i}_2')(relu)
        bn = tf.keras.layers.BatchNormalization(name=f'residual_block_bn_{i}_2')(conv_2d)
        skip_connection = tf.keras.layers.Add(name=f'residual_block_skip_conn_{i}')([bn, input_])
        relu = tf.keras.layers.ReLU(name=f'residual_block_relu_{i}_2')(skip_connection)
        return relu

    #  rectified batch-normalized convolutional layer
    input_ = tf.keras.Input(shape=input_shape, name='input_layer')
    conv_2d = tf.keras.layers.Conv2D(n_filters, (kernel_size, kernel_size), padding='same', name='first_conv')(input_)
    bn = tf.keras.layers.BatchNormalization(name='first_bn')(conv_2d)
    relu = tf.keras.layers.ReLU(name='first_relu')(bn)
    # 19 residual blocks
    rl = residual_block(relu, 0)
    for i in range(n_blocks):
        rl = residual_block(rl, i + 1)


    if conv_head:
        for i in range(len(use_batch_normalization) - 1):
            if i == 0:
                x = tf.keras.layers.Conv2D(n_filters, (kernel_size, kernel_size), padding='valid')(rl)
            else: 
                x = tf.keras.layers.Conv2D(n_filters, (kernel_size, kernel_size), padding='valid')(x)

            if use_batch_normalization[i]:
                x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)

        output = tf.keras.layers.Dense(10, activation='softmax')(x)
        model = tf.keras.Model(inputs=input_, outputs=output)
    else: 
        x = tf.keras.layers.Flatten()(rl)
        x = tf.keras.layers.Dropout(dropout_rate[0])(x)
        if use_batch_normalization[0]:
            x = tf.keras.layers.BatchNormalization()(x)
        
        
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(dropout_rate[1])(x)
        if use_batch_normalization[1]:
            x = tf.keras.layers.BatchNormalization()(x)
        
        
        
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(dropout_rate[2])(x)
        if use_batch_normalization[2]:
            x = tf.keras.layers.BatchNormalization()(x)
        
        output = tf.keras.layers.Dense(10, activation='softmax')(x)
        model = tf.keras.Model(inputs=input_, outputs=output)
    return model


def VGG_model(input_shape, use_conv2dtranspose, dropout_rate, use_batch_normalization, conv_head, train_full):
    if not isinstance(dropout_rate, Iterable): 
        dropout_rate = [dropout_rate]*3
    if not isinstance(use_batch_normalization, Iterable): 
        use_batch_normalization = [use_batch_normalization]*3
        
    vgg16 = tf.keras.applications.vgg16.VGG16(include_top=not train_full)
    my_vgg16 = tf.keras.models.Sequential()
    if not train_full:
        if use_conv2dtranspose:
            my_vgg16.add(tf.keras.layers.Conv2DTranspose(filters=3,
                                                         kernel_size=(3, 3), input_shape=input_shape))
            while my_vgg16.output.shape.as_list() != vgg16.input.shape.as_list():
                my_vgg16.add(tf.keras.layers.Conv2DTranspose(filters=3,
                                                             kernel_size=(3, 3)))
        else:
            my_vgg16.add(tf.keras.layers.UpSampling2D(size=(7, 7), input_shape=input_shape))
            
    layers_offset = 3 if not conv_head else 4 # we take flatten or dont 
    if train_full: 
        layers_offset = 0
    for i in range(1, len(vgg16.layers) - layers_offset):
        my_layer = vgg16.layers[i]
        if not train_full:
            my_layer.trainable = False
        my_vgg16.add(my_layer)
    
    if conv_head:
        for i in range(len(use_batch_normalization) - 1):
            if i == 0:
                my_vgg16.add(tf.keras.layers.Conv2D(256, (3, 3), padding='valid'))
            else :
                my_vgg16.add(tf.keras.layers.Conv2D(128, (3, 3), padding='valid'))
            if use_batch_normalization[0]:
                my_vgg16.add(tf.keras.layers.BatchNormalization())
        my_vgg16.add(tf.keras.layers.Flatten())
        my_vgg16.add(tf.keras.layers.Dense(128, activation='relu'))

    else: 
        my_vgg16.add(tf.keras.layers.Dropout(dropout_rate[0]))
        if use_batch_normalization[0]:
            my_vgg16.add(tf.keras.layers.BatchNormalization())
            
        my_vgg16.add(tf.keras.layers.Dense(1024, activation='relu'))
        
        my_vgg16.add(tf.keras.layers.Dropout(dropout_rate[1]))
        if use_batch_normalization[1]:
            my_vgg16.add(tf.keras.layers.BatchNormalization())
        
        my_vgg16.add(tf.keras.layers.Dense(256, activation='relu'))
        
        my_vgg16.add(tf.keras.layers.Dropout(dropout_rate[2]))
        if use_batch_normalization[2]:
            my_vgg16.add(tf.keras.layers.BatchNormalization())
            
    my_vgg16.add(tf.keras.layers.Dense(10, activation='softmax'))
    return my_vgg16
