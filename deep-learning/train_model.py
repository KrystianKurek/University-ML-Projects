import argparse
import os
import tensorflow as tf
import datetime
from tensorflow.keras.models import load_model
import time
tf.random.set_seed(42)
import architectures
from model_helper import ModelHelper

ARCHITECTURES_PARAMS = {
    'simpleNN': ['n_filters', 'kernel_size', 'dropout_rate', 'use_batch_normalization'],
    'residualNN': ['n_blocks', 'n_filters', 'kernel_size', 'dropout_rate', 'use_batch_normalization', 'conv_head'],
    'VGG': ['use_conv2dtranspose', 'dropout_rate', 'use_batch_normalization', 'conv_head', 'train_full']
}


def main(args):
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()
    n_classes = len(set(test_y.flatten()))
    decay_steps = len(train_x) // args['batch_size'] * 25
    mean = train_x.mean()
    std = train_x.std()

    architecture_kwargs = {kwarg: args[kwarg] for kwarg in ARCHITECTURES_PARAMS[args['model_type']]}
    architecture_kwargs['input_shape'] = train_x[0].shape
    model = architectures.return_architecture(args['model_type'], architecture_kwargs)
    model.summary()

    lr = ModelHelper.return_learning_rate(learning_rate=args['learning_rate'],
                                          learning_rate_type=args['learning_rate_type'],
                                          decay_steps=decay_steps)

    model_compile_kwargs = {'optimizer': tf.keras.optimizers.Adam(learning_rate=lr),
                            'loss': tf.keras.losses.CategoricalCrossentropy()}
    model.compile(metrics=['accuracy'], **model_compile_kwargs)

    early_stopping_kwargs = {'monitor': 'val_loss', 'mode': 'min', 'verbose': 1, 'patience': args['patience']}
    best_model_kwargs = {'monitor': 'val_loss', 'verbose': 1, 'mode': 'min', 'save_weights_only': False,
                         'save_best_only': True, 'filename': '_best_model.h5'}
    now = datetime.datetime.now().strftime('_%Y_%m_%d-%H_%M_%S')
    callbacks = ModelHelper.create_callbacks(args['models_dir'],
                                             args['run_name'] + now,
                                             early_stopping_kwargs,
                                             best_model_kwargs)
    train_generator = ModelHelper.return_generator(train_x, train_y, args['batch_size'], mean, std, n_classes,
                                                   data_aug_type=args['data_augmentation_type'],
                                                   use_vgg_preprocessing=args['model_type'] == 'VGG')
    test_generator = ModelHelper.return_generator(test_x, test_y, args['batch_size'], mean, std, n_classes,
                                                  data_aug_type='none',
                                                  use_vgg_preprocessing=args['model_type'] == 'VGG')

    model_fit_kwargs = {'epochs': args['epochs'], 'verbose': 1, 'workers': 8, 'use_multiprocessing': True}
    print(callbacks)
    start = time.time()
    history = model.fit(x=train_generator, validation_data=test_generator,
                        callbacks=callbacks, **model_fit_kwargs)

    end = time.time()
    model = load_model(os.path.join(args['models_dir'], args['run_name'] + now + best_model_kwargs['filename']))

    args['training_time'] = end - start
    args['scored_train'] = [float(elem) for elem in model.predict(train_x).flatten()]
    args['scored_test'] = [float(elem) for elem in model.predict(test_x).flatten()]

    ModelHelper.save_history(history, args['summary_dir'], args['run_name'] + now, args)


if __name__ == '__main__':
    def list_float_type(x):
        try:
            return float(x)
        except ValueError:
            return list(x)

    parser = argparse.ArgumentParser(description='Train CNN on cifar-10.')
    parser.add_argument('run_name', type=str, help='Name of run, used in saving architecture and summary')
    parser.add_argument('model_type', type=str, default='simpleNN', help='Architecture type',
                        choices=['simpleNN', 'residualNN', 'VGG'])

    parser.add_argument('batch_size', type=int, default=32, help='Minibatch size')

    parser.add_argument('learning_rate', type=float, default=0.01,
                        help='Learning rate value or initial learning rate value if learning_rate_type is \'decaying\'')
    parser.add_argument('learning_rate_type', type=str, default='constant', help='Type of learning rate',
                        choices=['constant', 'decaying'])

    parser.add_argument('data_augmentation_type', type=str, default=False, help='Type of data augmentation',
                        choices=['standard', 'none', 'advanced'])

    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=50, help='Number of epochs')

    parser.add_argument('--n_filters', type=int, default=32, help='Number of filters in conv layers')
    parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size in conv layers')
    parser.add_argument('--n_blocks', type=int, default=30, help='Number of blocks in residual network')

    parser.add_argument('--use_batch_normalization', type=list_float_type, default=0.0,
                        help='Flag indicating use of batch normalization')
    parser.add_argument('--dropout_rate', type=list_float_type, default=0.0, help='Rate used in dropout layer')
    parser.add_argument('--use_conv2dtranspose', type=bool, default=True,
                        help='Flag indicating use of conv2dtranspose layer when using VGG architecture',
                        choices=[True, False], nargs='?')
    parser.add_argument('--models_dir', type=str, help='Directory to store best models', default='models')
    parser.add_argument('--summary_dir', type=str, help='Directory to store summaries', default='summary')
    
    args = parser.parse_args()
    args = vars(args)
    print(args)
    main(args)
