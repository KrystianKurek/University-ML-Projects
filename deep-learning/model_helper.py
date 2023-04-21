import os
import tensorflow as tf
import json


class ModelHelper:
    @staticmethod
    def create_callbacks(models_dir, run_name, early_stopping_kwargs, best_model_kwargs, logs_dir='logs'):
        """
        Creates callbacks determined with bools with options given in "kwargs" arguments what will be used in model

        Parameters
        ----------
        architecture_name : str
            architecture name that will be name of folder with logs and summary
        run_name : str
            name of particular run that will be stored in architecture folder
        tensorboard_callback : bool
            bool determining whether to use tensorboard callback, optional
        early_stopping_callback : bool
            bool determining whether to use early stopping callback, optional
        best_model_callback : bool
            bool determining whether to use best model callback, optional
        tensorboard_kwargs : dict
            kwargs to tensorboard callback, optional
        early_stopping_kwargs : dict
            kwargs to early stopping callback, optional
        best_model_kwargs : dict
            kwargs to best model callback, optional

        Returns
        -------
        callbacks : list
            configured callbacks
        """

        os.makedirs(models_dir, exist_ok=True)

        logs_dir = os.path.join('logs', run_name)
        callbacks = [ tf.keras.callbacks.TensorBoard(log_dir=logs_dir, **{'histogram_freq': 1,
                                                                    'write_graph': True,
                                                                    'write_images': True,
                                                                    'profile_batch': 0})]
        

        os.makedirs(models_dir, exist_ok=True)

        #callbacks = []

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(models_dir, run_name + best_model_kwargs['filename']),
            **best_model_kwargs)
        callbacks.append(checkpoint)

        if early_stopping_kwargs is None:
            early_stopping_kwargs = {'monitor': 'val_loss', 'mode': 'min', 'verbose': 1, 'patience': 10}
        es = tf.keras.callbacks.EarlyStopping(**early_stopping_kwargs)
        callbacks.append(es)

        return callbacks

    @staticmethod
    def save_history(history, summary_dir, run_name, params):
        to_save = {'params': params, 'history': history.history}
        os.makedirs(summary_dir, exist_ok=True)
        with open(os.path.join(summary_dir, f'{run_name}.json'), 'w') as f:
            json.dump(to_save, f)

    @staticmethod
    @tf.function
    def get_box(lambda_value):
        # taken from https://keras.io/examples/vision/cutmix/
        IMG_SIZE = 32
        cut_rat = tf.math.sqrt(1.0 - lambda_value)

        cut_w = IMG_SIZE * cut_rat  # rw
        cut_w = tf.cast(cut_w, tf.int32)

        cut_h = IMG_SIZE * cut_rat  # rh
        cut_h = tf.cast(cut_h, tf.int32)

        cut_x = tf.random.uniform((1,), minval=0, maxval=IMG_SIZE, dtype=tf.int32)[0]  # rx
        cut_y = tf.random.uniform((1,), minval=0, maxval=IMG_SIZE, dtype=tf.int32)[0]  # ry

        boundary_x1 = tf.clip_by_value(cut_x - cut_w // 2, 0, IMG_SIZE)
        boundary_y1 = tf.clip_by_value(cut_y - cut_h // 2, 0, IMG_SIZE)
        bbx2 = tf.clip_by_value(cut_x + cut_w // 2, 0, IMG_SIZE)
        bby2 = tf.clip_by_value(cut_y + cut_h // 2, 0, IMG_SIZE)

        target_h = bby2 - boundary_y1
        if target_h == 0:
            target_h += 1

        target_w = bbx2 - boundary_x1
        if target_w == 0:
            target_w += 1

        return boundary_x1, boundary_y1, target_h, target_w

    @staticmethod
    @tf.function
    def cutmix(train_ds_one, train_ds_two):
        IMG_SIZE = 32
        (image1, label1), (image2, label2) = train_ds_one, train_ds_two

        # Get a sample from the Beta distribution
        lambda_value = tf.random.uniform(shape=(1,), minval=0, maxval=1)[0]


        # Get the bounding box offsets, heights and widths
        boundaryx1, boundaryy1, target_h, target_w = ModelHelper.get_box(lambda_value)

        # Get a patch from the second image (`image2`)
        crop2 = tf.image.crop_to_bounding_box(
            image2, boundaryy1, boundaryx1, target_h, target_w
        )
        # Pad the `image2` patch (`crop2`) with the same offset
        image2 = tf.image.pad_to_bounding_box(
            crop2, boundaryy1, boundaryx1, IMG_SIZE, IMG_SIZE
        )
        # Get a patch from the first image (`image1`)
        crop1 = tf.image.crop_to_bounding_box(
            image1, boundaryy1, boundaryx1, target_h, target_w
        )
        # Pad the `image1` patch (`crop1`) with the same offset
        img1 = tf.image.pad_to_bounding_box(
            crop1, boundaryy1, boundaryx1, IMG_SIZE, IMG_SIZE
        )

        # Modify the first image by subtracting the patch from `image1`
        # (before applying the `image2` patch)
        image1 = image1 - img1
        # Add the modified `image1` and `image2`  together to get the CutMix image
        image = image1 + image2

        # Adjust Lambda in accordance to the pixel ration
        lambda_value = 1 - (target_w * target_h) / (IMG_SIZE * IMG_SIZE)
        lambda_value = tf.cast(lambda_value, tf.float32)

        # Combine the labels of both images
        label = lambda_value * label1 + (1 - lambda_value) * label2
        return image, label

    @staticmethod
    def preprocess_image(x, y, mean, std, n_classes, use_data_aug_simple):
        # add tf.keras.applications.vgg16.preprocess_input(x, data_format=None)

        x = tf.cast(x, dtype=tf.float32)
        y = tf.reshape(tf.one_hot(y, n_classes), shape=(n_classes,))
        if use_data_aug_simple:
            x = ModelHelper.data_aug_simple(x)
        x = (x - mean) / std
        return x, y

    @staticmethod
    def preprocess_image_vgg(x, y, n_classes, use_data_aug_simple):
        x = tf.cast(x, dtype=tf.float32)
        x = tf.keras.applications.vgg16.preprocess_input(x)
        y = tf.reshape(tf.one_hot(y, n_classes), shape=(n_classes,))
        if use_data_aug_simple:
            x = ModelHelper.data_aug_simple(x)
        return x, y

    @staticmethod
    def return_heavy_augmented_data(x, y, batch_size, mean, std, n_classes):
        preprocess_lambda = lambda x, y: ModelHelper.preprocess_image(x, y, mean, std, n_classes,
                                                                      use_data_aug_simple=False)

        dataset_left = (
            tf.data.Dataset.from_tensor_slices((x, y))
                .shuffle(len(x))
                .map(preprocess_lambda, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        )
        dataset_right = (
            tf.data.Dataset.from_tensor_slices((x, y))
                .shuffle(len(x))
                .map(preprocess_lambda, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        )

        dataset = tf.data.Dataset.zip((dataset_left, dataset_right)) \
            .map(ModelHelper.cutmix, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .batch(batch_size, drop_remainder=True)

        return dataset

    @staticmethod
    def return_generator(x, y, batch_size, mean, std, n_classes, data_aug_type='none', use_vgg_preprocessing=False):
        if data_aug_type == 'advanced':
            return ModelHelper.return_heavy_augmented_data(x, y, batch_size, mean, std, n_classes)
        
        if use_vgg_preprocessing:
            preprocess_lambda = lambda x, y: ModelHelper.preprocess_image_vgg(x, y, n_classes,
                                                                          use_data_aug_simple=data_aug_type == 'standard')
        elif not use_vgg_preprocessing:
            preprocess_lambda = lambda x, y: ModelHelper.preprocess_image(x, y, mean, std, n_classes,
                                                                          use_data_aug_simple=data_aug_type == 'standard')
        else:
            raise ValueError(f'Unknown data_aug_type: {data_aug_type}, use_vgg_preprocessing: {use_vgg_preprocessing}')
        
        generator = tf.data.Dataset.from_tensor_slices((x, y)) \
            .shuffle(len(x)) \
            .map(preprocess_lambda, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .batch(batch_size, drop_remainder=True)

        return generator

    @staticmethod
    def return_learning_rate(learning_rate, learning_rate_type, decay_steps=None):
        if learning_rate_type == 'decaying':
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                learning_rate,
                decay_steps=decay_steps,
                decay_rate=0.9,
                staircase=True)
            return lr_schedule
        elif learning_rate_type == 'constant':
            return learning_rate
        else:
            raise ValueError(f'Unknown learning_rate_type: {learning_rate_type}')

    @staticmethod
    def data_aug_simple(x):
        data_aug_type = tf.cast(tf.random.uniform(shape=(1,), minval=0.0, maxval=4), dtype=tf.int32)[0]
        if data_aug_type == 0:
            offset_height = tf.cast(tf.random.uniform(shape=(1,), minval=0.0, maxval=8), dtype=tf.int32)[0]
            offset_width = tf.cast(tf.random.uniform(shape=(1,), minval=0.0, maxval=8), dtype=tf.int32)[0]
            x = tf.image.crop_to_bounding_box(x, offset_height, offset_width, 24, 24)
            x = tf.image.resize(x, (32, 32))
            return x
        elif data_aug_type == 1:
            k = tf.cast(tf.random.uniform(shape=(1,), minval=1, maxval=4), dtype=tf.int32)[0]
            x = tf.image.rot90(x, k)
            return x
        elif data_aug_type == 2:
            alpha = 100.0
            frac = 0.5
            noise = tf.random.gamma(x.shape, alpha)
            x = x * (1 - frac) + noise * frac
            return x
        elif data_aug_type == 3:
            return x
        else:
            return x  # only because tensorflow wants to have return in "else" statement
