import time
from glob import glob
from pathlib import Path

import click
import matplotlib.pyplot as plt
import neptune.new as neptune
import numpy as np
import tensorflow as tf
# from neptune.new.types import File
from neptune.new.integrations.tensorflow_keras import NeptuneCallback
from neptune.new.types import File
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import (LearningRateScheduler, ModelCheckpoint,
                                        TensorBoard)
from tensorflow.keras.models import load_model

try:
    from tensorflow.keras.optimizers.legacy import Adam
except:
    from tensorflow.keras.optimizers import Adam

from tensorflow.keras.preprocessing.image import \
    ImageDataGenerator as defaultImageDataGenerator
from tensorflow.keras.utils import custom_object_scope
from tensorflow.python.client import device_lib

from src.keras_patch_generator.customImageDataGenerator import \
    customImageDataGenerator
from src.unet7 import Unet
from src.utils.config import read_json_config
from src.utils.io import (image_mask_generator_from_directory,
                          image_mask_generator_from_npz)
from src.utils.losses import dice_bce_loss, dice_coeff, dice_loss
from src.utils.train import average_performance

# test_imgs/masks_path without '/' at the end

@click.command()
@click.argument('config_file_path', type=click.Path(exists=True))
def main(config_file_path):
    config = read_json_config(config_file_path)

    current_dir = config['project_result_dir']

    if current_dir:
        current_dir = Path(current_dir)
    else:
        current_dir = Path(__file__).parent

    if config['test']['from_npz']:
        raise NotImplementedError('Reading test from npz is currently not supported')

    global NEPTUNE
    NEPTUNE = config['use_neptune']

    global SCHEDULER
    SCHEDULER = config['train']['use_scheduler']

    if NEPTUNE:
        run = neptune.init(**config['neptune_args'])
        run["config.json"].upload(config_file_path)
    else:
        run = None

    # code to check for GPU
    print("Num CPUs Available: ", len(
        tf.config.list_physical_devices('CPU')))
    print("Num GPUs Available: ", len(
        tf.config.list_physical_devices('GPU')))
    print(device_lib.list_local_devices())

    # TF dimension ordering in this code
    K.set_image_data_format('channels_last')

    parameters = config['parameters']

    seed = parameters['seed']
    img_cols = parameters['img_cols']
    img_rows = parameters['img_rows']
    epochs = parameters['epochs']
    steps_per_epoch = parameters['steps_per_epoch']
    smooth = float(parameters['smooth'])
    start_lr = parameters['start_lr']
    target_width = parameters['target_width']
    target_height = parameters['target_height']
    thresh_obj_perc = parameters['thresh_obj_perc']
    max_iter = parameters['max_iter']
    val_freq = parameters['val_freq']
    val_steps = 3030//config['validation']['batch_size']
    upscale_factor_width = img_cols / target_width
    upscale_factor_height = img_rows / target_height
    resampling_number = parameters['resampling_number']

    parameters.update({
        "upscale_factor_width": upscale_factor_width,
        "upscale_factor_height": upscale_factor_height,
    })

    if NEPTUNE:
        run["model/parameters"] = parameters

    runningTime = time.strftime('%b-%d-%Y_%H-%M')
    model_dir = current_dir / 'model'
    log_dir = model_dir / 'logs'/ f'{runningTime}'
    log_dir.mkdir(parents=True, exist_ok=True)


    if NEPTUNE:
        run["model/parameters/runningTime"] = runningTime
        run["model/parameters/model_dir"] = model_dir
        run["model/parameters/log_dir"] = log_dir

    # Training
    tensorboard = TensorBoard(
        log_dir=model_dir / f'{runningTime}',
        histogram_freq=1,
    )
    # tensorboard = TensorBoard(log_dir=log_dir)

    filepath = 'model.epoch{epoch:03d}.hdf5'
    checkpoint = ModelCheckpoint(filepath=log_dir / filepath,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min')

    def scheduler(epoch, lr):
        if (epoch > parameters['scheduler']['start_after']
            and epoch % parameters['scheduler']['step_size'] == 0):
            return lr * parameters['scheduler']['gamma']
        else:
            return lr

    callbacksList = [tensorboard, checkpoint]

    if SCHEDULER:
        scheduler = LearningRateScheduler(scheduler)
        callbacksList.append(scheduler)

    if NEPTUNE:
        callbacksList.append(NeptuneCallback(run=run))

    if config['checkpoint_file_path'] is None:
        model = Unet(
            target_height,
            target_width,
            filters=parameters['bottleneck_size'],
            nclasses=1,
            do=parameters['dropout'],
            l2=parameters['l2_regularization'],
            )
        model.compile(optimizer=Adam(learning_rate=start_lr),
                    loss=dice_bce_loss, metrics=[dice_coeff])
    else:
        with custom_object_scope({'dice_bce_loss': dice_bce_loss, 'dice_coeff': dice_coeff}):
            model = load_model(current_dir / config['checkpoint_file_path'])

        first_few_layers = parameters['lys_to_freeze'] #number of the first few layers to freeze

        if parameters['lys_to_freeze'] is not None:
            for layer in model.layers[:first_few_layers]:
                layer.trainable = False

            model.compile(optimizer=Adam(learning_rate=start_lr),
                              loss=dice_bce_loss, metrics=[dice_coeff])
    for l in model.layers:
        print(l.name, l.trainable)
    model.summary()
    print(f'start tensorboard, cmd: tensorboard --logdir="{log_dir}"')

    # Data augmentation of training images and masks

    # sets defining what which transformations are applicable to which
    image_tf = config['image_tf']
    mask_tf = config['mask_tf']

    img_mask_gen_args_train = config['train']['img_mask_gen_args']

    if config['train']['from_npz']:
        image_mask_datagen_train = defaultImageDataGenerator(
            **img_mask_gen_args_train)
        train_data_path = Path(config['train']['data_path'])
        image_mask_generator_train = image_mask_generator_from_npz(
            str(train_data_path),
            image_mask_datagen_train,
            config['train']['batch_size'],
            seed,
        )
    else:
        train_img_dir = Path(config['train']['img_dir'])
        train_mask_dir = Path(config['train']['mask_dir'])
        image_mask_generator_train = image_mask_generator_from_directory(
            img_mask_gen_args_train, max_iter, thresh_obj_perc, target_width, target_height, image_tf,
            mask_tf, train_img_dir, train_mask_dir, img_rows, img_cols, config['train']['batch_size'], seed,
            resampling_number, config['train']['fold_resolution']
        )

    img_mask_gen_args_val = config['validation']['img_mask_gen_args']

    if config['validation']['from_npz']:
        image_mask_datagen_val = defaultImageDataGenerator(
            **img_mask_gen_args_val)
        val_data_path = Path(config['validation']['data_path'])
        image_mask_generator_val = image_mask_generator_from_npz(
            str(val_data_path),
            image_mask_datagen_val,
            config['validation']['batch_size'],
            seed,
        )
    else:
        val_img_dir = Path(config['validation']['img_dir'])
        val_mask_dir = Path(config['validation']['mask_dir'])
        image_mask_generator_val = image_mask_generator_from_directory(
            img_mask_gen_args_val, max_iter, thresh_obj_perc, target_width, target_height, image_tf,
            mask_tf, val_img_dir, val_mask_dir, img_rows, img_cols, config['validation']['batch_size'], seed,
            resampling_number, config['validation']['fold_resolution']
        )

    img_mask_gen_args_test = config['test']['img_mask_gen_args']

    if config['test']['from_npz']:
        image_mask_datagen_test = defaultImageDataGenerator(
            **img_mask_gen_args_test)
        test_data_path = Path(config['test']['data_path'])
        image_mask_generator_test = image_mask_generator_from_npz(
            str(test_data_path),
            image_mask_datagen_test,
            config['test']['batch_size'],
            seed,
        )
    else:
        test_img_dir = Path(config['test']['img_dir'])
        test_mask_dir = Path(config['test']['mask_dir'])
        image_mask_generator_test = image_mask_generator_from_directory(
            img_mask_gen_args_test, max_iter, thresh_obj_perc, target_width, target_height, image_tf,
            mask_tf, test_img_dir, test_mask_dir, img_rows, img_cols, config['test']['batch_size'], seed,
            resampling_number, config['test']['fold_resolution']
        )

    print(tf.shape(image_mask_generator_train.next()))
    print(tf.shape(image_mask_generator_val.next()))
    # print(tf.shape(image_mask_generator_test.next()))

    # Add additional parameters
    if NEPTUNE:
        # 10000//batch_size
        run["model/parameters/steps_per_epoch"] = steps_per_epoch

    history = model.fit(
        image_mask_generator_train,
        # image_mask_generator,
        batch_size=config['train']['batch_size'],
        # steps_per_epoch= steps_per_epoch, #10000//batch_size, 50, 200, 40
        # steps_per_epoch=10000 // batch_size, #correct value
        epochs=epochs,
        validation_freq=val_freq,
        validation_data=image_mask_generator_val,  # image_mask_generator_val,
        # validation_steps=val_steps, #100//batch_size_val,
        callbacks=callbacksList)

    summarise_history(history, 'dice_coeff', log_dir, run)
    summarise_history(history, 'loss', log_dir, run)

    find_best_model(log_dir, config, img_cols, img_rows, run)

    if NEPTUNE:
        run.stop()


def summarise_history(history, attribute: str, log_dir: str, run):
    plt.figure()
    plt.plot(history.history[attribute])
    plt.plot(history.history[f'val_{attribute}'])
    plt.title(f'Manual labels training {attribute}')
    plt.ylabel(attribute)
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(log_dir / f'{attribute}.svg', bbox_inches='tight', dpi=300)
    if NEPTUNE:
        run[f"model/evaluation/{attribute}_plot"].upload(str(log_dir / f'{attribute}.svg'))


def find_best_model(log_dir: str, config: dict, img_cols: int, img_rows: int, run):
    checkpoints = glob(str(log_dir / '*.hdf5'))
    ckp_nr = []
    ckp_str = []
    substr = '.hdf5'
    for ckp in checkpoints:
        index = ckp.find(substr)
        ckp_nr.append(int(ckp[index-3:index]))
        ckp_str.append(ckp[index-3:index])

    ckp_nr, ckp_str = zip(*sorted(zip(ckp_nr, ckp_str)))

    best_model = log_dir / f'model.epoch{ckp_str[-1]}.hdf5'

    custom_objects = {
        "dice_loss": dice_loss,
        "dice_bce_loss": dice_bce_loss,
        "dice_coeff": dice_coeff
    }
    reloaded_model = load_model(best_model, custom_objects=custom_objects)
    reloaded_model.summary()

    img_mask_args_avg_test_per = config['img_mask_args_avg_test_per']
    img_mask_args_avg_test_per.update({
        "target_width": img_cols,
        "target_height": img_rows,
    })

    image_mask_datagen_avg_test_per = customImageDataGenerator(
        **img_mask_args_avg_test_per)

    # Code to get patches of all the test images and then calculate the average
    # dice coeff between them and their predicted counterparts

    parameters = config['parameters']
    avg_performance = average_performance(
        test_imgs_path=Path(config['test']['img_dir']),
        test_masks_path=Path(config['test']['mask_dir']),
        model=reloaded_model,
        log_dir=log_dir,
        img_rows=parameters['img_rows'],
        img_cols=parameters['img_cols'],
        target_height=parameters['target_height'],
        target_width=parameters['target_width'],
        image_mask_datagen_avg_test_per=image_mask_datagen_avg_test_per,
        fold_resolution=config['test']['fold_resolution']
    )

    if NEPTUNE:
        run["model/evaluation/sample_pred"].upload(str(log_dir / 'sample_ground_truth_test.svg'))

    metrics = {"avg_dice_coeff": avg_performance}
    print(metrics)
    if NEPTUNE:
        run["model/evaluation/metrics"] = metrics


if __name__ == "__main__":
    main()
