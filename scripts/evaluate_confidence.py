import glob
from pathlib import Path

import click
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from src.keras_patch_generator.customImageDataGenerator import \
    customImageDataGenerator
from src.utils.config import read_json_config
from src.utils.io import image_mask_generator_from_directory
from src.utils.losses import dice_bce_loss, dice_coeff
from src.utils.model import load_unet_weights
from src.utils.train import get_img_array, load_image_mask, predict


def average_confidence_binary(
    test_imgs_path,
    test_masks_path,
    model,
    image_mask_datagen_avg_test_per,
    img_rows,
    img_cols,
    target_height,
    target_width,
    fold_resolution=0.5,
    conf_perc = 50
    ):
    test_image_tf = dict(samplewise_center=False,
                    samplewise_std_normalization=False,
                    rescale=True,
                    vertical_flip=False,
                    horizontal_flip=False,
                    preprocessing_function=False,
                    fill_mode=True,
                    max_iter = False,
                    thresh_obj_perc = False,
                    target_width = False,
                    target_height = False
                    )

    test_mask_tf = dict(samplewise_center=False,
                    samplewise_std_normalization=False,
                    vertical_flip=False,
                    horizontal_flip=False,
                    preprocessing_function=True,
                    fill_mode=True,
                    max_iter = False,
                    thresh_obj_perc = False,
                    target_width = False,
                    target_height = False
                    )

    conf_ls = []
    for f in glob.glob(str(test_imgs_path / 'cls' / '*.tif')):
        img_path = f
        msk_path = test_masks_path / 'cls' / f.split('/')[-1].replace('.tif','_Simple Segmentation.tif')
        print(img_path)
        print(msk_path)

        x, y = load_image_mask(
            img_path,
            msk_path,
            test_image_tf,
            test_mask_tf,
            image_mask_datagen_avg_test_per,
            fold_resolution=fold_resolution,
        )

        img_array = get_img_array(x, img_rows, img_cols, target_height, target_width)

        pred = predict(model, img_array)

        conf_ls.append(conf(1-pred,conf_perc))

    print(len(conf_ls))

    return np.mean(conf_ls)

def conf(x: np.array, perc: int = 50) -> float:
    p = np.percentile(x, perc)
    return p

@click.command()
@click.argument('config_file_path', type=click.Path(exists=True))
def main(config_file_path):
    print("Num CPUs Available: ", len(
        tf.config.list_physical_devices('CPU')))
    print("Num GPUs Available: ", len(
        tf.config.list_physical_devices('GPU')))

    config = read_json_config(config_file_path)
    parameters = config['parameters']

    img_dir = Path(config['img_dir'])
    mask_dir = Path(config['mask_dir'])

    model = load_unet_weights(parameters, config['checkpoint_file_path'])


    parameters = config['parameters']

    img_cols = parameters['img_cols']
    img_rows = parameters['img_rows']

    img_mask_args_avg_test_per = config['img_mask_args_avg_test_per']
    img_mask_args_avg_test_per.update({
        "target_width": img_cols,
        "target_height": img_rows,
    })

    image_mask_datagen_avg_test_per = customImageDataGenerator(
        **img_mask_args_avg_test_per)

    avg_conf = average_confidence_binary(
        test_imgs_path=Path(img_dir),
        test_masks_path=Path(mask_dir),
        model=model,
        img_rows=parameters['img_rows'],
        img_cols=parameters['img_cols'],
        target_height=parameters['target_height'],
        target_width=parameters['target_width'],
        image_mask_datagen_avg_test_per=image_mask_datagen_avg_test_per,
        fold_resolution=config['fold_resolution']
    )
    print(f"avg_conf: {avg_conf}")

if __name__ == '__main__':
    main()
