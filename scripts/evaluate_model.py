from pathlib import Path

import click
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from src.utils.config import read_json_config
from src.utils.io import image_mask_generator_from_directory
from src.utils.losses import dice_bce_loss, dice_coeff
from src.utils.model import load_unet_weights


def evaluate(model, generator):
    dce = []
    loss = []

    step = 0
    num_steps = generator.samples_x // generator.batch_size

    for images, masks in tqdm(generator):
        # Generator loops infinitely
        if step >= num_steps:
            break

        pred = model.predict(images, verbose=0)
        dce.append(dice_coeff(pred, masks))
        loss.append(dice_bce_loss(pred, masks))

        step += 1

    return np.mean(dce), np.mean(loss)

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

    image_mask_generator = image_mask_generator_from_directory(
        config['img_mask_gen_args'], parameters['max_iter'], parameters['thresh_obj_perc'],
        parameters['target_width'], parameters['target_height'], config['image_tf'],
        config['mask_tf'], img_dir, mask_dir, parameters['img_rows'],
        parameters['img_cols'], parameters['batch_size'], parameters['seed'],
        parameters['resampling_number'], config['fold_resolution']
    )

    dce, loss = evaluate(model, image_mask_generator)

    print(f"dice_coeff = {dce}, loss = {loss}")

if __name__ == '__main__':
    main()
