from pathlib import Path

import click
import tensorflow as tf

from src.keras_patch_generator.customImageDataGenerator import \
    customImageDataGenerator
from src.utils.config import read_json_config
from src.utils.train import get_img_array, load_image_mask, predict
from src.utils.visualization import visualise
from src.utils.model import load_unet_weights


@click.command()
@click.argument('config_file_path', type=click.Path(exists=True))
@click.argument('checkpoint_file_path', type=click.Path(exists=True))
@click.argument('img_path', type=click.Path(exists=True))
@click.argument('msk_path', type=click.Path(exists=True))
def main(config_file_path, checkpoint_file_path, img_path, msk_path):
    print("Num CPUs Available: ", len(
        tf.config.list_physical_devices('CPU')))
    print("Num GPUs Available: ", len(
        tf.config.list_physical_devices('GPU')))

    config = read_json_config(config_file_path)
    parameters = config['parameters']

    model = load_unet_weights(parameters, checkpoint_file_path)

    model_name = Path(checkpoint_file_path).name
    save_dir = Path(f'./plots/{model_name}')
    save_dir.mkdir(parents=True, exist_ok=True)

    img_mask_args_avg_test_per = config['img_mask_args_avg_test_per']
    img_mask_args_avg_test_per.update({
        "target_width": parameters['img_cols'],
        "target_height": parameters['img_rows'],
    })

    image_mask_datagen_avg_test_per = customImageDataGenerator(
        **img_mask_args_avg_test_per
    )

    # Code to get patches of all the test images and then calculate the average
    # dice coeff between them and their predicted counterparts

    image_tf = config['image_tf']
    mask_tf = config['mask_tf']

    x, y = load_image_mask(
        img_path,
        msk_path,
        image_tf,
        mask_tf,
        image_mask_datagen_avg_test_per,
        fold_resolution=0.5,
    )

    parameters = config['parameters']
    img_array = get_img_array(
        x,
        parameters['img_rows'],
        parameters['img_cols'],
        parameters['target_height'],
        parameters['target_width'],
    )

    pred = predict(model, img_array)

    img_name = Path(img_path).stem

    visualise(x, pred, y, save_dir / f'{img_name}.svg')

if __name__ == '__main__':
    main()
