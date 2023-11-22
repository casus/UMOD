from pathlib import Path

import click
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from src.utils.config import read_json_config
from src.utils.io import image_mask_generator_from_directory
from src.utils.losses import (auc, dice_bce_loss, dice_coeff, iou, precision,
                              recall)
from src.utils.model import load_unet_weights


def evaluate(model, generator, thresh):
    dce = []
    loss = []
    iou_ls = []
    precision_ls = []
    recall_ls = []
    auc_ls = []

    step = 0
    num_steps = generator.samples_x // generator.batch_size

    for images, masks in tqdm(generator):
        # Generator loops infinitely
        if step >= num_steps:
            break

        pred = model.predict(images, verbose=0)
        pred_binarized = np.where(pred>thresh, 1.0, 0.0).astype(np.float32)
        
        dce.append(dice_coeff(masks, pred))
        loss.append(dice_bce_loss(masks, pred))
        iou_ls.append(iou(masks, pred_binarized))
        precision_ls.append(precision(masks, pred_binarized))
        recall_ls.append(recall(masks, pred_binarized))
        auc_ls.append(auc(masks, pred))

        step += 1

    return np.mean(dce), np.mean(loss), np.mean(iou_ls), np.mean(precision_ls), np.mean(recall_ls), np.mean(auc_ls)


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

    dce, loss, mean_iou, mean_precision, mean_recall, mean_auc = evaluate(model,
                                                                image_mask_generator,
                                                                parameters['thresh'])

    print(f"dice_coeff = {dce}, loss = {loss}, iou = {mean_iou}, \
    precision = {mean_precision}, recall = {mean_recall}, auc = {mean_auc}")

if __name__ == '__main__':
    main()
