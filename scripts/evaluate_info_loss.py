import ast
import glob
import io
import json
import os
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None

if pil_image is not None:
    _PIL_INTERPOLATION_METHODS = {
        "nearest": pil_image.NEAREST,
        "bilinear": pil_image.BILINEAR,
        "bicubic": pil_image.BICUBIC,
        "hamming": pil_image.HAMMING,
        "box": pil_image.BOX,
        "lanczos": pil_image.LANCZOS,
    }

import random
import time
import warnings

import seaborn as sns

from src.keras_patch_generator.utils import img_to_array, load_img
from src.utils.config import read_json_config
from src.utils.losses import dice_coeff as dice_coeff_tf
from src.utils.train import checkDir


class PythonLiteralOption(click.Option):

    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except:
            raise click.BadParameter(value)


@click.command()
@click.option('--config_file_path', type=click.Path(exists=True))
@click.option('--fold_res_l', cls=PythonLiteralOption, default=[0.2,0.3,0.5],help='Fold resolutions to compare', show_default=True)
@click.option('--k', default=25, type=int, help='Number of files to sample for comparing info loss', show_default=True)
@click.option('--iter', default=5, type=int, help='Number of times to iterate experiment for comparing info loss', show_default=True)

def main(config_file_path, fold_res_l, k, iter):
    current_dir = Path(__file__).parent
    runningTime = time.strftime('%b-%d-%Y_%H-%M')
    model_dir = current_dir / 'model'
    log_dir = model_dir / 'logs'/ f'{runningTime}'
    log_dir.mkdir(parents=True, exist_ok=True)

    config = read_json_config(config_file_path)

    test_masks_path=Path(config['test']['mask_dir'])
    info_loss_iter_ls = find_info_loss(test_masks_path, fold_res_l, k, iter)
    info_loss_iter_ls = np.array(info_loss_iter_ls).flatten()
    info_loss_df = pd.DataFrame({
        "dice_coeff": info_loss_iter_ls,
        "fold": np.array([[fold_res_l]*iter]).flatten()
    })


    print(info_loss_df)

    # boxplot_info_loss(info_loss_df, save_path = log_dir / 'graph_info_loss.svg')
    barplot_info_loss(info_loss_df, save_path = log_dir / 'graph_info_loss.svg')

def load_img_res(msk_path,fold_resolution):
    mask = load_img(
                msk_path,
                color_mode="grayscale",
                target_size=(1040,1392),
                interpolation="nearest",
                keep_aspect_ratio=True,
                fold_resolution=fold_resolution,
                )
    return mask

def dice_coeff(y_true, y_pred, smooth = 1.):
    """ Calculate dice coefficient.

    Args:
        y_true (np.array): Ground truth values. shape = [d0, ..., dN]
        y_pred (np.array): Predicted values. shape = [d0, ..., dN]
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        float: Dice coefficient value. shape = [d0, ..., dN-1]
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    return score

def find_info_loss(test_masks_path, fold_res_l, k, iter):

    file_ls = list(glob.glob(str(test_masks_path / 'cls' / '*.tif')))

    info_loss_iter_ls = []
    i = 0
    while(i < iter):

        info_loss_over_res_ls = []
        file_ls_temp = random.choices(file_ls,k=k)

        for f in file_ls_temp:
            info_loss_ls = []
            msk_path = f


            msk_orig = load_img_res(msk_path,fold_resolution=1.0)
            msk_orig = img_to_array(msk_orig, data_format='channels_last')/255.0

            msk_ls = [load_img_res(msk_path,fold) if fold!=1.0 else msk_orig for (msk_path,fold) in zip([msk_path]*len(fold_res_l),fold_res_l)]
            msk_ls = [msk.resize((msk_orig.shape[1],msk_orig.shape[0]), resample=pil_image.BOX) if fold!=1.0 else msk_orig for (msk,fold) in zip(msk_ls,fold_res_l)]
            msk_ls = [np.rint(img_to_array(msk, data_format='channels_last')/255.0) if fold!=1.0 else msk_orig for (msk,fold) in zip(msk_ls,fold_res_l)]

            info_loss_ls = [dice_coeff(np.squeeze(msk),np.squeeze(msk_low_res)) for msk,msk_low_res in list(zip([msk_orig]*len(fold_res_l),msk_ls))]
            info_loss_over_res_ls.append(info_loss_ls)

        info_loss_over_res_ls = np.array(info_loss_over_res_ls)
        info_loss_over_res_ls = info_loss_over_res_ls.mean(axis=0)
        info_loss_iter_ls.append(info_loss_over_res_ls)

        i+=1

    return info_loss_iter_ls


def boxplot_info_loss(info_loss_df, save_path):
    sns.boxplot(
        data=info_loss_df,
        x="fold",
        y="dice_coeff",
        notch=False,
        showcaps=True,
        flierprops={"marker": "x"},
        boxprops={"facecolor": (.4, .6, .8, .5)},
        medianprops={"color": "coral"}
    ).set(xlabel='Fold Resolution', ylabel='Dice Coefficient')

    # plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    return

def barplot_info_loss(info_loss_df, save_path):
    sns.set(font_scale=1.5)
    sns.set_style("ticks")
    sns.barplot(
        x="fold",
        y="dice_coeff",
        data=info_loss_df,
        errorbar="sd",
        capsize=.2,
        color='black',edgecolor='black'
    ).set(xlabel='Fold resolution', ylabel='Dice coefficient')

    # plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    return

if __name__ == "__main__":
    main()
