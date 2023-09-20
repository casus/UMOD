import glob
import os
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import Model
from skimage.io import imread
from tensorflow import Tensor

from src.keras_patch_generator.utils import img_to_array, load_img
from src.utils.losses import dice_coeff
from src.utils.visualization import visualise

 # sequenced_patchified_predict uses product, but it's undefined. Is this correct?

def checkDir(directory: str) -> None:
    """ Check if given directory exists; create it if it does not.

    Args:
        directory (str): Path to the directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def read_imgs(dir: str, img_rows: int, img_cols: int) -> np.ndarray:
    """ Read images in the directory into a numpy array.

    Args:
        dir (str): Path to the directory containing images.
        img_rows (int): _description_
        img_cols (int): _description_

    Returns:
        np.ndarray: Array containg the images.
    """
    images = [f for f in os.listdir(dir) if f.endswith('.tif')]
    imgs = np.ndarray((len(images), img_rows, img_cols, 1), dtype=np.float)
    for idx, img in enumerate(images):
        #print(idx)
        img = imread(os.path.join(dir, img), as_gray=True)
        img = np.expand_dims(img, axis=-1)
        imgs[idx] = img
    return imgs

def test_model(model: Model, x: Tensor, y_true: Tensor, log_dir: str) -> None:
    """ Test the model on a given sample and plot the comparison.

    Args:
        model (Model): Model to be tested.
        x (Tensor): Sample to test on.
        y_true (Tensor): Ground truth for the sample.
        log_dir (str): Directory in which to save the plot.
    """
    y_true = y_true[0]
    y_pred = model.predict(x)
    y_pred = y_pred[0]
    x = x[0]

    plt.figure(figsize=(40,8))
    plt.subplot(1, 3, 1)
    plt.title('Input')
    plt.imshow(np.squeeze(x))

    plt.subplot(1, 3, 2)
    plt.title('Prediction')
    plt.imshow(np.squeeze(y_pred))

    plt.subplot(1, 3, 3)
    plt.title('Ground truth')
    plt.imshow(np.squeeze(y_true))

    plt.savefig(os.path.join(log_dir,'ground_truth_e{}.png'.format('test')), bbox_inches='tight', dpi=300)
    # plt.show()

def read_imgs_masks(img_dir: str, mask_dir: str, img_rows: int, img_cols: int,
                    target_width: int, target_height: int) -> tuple[np.ndarray, np.ndarray]:
    """ Read images and masks in the given directories into two numpy arrays.

    Args:
        img_dir (str): The directory containing the images.
        mask_dir (str): The directory containing the masks.
        img_rows (int): _description_
        img_cols (int): _description_
        target_width (int): _description_
        target_height (int): _description_

    Returns:
        tuple[np.ndarray, np.ndarray]: Arrays containing respectively images and masks.
    """
    images = [f for f in os.listdir(img_dir) if f.endswith('.tif')]
    imgs = np.ndarray((len(images), target_height, target_width, 1), dtype=np.float64)
    masks = np.ndarray((len(images), target_height, target_width, 1), dtype=np.float64)
    for idx, img in enumerate(images):
        msk = img.replace('.tif', '_Simple Segmentation.tif')
        img = imread(os.path.join(img_dir, img), as_gray=True)
        img = np.expand_dims(img, axis=-1)
        imgs[idx] = img

        mask = imread(os.path.join(mask_dir, msk), as_gray=True)
        mask = np.expand_dims(mask, axis=-1)
        masks[idx] = mask
    return imgs, masks

def sequenced_patchified_predict(img: np.ndarray, size: int, model: Model) -> np.ndarray:
    """ Takes an image and a given size and returns chunks of that image as a list but in an ordered sequence

    Args:
        img (np.ndarray): _description_
        size (int): _description_
        model (Model): _description_

    Returns:
        _type_: _description_
    """
    weight, height = img.size
    pred = np.zeros(img.size)
    grid = product(range(0, height-height%size, size), range(0, weight-weight%size, size))
    for i, j in grid:
        box = (j, i, j+size, i+size)
        pred[j:j+size,i:i+size] = model.predict(img.crop(box))
    return pred

def divide_img_blocks(img, n_blocks=(1280//256, 1536//256)):
    horizontal = np.array_split(img, n_blocks[0])
    splitted_img = [np.array_split(block, n_blocks[1], axis=1) for block in horizontal]
    return np.asarray(splitted_img, dtype=np.ndarray)

def combine_img_blocks(img_array):
    combined_img = np.array([])
    for i in range(img_array.shape[0]):
        temp = np.array([])
        for j in range(img_array.shape[1]):
            temp = np.hstack([temp,np.squeeze(img_array[i,j])]).astype(np.float32) if temp.size else np.squeeze(img_array[i,j])
        combined_img = np.vstack([combined_img,temp]).astype(np.float32) if combined_img.size else temp

    return combined_img

def average_performance(
    test_imgs_path,
    test_masks_path,
    model,
    log_dir,
    image_mask_datagen_avg_test_per,
    img_rows,
    img_cols,
    target_height,
    target_width,
    fold_resolution=0.5,
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

    dice_coeff_ls = []
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

        if "5868 I 280921_Simple Segmentation.tif" in str(msk_path):
            visualise(x, pred, y, log_dir / 'sample_ground_truth_test.svg')

        dice_coeff_ls.append(dice_coeff(np.squeeze(y),pred))

    print(len(dice_coeff_ls))

    return np.mean(dice_coeff_ls)

def normalize(array):
    return (array - array.min()) / (array.max() - array.min())

def load_image_mask(
    img_path,
    msk_path,
    image_tf,
    mask_tf,
    image_mask_datagen,
    fold_resolution=0.5,
    ):

        img = load_img(
                img_path,
                color_mode="grayscale",
                target_size=(1040,1392),
                interpolation="nearest",
                keep_aspect_ratio=True,
                fold_resolution=fold_resolution,
            )
        mask = load_img(
                msk_path,
                color_mode="grayscale",
                target_size=(1040,1392),
                interpolation="nearest",
                keep_aspect_ratio=True,
                fold_resolution=fold_resolution,
                )
        x = img_to_array(img, data_format='channels_last')
        y = img_to_array(mask, data_format='channels_last')
        # Pillow images should be closed after `load_img`,
        # but not PIL images.
        if hasattr(img, "close"):
            img.close()
        if hasattr(mask, "close"):
            mask.close()

        x = image_mask_datagen.standardize(x, image_tf)
        y = image_mask_datagen.standardize(y, mask_tf)
        y = normalize(y)

        return x, y

def round_dimension(img_dim, target_dim):
    return (img_dim//target_dim+1)*target_dim

def get_img_array(x, img_rows, img_cols, target_height, target_width):
    x_padded = np.pad(
        np.squeeze(x),
        (
            (0, round_dimension(img_rows, target_height) - img_rows),
            (0, round_dimension(img_cols, target_width) - img_cols)
        ),
        mode='constant',
        constant_values=0.5,
    )


    img_array = divide_img_blocks(
        np.expand_dims(x_padded, axis=-1),
        n_blocks=(
            x_padded.shape[0]//target_height,
            x_padded.shape[1]//target_width
        )
    )


    return img_array



def predict(model, img_array):
        pred_array = np.empty_like(img_array)
        for i in range(img_array.shape[0]):
            for j in range(img_array.shape[1]):
                pred_array[i,j] = model.predict(tf.convert_to_tensor(img_array[i,j][np.newaxis],tf.float32))

        pred = combine_img_blocks(pred_array)

        pred = np.float32(pred[:1040,:1392])
        pred = normalize(pred)

        return pred
