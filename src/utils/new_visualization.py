from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np


def visualise(image, pred, ground_truth, save_path):
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(40,8))
    plt.subplot(1, 3, 1)
    plt.title('Input')
    plt.imshow(np.squeeze(image))

    plt.subplot(1, 3, 2)
    plt.title('Prediction')
    plt.imshow(np.squeeze(np.rint(pred)),cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title('Ground truth')
    plt.imshow(np.squeeze(ground_truth),cmap='gray')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def visualise_batch(images, preds, ground_truths, save_dir):
    for img_num in range(len(images)):
        visualise(images[img_num], preds[img_num], ground_truths[img_num], save_dir / f'{img_num}.svg')

def plot_metrics(history, attribute: str, save_path: Union[str, Path]):
    plt.figure()
    plt.plot(history.history[attribute])
    plt.plot(history.history[f'val_{attribute}'])
    plt.title(f'Manual labels training {attribute}')
    plt.ylabel(attribute)
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
