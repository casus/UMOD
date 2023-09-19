from pathlib import Path

import click
import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage import measure
from skimage.io import imread
from tqdm.auto import tqdm

CLASS_MAP = {
    'class 1: rod': 1,
    'class 2: RBC/WBC': 2,
    'class 3: yeast': 3,
    'class 4: misc': 4,
    'class 5: single EPC': 5,
    'class 6: few EPC ': 6,
    'class 7: several EPC': 7
    }


def read_labels(labels_dirs):
    labels_pdfs = []
    for labels_dir in labels_dirs:
        labels_dir = Path(labels_dir)
        for csv_file in labels_dir.rglob('*.csv'):
            labels_pdfs.append(pd.read_csv(csv_file))
    return pd.concat(labels_pdfs).drop_duplicates()

def label_mask(binary_mask, labels_pdf):
    connected_components_mask = measure.label(binary_mask, background=0)
    region_props = measure.regionprops(connected_components_mask)

    centroids = np.array([rp.centroid for rp in region_props])
    centroids = centroids[:, [1, 0]]

    labels_centroids = labels_pdf[['x', 'y']].values

    if len(centroids) < len(labels_centroids):
        raise RuntimeError('Found more labels than connected components')

    distances = cdist(centroids, labels_centroids)
    row_indices, col_indices = linear_sum_assignment(distances)

    multiclass_mask = np.zeros_like(binary_mask)

    for row_idx, col_idx in zip(row_indices, col_indices):
        connected_component_label = region_props[row_idx].label
        true_label = labels_pdf.iloc[col_idx].label
        multiclass_mask[
            connected_components_mask == connected_component_label
        ] = CLASS_MAP[true_label]

    return multiclass_mask


@click.command()
@click.argument('mask_dir', type=click.Path(exists=True))
@click.argument('out_dir', type=click.Path(exists=True))
@click.option('--labels', '-l', multiple=True)
def main(mask_dir, out_dir, labels):

    mask_dir = Path(mask_dir)
    out_dir = Path(out_dir)

    labels_pdf = read_labels(labels)

    # Creates multiclass mask for each file included in labels_pdf and saves them to out_dir
    for img_name, image_pdf in tqdm(labels_pdf.groupby('img')):
        if (mask_dir / f'{img_name}_Simple Segmentation.tif').exists():
            binary_mask = imread(mask_dir / f'{img_name}_Simple Segmentation.tif')
        elif (mask_dir / f'{img_name}_Simple Segmentation_.tif').exists():
            binary_mask = imread(mask_dir / f'{img_name}_Simple Segmentation_.tif')
        else:
            raise RuntimeError(f'Mask not found for image {img_name}')

        multiclass_mask = label_mask(
            binary_mask,
            image_pdf[['x', 'y', 'label']],
        )

        multiclass_mask = Image.fromarray(multiclass_mask)
        multiclass_mask.save(out_dir / f'{img_name}_Simple Segmentation.tif')

    # Some masks are empty, and thus are not included in labels_pdf
    # Checks if all missing masks are empty, and copies them
    file_names = [mask_path.stem.split('_')[0] for mask_path in mask_dir.glob('*_Simple Segmentationx.tif')]
    print(file_names)
    missing_files = np.setdiff1d(file_names, np.unique(labels_pdf['img']))

    print(missing_files)
    for missing_file in missing_files:
        missing_mask = imread(mask_dir / f'{missing_file}_Simple Segmentation.tif')
        if np.any(missing_mask != 0):
            raise RuntimeError('Missing mask is not empty')
        empty_mask = Image.fromarray(missing_mask)
        empty_mask.save(out_dir / f'{missing_file}_Simple Segmentation.tif')

    print(f'{len(missing_files)} empty masks found.')


if __name__ == '__main__':
    main() # pylint: disable=no-value-for-parameter
