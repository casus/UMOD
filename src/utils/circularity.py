from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as TIFF
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage import measure

STREL_4 = np.array([[0, 1, 0],
                    [1, 1, 1],
                    [0, 1, 0]], dtype=np.uint8)
STREL_8 = np.ones((3, 3), dtype=np.uint8)

def picks_area(image, neighbourhood=4):
    if neighbourhood == 4:
        strel = STREL_4
    else:
        strel = STREL_8
    image = image.astype(np.uint8)
    eroded_image = ndi.binary_erosion(image, strel, border_value=0)
    border_image = image - eroded_image

    perimeter_weights = np.zeros(50, dtype=np.double)
    perimeter_weights[[5, 7, 15, 17, 25, 27]] = 0.25
    perimeter_weights[[21, 33]] = 1
    perimeter_weights[[13, 23]] = 0.125

    perimeter_image = ndi.convolve(border_image, np.array([[10, 2, 10],
                                                           [2, 1, 2],
                                                           [10, 2, 10]]),
                                   mode='constant', cval=0)

    # You can also write
    # return perimeter_weights[perimeter_image].sum()
    # but that was measured as taking much longer than bincount + np.dot (5x
    # as much time)
    perimeter_histogram = np.bincount(perimeter_image.ravel(), minlength=50)
    total_perimeter = np.dot(perimeter_histogram, perimeter_weights)

    v = np.count_nonzero(eroded_image)

    if v == 0:
        s = total_perimeter
    else:
        s = v + total_perimeter / 2 - 1

    return s

def new_std_perimeter(image, neighbourhood=4):
    STREL_4 = np.array([[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]], dtype=np.uint8)
    STREL_8 = np.ones((3, 3), dtype=np.uint8)

    if neighbourhood == 4:
        strel = STREL_4
    else:
        strel = STREL_8
    image = image.astype(np.uint8)

    (w, h) = image.shape
    data = np.zeros((w + 2, h + 2), dtype=image.dtype)
    data[1:-1, 1:-1] = image
    image = data

    eroded_image = ndi.binary_dilation(image, strel, border_value=0)
    border_image = eroded_image - image

    perimeter_weights = np.zeros(50, dtype=np.double)
    perimeter_weights[[5, 7, 15, 17, 25, 27]] = 1
    perimeter_weights[[21, 33]] = np.sqrt(2)
    perimeter_weights[[13, 23]] = (1 + np.sqrt(2)) / 2

    perimeter_image = ndi.convolve(border_image, np.array([[10, 2, 10],
                                                           [2, 1, 2],
                                                           [10, 2, 10]]),
                                   mode='constant', cval=0)

    # You can also write
    # return perimeter_weights[perimeter_image].sum()
    # but that was measured as taking much longer than bincount + np.dot (5x
    # as much time)
    perimeter_histogram = np.bincount(perimeter_image.ravel(), minlength=50)
    total_perimeter = np.dot(perimeter_histogram, perimeter_weights)
    return total_perimeter

def modified_circ(area, perimeter):
    if perimeter !=0:
        return (4 * np.pi * area) / (perimeter * perimeter)
    else:
        return 0


def get_corres_circularity(df,split='train'):
    img_list = df['img'].unique()
    new_df = pd.DataFrame()
    for f in img_list:
        temp = df[df['img']==f]

        # print(temp.shape)
        temp_file = pd.DataFrame()

        # temp_file['label'] = temp['label']
        # temp_file['x'] = temp['x']
        # temp_file['y'] = temp['y']
        # temp_file['img'] = temp['img']


        # temp_file2 = pd.DataFrame()

        if split=='train':
            image_path = Path('../../../ds1/train/man_mask/cls/') / (f + '_Simple Segmentation.tif')
        elif split =='validation':
            image_path = Path('../../../ds1/validation/man_mask/cls/') / (f + '_Simple Segmentation.tif')
        elif split=='test':
            image_path = Path('../../../ds1/test/man_mask/cls/') / (f + '_Simple Segmentation.tif')

        img = TIFF.imread(image_path)
        img2 = np.empty((img.shape[0],img.shape[1],3))
        # print(img2.shape)
        img2[:,:,0] = img
        img2[:,:,1] = img
        img2[:,:,2] = img

        labelImage = measure.label(np.squeeze(img2[:,:,0]), background=0)
        props = measure.regionprops(labelImage)
        # area = []
        # circu = []
        # c1 = []
        # c2 = []
        # for prop in props:
        #     y,x =prop.centroid
        #     area.append(prop.area)
        #     circu.append(modified_circ(picks_area(prop.image),new_std_perimeter(prop.image)))
        #     # circu.append(apply_correction(prop,circ(prop)))
        #     c1.append(x)
        #     c2.append(y)


        # temp_file2['area'] = area
        # temp_file2['circularity']=circu
        # temp_file2['x'] = c1
        # temp_file2['y'] = c2

        #Assigning new found centroids to centroids in csv
        centroids = np.array([rp.centroid for rp in props])
        centroids = centroids[:, [1, 0]]

        labels_centroids = temp[['x', 'y']].values

        if len(centroids) < len(labels_centroids):
            raise RuntimeError('Found more labels than connected components')

        distances = cdist(centroids, labels_centroids)
        row_indices, col_indices = linear_sum_assignment(distances)

        area = []
        circu = []
        c1 = []
        c2 = []
        label=[]
        for row_idx, col_idx in zip(row_indices, col_indices):
            c1.append(temp.iloc[col_idx].x)
            c2.append(temp.iloc[col_idx].y)
            area.append(temp.iloc[col_idx].area)
            circu.append(modified_circ(picks_area(props[row_idx].image),new_std_perimeter(props[row_idx].image)))
            label.append(temp.iloc[col_idx].label)

        # for x,y in list(zip(list(temp_file['x']),list(temp_file['y']))):
        #     row = lookup_table(temp_file2, (y,x))
        #     area.append(row.iloc[0]['area'])
        #     circu.append(row.iloc[0]['circularity'])

        temp_file['label'] = label
        temp_file['x'] = c1
        temp_file['y'] = c2
        temp_file['img'] = f
        temp_file['area'] = area
        temp_file['circularity'] = circu

        new_df = pd.concat([new_df,temp_file])

    return new_df
