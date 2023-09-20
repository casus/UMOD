import multiprocessing
import os
import random

import numpy as np
import tensorflow.keras.backend as backend
import tensorflow.keras.preprocessing.image as image

from .customBatchFromFilesMixin import customBatchFromFilesMixin
from .utils import _list_valid_filenames_in_directory


class customDirectoryIterator(customBatchFromFilesMixin, image.Iterator):
    """Iterator capable of reading images from a directory on disk.

    Deprecated: `tf.keras.preprocessing.image.DirectoryIterator` is not
    recommended for new code. Prefer loading images with
    `tf.keras.utils.image_dataset_from_directory` and transforming the output
    `tf.data.Dataset` with preprocessing layers. For more information, see the
    tutorials for [loading images](
    https://www.tensorflow.org/tutorials/load_data/images) and
    [augmenting images](
    https://www.tensorflow.org/tutorials/images/data_augmentation), as well as
    the [preprocessing layer guide](
    https://www.tensorflow.org/guide/keras/preprocessing_layers).

    Args:
        directory: Path to the directory to read images from. Each subdirectory
          in this directory will be considered to contain images from one class,
          or alternatively you could specify class subdirectories via the
          `classes` argument.
        image_data_generator: Instance of `ImageDataGenerator` to use for random
          transformations and normalization.
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"rgba"`, `"grayscale"`. Color mode to read
          images.
        classes: Optional list of strings, names of subdirectories containing
          images from each class (e.g. `["dogs", "cats"]`). It will be computed
          automatically if not set.
        class_mode: Mode for yielding the targets:
            - `"binary"`: binary targets (if there are only two classes),
            - `"categorical"`: categorical targets,
            - `"sparse"`: integer targets,
            - `"input"`: targets are images identical to input images (mainly
              used to work with autoencoders),
            - `None`: no targets get yielded (only input images are yielded).
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures being
          yielded, in a viewable format. This is useful for visualizing the
          random transformations being applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample images (if
          `save_to_dir` is set).
        save_format: Format to use for saving sample images (if `save_to_dir` is
          set).
        subset: Subset of data (`"training"` or `"validation"`) if
          validation_split is set in ImageDataGenerator.
        interpolation: Interpolation method used to resample the image if the
          target size is different from that of the loaded image. Supported
          methods are "nearest", "bilinear", and "bicubic". If PIL version 1.1.3
          or newer is installed, "lanczos" is also supported. If PIL version
          3.4.0 or newer is installed, "box" and "hamming" are also supported.
          By default, "nearest" is used.
        keep_aspect_ratio: Boolean, whether to resize images to a target size
            without aspect ratio distortion. The image is cropped in the center
            with target aspect ratio before resizing.
        dtype: Dtype to use for generated arrays.
    """

    allowed_class_modes = {"categorical", "binary", "sparse", "input", None}

    def __init__(
        self,
        directory_x,
        directory_y,
        image_data_generator,
        target_size=(256, 256),
        color_mode="rgb",
        classes=None,
        class_mode="categorical",
        batch_size=32,
        shuffle=True,
        seed=None,
        data_format=None,
        save_to_dir=None,
        save_prefix="",
        save_format="png",
        follow_links=False,
        subset=None,
        interpolation="nearest",
        keep_aspect_ratio=False,
        dtype=None,
        resampling_number = 1000,
        fold_resolution = 0.5,
    ):
        if data_format is None:
            data_format = backend.image_data_format()
        if dtype is None:
            dtype = backend.floatx()
        super().set_processing_attrs(
            image_data_generator,
            target_size,
            color_mode,
            data_format,
            save_to_dir,
            save_prefix,
            save_format,
            subset,
            interpolation,
            keep_aspect_ratio,
            0,
            fold_resolution,
        )
        self.directory_x = directory_x
        self.directory_y = directory_y
        self.classes = classes
        if class_mode not in self.allowed_class_modes:
            raise ValueError(
                "Invalid class_mode: {}; expected one of: {}".format(
                    class_mode, self.allowed_class_modes
                )
            )
        self.class_mode = class_mode
        self.dtype = dtype
        # First, count the number of samples and classes.
        self.samples = 0

        if not classes:
            classes = []
            for subdir in sorted(os.listdir(directory_x)):
                if os.path.isdir(os.path.join(directory_x, subdir)):
                    classes.append(subdir)

        self.num_classes = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        pool = multiprocessing.pool.ThreadPool()

        # Second, build an index of the images
        # in the different class subfolders.
        results_x = []
        self.filenames_x = []


        i = 0
        for dirpath in (os.path.join(directory_x, subdir) for subdir in classes):
            results_x.append(
                pool.apply_async(
                    _list_valid_filenames_in_directory,
                    (
                        dirpath,
                        self.white_list_formats,
                        self.split,
                        self.class_indices,
                        follow_links,
                    ),
                )
            )

        pool.close()
        pool.join()

        pool = multiprocessing.pool.ThreadPool()
        results_y = []
        self.filenames_y = []

        for dirpath in (os.path.join(directory_y, subdir) for subdir in classes):
            results_y.append(
                pool.apply_async(
                    _list_valid_filenames_in_directory,
                    (
                        dirpath,
                        self.white_list_formats,
                        self.split,
                        self.class_indices,
                        follow_links,
                    ),
                )
            )

        pool.close()
        pool.join()

        classes_list = []
        for res in results_x:
            classes, filenames = res.get()
            classes_list.append(classes)
            self.filenames_x += filenames

        for res in results_y:
            classes, filenames = res.get()
            # classes_list.append(classes)
            self.filenames_y += filenames

        # resample
        sample_idxs = random.choices(list(range(len(self.filenames_x))), k = resampling_number)
        resampled_x = []
        resampled_y = []

        for idx in sample_idxs:
            resampled_x.append(self.filenames_x[idx])
            resampled_y.append(self.filenames_y[idx])

        cls_list = []
        resampled_classes = []
        for cls in classes_list:
            for idx in sample_idxs:
                cls_list.append(cls[idx])
            resampled_classes.append(cls_list)

        self.filenames_x = resampled_x
        self.filenames_y = resampled_y
        classes_list = resampled_classes

        self.samples_x = len(self.filenames_x)
        self.samples_y = len(self.filenames_y)

        # print(self.samples_x)
        # print(self.samples_y)
        # print(classes_list)

        self.classes = np.zeros((self.samples_x,), dtype="int32")
        for classes in classes_list:
            self.classes[i : i + len(classes)] = classes
            i += len(classes)

        print(
            "Found %d images belonging to %d classes."
            % (self.samples_x, self.num_classes)
        )

        self._filepaths_x = [
            os.path.join(self.directory_x, fname) for fname in self.filenames_x
        ]
        self._filepaths_y = [
            os.path.join(self.directory_y, fname) for fname in self.filenames_y
        ]

        super().set_processing_attrs(
            image_data_generator,
            target_size,
            color_mode,
            data_format,
            save_to_dir,
            save_prefix,
            save_format,
            subset,
            interpolation,
            keep_aspect_ratio,
            self.samples_x,
            fold_resolution,
        )

        super().__init__(self.samples_x, batch_size, shuffle, seed)

    @property
    def filepaths(self):
        return self._filepaths_x, self._filepaths_y

    @property
    def labels(self):
        return self.classes

    @property  # mixin needs this property to work
    def sample_weight(self):
        # no sample weights will be returned
        return None
