import numpy as np
import os
from .utils import load_img, img_to_array, array_to_img

class customBatchFromFilesMixin:
    """Adds methods related to getting batches from filenames.

    It includes the logic to transform image files to batches.
    """

    def set_processing_attrs(
        self,
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
        samples_x,
        fold_resolution
    ):
        """Sets attributes to use later for processing files into a batch.

        Args:
            image_data_generator: Instance of `ImageDataGenerator`
                to use for random transformations and normalization.
            target_size: tuple of integers, dimensions to resize input images
            to.
            color_mode: One of `"rgb"`, `"rgba"`, `"grayscale"`.
                Color mode to read images.
            data_format: String, one of `channels_first`, `channels_last`.
            save_to_dir: Optional directory where to save the pictures
                being yielded, in a viewable format. This is useful
                for visualizing the random transformations being
                applied, for debugging purposes.
            save_prefix: String prefix to use for saving sample
                images (if `save_to_dir` is set).
            save_format: Format to use for saving sample images
                (if `save_to_dir` is set).
            subset: Subset of data (`"training"` or `"validation"`) if
                validation_split is set in ImageDataGenerator.
            interpolation: Interpolation method used to resample the image if
                the target size is different from that of the loaded image.
                Supported methods are "nearest", "bilinear", and "bicubic". If
                PIL version 1.1.3 or newer is installed, "lanczos" is also
                supported. If PIL version 3.4.0 or newer is installed, "box" and
                "hamming" are also supported. By default, "nearest" is used.
            keep_aspect_ratio: Boolean, whether to resize images to a target
                size without aspect ratio distortion. The image is cropped in
                the center with target aspect ratio before resizing.
            fold_resolution: Float, a factor by which to rescale the images.
        """
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.keep_aspect_ratio = keep_aspect_ratio
        self.fold_resolution = fold_resolution
        if color_mode not in {"rgb", "rgba", "grayscale"}:
            raise ValueError(
                "Invalid color mode:",
                color_mode,
                '; expected "rgb", "rgba", or "grayscale".',
            )
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == "rgba":
            if self.data_format == "channels_last":
                self.image_shape = self.target_size + (4,)
            else:
                self.image_shape = (4,) + self.target_size
        elif self.color_mode == "rgb":
            if self.data_format == "channels_last":
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == "channels_last":
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.interpolation = interpolation
        self.batch_trial_iter = 0
        self.batch_best_thresh_met = 0
        self.batch_best_yet_x = np.zeros(tuple([samples_x] + list((self.image_data_generator.target_height,
                                            self.image_data_generator.target_width,1))), dtype=self.image_data_generator.dtype
        )
        self.batch_best_yet_y = np.zeros(tuple([samples_x] + list((self.image_data_generator.target_height,
                                            self.image_data_generator.target_width,1))), dtype=self.image_data_generator.dtype
        )


        if subset is not None:
            validation_split = self.image_data_generator._validation_split
            if subset == "validation":
                split = (0, validation_split)
            elif subset == "training":
                split = (validation_split, 1)
            else:
                raise ValueError(
                    "Invalid subset name: %s;"
                    'expected "training" or "validation"' % (subset,)
                )
        else:
            split = None
        self.split = split
        self.subset = subset

    def _get_batches_of_transformed_samples(self, index_array, background_fill = 0.5):
        """Gets a batch of transformed samples.

        Args:
            index_array: Array of sample indices to include in batch.
        Returns:
            A batch of transformed samples.
        """

        #Check condition for validity of the batch
        while (self.batch_trial_iter<=self.image_data_generator.max_iter):
            batch_dict = {}
            batch_x = np.zeros(
            tuple([len(index_array)] + list((self.image_data_generator.target_height,
                                            self.image_data_generator.target_width,1))), dtype=self.image_data_generator.dtype
            )
            batch_y = np.zeros(
                tuple([len(index_array)] + list((self.image_data_generator.target_height,
                                                self.image_data_generator.target_width,1))), dtype=self.image_data_generator.dtype
            )
            # build batch of image data
            # self.filepaths is dynamic, is better to call it once outside the loop
            filepaths_x, filepaths_y = self.filepaths[0],self.filepaths[1]

            choice_indices = np.random.choice(range(len(filepaths_x)), size=len(filepaths_x), replace=True)
            choice_indices = list(choice_indices)
            filepaths_x, filepaths_y = [filepaths_x[a] for a in choice_indices], [filepaths_y[a] for a in choice_indices]

            # print("Retrieved filepaths...")
            # print(filepaths_x)
            # print(filepaths_y)
            # print(index_array)
            for i, j in enumerate(index_array):
                img = load_img(
                    filepaths_x[j],
                    color_mode=self.color_mode,
                    target_size=self.target_size,
                    interpolation=self.interpolation,
                    keep_aspect_ratio=self.keep_aspect_ratio,
                    fold_resolution=self.fold_resolution
                )
                mask = load_img(
                    filepaths_y[j],
                    color_mode=self.color_mode,
                    target_size=self.target_size,
                    interpolation=self.interpolation,
                    keep_aspect_ratio=self.keep_aspect_ratio,
                    fold_resolution=self.fold_resolution
                )
                x = img_to_array(img, data_format=self.data_format)
                y = img_to_array(mask, data_format=self.data_format)
                # Pillow images should be closed after `load_img`,
                # but not PIL images.
                if hasattr(img, "close"):
                    img.close()
                if hasattr(mask, "close"):
                    mask.close()

                if self.image_data_generator:
                    params = self.image_data_generator.get_random_transform(x.shape) #dict
                    x = self.image_data_generator.apply_transform(
                    x.astype(self.image_data_generator.dtype), params, self.image_data_generator.image_aug_check
                    )

                    y = self.image_data_generator.apply_transform(
                        y.astype(self.image_data_generator.dtype), params, self.image_data_generator.mask_aug_check
                    )

                    x = self.image_data_generator.standardize(x,self.image_data_generator.image_aug_check)
                    y = self.image_data_generator.standardize(y,self.image_data_generator.mask_aug_check)

                    x[x == 0] = background_fill
                    batch_x[i] = x
                    y = np.where(y > 0.5, 1, 0)
                    batch_y[i] = y


                    # y_bw = y
                    number_of_white_pix = np.sum(y > 0)  # extracting non-white pixels
                    perc_of_white_pix = number_of_white_pix/(y.shape[0]*y.shape[1])
                    batch_dict[i] = perc_of_white_pix

            batch_white_pix_perc_l = [batch_dict[idx] for idx in range(len(batch_dict))]

#             batch_x = batch_x / batch_x.max()

            if np.mean(batch_white_pix_perc_l) < self.image_data_generator.thresh_obj_perc :
                if np.mean(batch_white_pix_perc_l) > self.batch_best_thresh_met:
                    self.batch_best_thresh_met = np.mean(batch_white_pix_perc_l)
                    self.batch_best_yet_x = batch_x
                    self.batch_best_yet_y = batch_y
                else:
                    pass
            else:
                self.batch_best_thresh_met = np.mean(batch_white_pix_perc_l)
                self.batch_best_yet_x = batch_x
                self.batch_best_yet_y = batch_y
                break
            self.batch_trial_iter+=1

        batch_x2 = self.batch_best_yet_x
        batch_y2 = self.batch_best_yet_y

        self.batch_trial_iter = 0
        self.batch_best_thresh_met = 0
        self.batch_best_yet_x = np.zeros(
        tuple([len(index_array)] + list((self.image_data_generator.target_height,
                                        self.image_data_generator.target_width,1))), dtype=self.image_data_generator.dtype
        )

        self.batch_best_yet_y = np.zeros(
        tuple([len(index_array)] + list((self.image_data_generator.target_height,
                                        self.image_data_generator.target_width,1))), dtype=self.image_data_generator.dtype
        )

        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(
                    batch_x2[i], self.data_format, scale=True
                )
                mask = array_to_img(
                    batch_y2[i], self.data_format, scale=True
                )
                common_hash = np.random.randint(1e7)
                fname_img = "{prefix}_{index}_{hash}.{format}".format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=common_hash,
                    format=self.save_format,
                )
                fname_mask = "{prefix}_{index}_{hash}.{format}".format(
                    prefix=self.save_prefix+"_mask",
                    index=j,
                    hash=common_hash,
                    format=self.save_format,
                )
                img.save(os.path.join(self.save_to_dir, fname_img))
                mask.save(os.path.join(self.save_to_dir, fname_mask))

        # build batch of labels
        if self.class_mode == "input":
            batch_y2 = batch_x2.copy()
        elif self.class_mode in {"binary", "sparse"}:
            batch_y2 = np.empty(len(batch_x2), dtype=self.image_data_generator.dtype)
            for i, n_observation in enumerate(index_array):
                batch_y2[i] = self.classes[n_observation]
        elif self.class_mode == "categorical":
            batch_y2 = np.zeros(
                (len(batch_x2), len(self.class_indices)), dtype=self.image_data_generator.dtype
            )
            for i, n_observation in enumerate(index_array):
                batch_y2[i, self.classes[n_observation]] = 1.0
        elif self.class_mode == "multi_output":
            batch_y2 = [output[index_array] for output in self.labels]
        elif self.class_mode == "raw":
            batch_y2 = self.labels[index_array]
        else:
            return batch_x2, batch_y2
        if self.sample_weight is None:
            return batch_x2, batch_y2
        else:
            return batch_x2, batch_y2, self.sample_weight[index_array]

    @property
    def filepaths(self):
        """List of absolute paths to image files."""
        raise NotImplementedError(
            "`filepaths` property method has not "
            "been implemented in {}.".format(type(self).__name__)
        )

    @property
    def labels(self):
        """Class labels of every observation."""
        raise NotImplementedError(
            "`labels` property method has not been implemented in {}.".format(
                type(self).__name__
            )
        )

    @property
    def sample_weight(self):
        raise NotImplementedError(
            "`sample_weight` property method has not "
            "been implemented in {}.".format(type(self).__name__)
        )
