import tensorflow.keras.preprocessing.image as image
import tensorflow.keras.backend as backend
from .utils import flip_axis, load_img, img_to_array, array_to_img, apply_random_patch_gen, _list_valid_filenames_in_directory

import numpy as np
import warnings
import os

class customNumpyArrayIterator(image.Iterator):
    """Iterator yielding data from a Numpy array.

    Deprecated: `tf.keras.preprocessing.image.NumpyArrayIterator` is not
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
        x: Numpy array of input data or tuple. If tuple, the second elements is
          either another numpy array or a list of numpy arrays, each of which gets
          passed through as an output without any modifications.
        y: Numpy array of targets data.
        image_data_generator: Instance of `ImageDataGenerator` to use for random
          transformations and normalization.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        sample_weight: Numpy array of sample weights.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures being yielded,
          in a viewable format. This is useful for visualizing the random
          transformations being applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample images (if
          `save_to_dir` is set).
        save_format: Format to use for saving sample images (if `save_to_dir` is
          set).
        subset: Subset of data (`"training"` or `"validation"`) if
          validation_split is set in ImageDataGenerator.
        ignore_class_split: Boolean (default: False), ignore difference
          in number of classes in labels across train and validation
          split (useful for non-classification tasks)
        dtype: Dtype to use for the generated arrays.
    """

    def __init__(
        self,
        x,
        y,
        image_data_generator,
        target_size=(256,256),
        batch_size=32,
        shuffle=False,
        sample_weight=None,
        seed=None,
        data_format=None,
        save_to_dir=None,
        save_prefix="",
        save_format="png",
        subset=None,
        ignore_class_split=False,
        dtype=None,
    ):
        if data_format is None:
            data_format = backend.image_data_format()
        if dtype is None:
            dtype = backend.floatx()
        self.dtype = dtype
        if isinstance(x, tuple) or isinstance(x, list):
            if not isinstance(x[1], list):
                x_misc = [np.asarray(x[1])]
            else:
                x_misc = [np.asarray(xx) for xx in x[1]]
            x = x[0]
            for xx in x_misc:
                if len(x) != len(xx):
                    raise ValueError(
                        "All of the arrays in `x` "
                        "should have the same length. "
                        "Found a pair with: len(x[0]) = %s, len(x[?]) = %s"
                        % (len(x), len(xx))
                    )
        else:
            x_misc = []

        if y is not None and len(x) != len(y):
            raise ValueError(
                "`x` (images tensor) and `y` (labels) "
                "should have the same length. "
                "Found: x.shape = %s, y.shape = %s"
                % (np.asarray(x).shape, np.asarray(y).shape)
            )
        if sample_weight is not None and len(x) != len(sample_weight):
            raise ValueError(
                "`x` (images tensor) and `sample_weight` "
                "should have the same length. "
                "Found: x.shape = %s, sample_weight.shape = %s"
                % (np.asarray(x).shape, np.asarray(sample_weight).shape)
            )
        if subset is not None:
            if subset not in {"training", "validation"}:
                raise ValueError(
                    "Invalid subset name:",
                    subset,
                    '; expected "training" or "validation".',
                )
            split_idx = int(len(x) * image_data_generator._validation_split)

            if (
                y is not None
                and not ignore_class_split
                and not np.array_equal(
                    np.unique(y[:split_idx]), np.unique(y[split_idx:])
                )
            ):
                raise ValueError(
                    "Training and validation subsets "
                    "have different number of classes after "
                    "the split. If your numpy arrays are "
                    "sorted by the label, you might want "
                    "to shuffle them."
                )

            if subset == "validation":
                x = x[:split_idx]
                x_misc = [np.asarray(xx[:split_idx]) for xx in x_misc]
                if y is not None:
                    y = y[:split_idx]
            else:
                x = x[split_idx:]
                x_misc = [np.asarray(xx[split_idx:]) for xx in x_misc]
                if y is not None:
                    y = y[split_idx:]

        self.x = np.asarray(x, dtype=self.dtype)
        self.x_misc = x_misc
        if self.x.ndim != 4:
            raise ValueError(
                "Input data in `NumpyArrayIterator` "
                "should have rank 4. You passed an array "
                "with shape",
                self.x.shape,
            )
        channels_axis = 3 if data_format == "channels_last" else 1
        if self.x.shape[channels_axis] not in {1, 3, 4}:
            warnings.warn(
                "NumpyArrayIterator is set to use the "
                'data format convention "' + data_format + '" '
                "(channels on axis "
                + str(channels_axis)
                + "), i.e. expected either 1, 3, or 4 "
                "channels on axis " + str(channels_axis) + ". "
                "However, it was passed an array with shape "
                + str(self.x.shape)
                + " ("
                + str(self.x.shape[channels_axis])
                + " channels)."
            )
        if y is not None:
            self.y = np.asarray(y)
        else:
            self.y = None
        if sample_weight is not None:
            self.sample_weight = np.asarray(sample_weight)
        else:
            self.sample_weight = None
        self.image_data_generator = image_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        if self.image_data_generator.max_iter != None:
            self.batch_trial_iter = 0
            self.batch_best_thresh_met = 0
            self.batch_best_yet_x = np.zeros(tuple([self.x.shape[0]] + list((self.image_data_generator.target_height,
                                                self.image_data_generator.target_width,1))),
                                            dtype=self.image_data_generator.dtype
            )
            self.batch_best_yet_y = np.zeros(tuple([self.y.shape[0]] + list((self.image_data_generator.target_height,
                                                self.image_data_generator.target_width,1))),
                                            dtype=self.image_data_generator.dtype
            )
        super().__init__(x.shape[0], batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array, background_fill = 0.5):
        
        if self.image_data_generator.max_iter != None:
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
                for i, j in enumerate(index_array):
                    x = self.x[j]
                    y = self.y[j]

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
                    number_of_white_pix = np.sum(y > 0)  # extracting non-white pixels 
                    perc_of_white_pix = number_of_white_pix/(y.shape[0]*y.shape[1])
                    batch_dict[i] = perc_of_white_pix

                
                batch_white_pix_perc_l = [batch_dict[idx] for idx in range(len(batch_dict))]

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
        else:
            batch_x = np.zeros(
            tuple([len(index_array)] + list(self.x.shape)[1:]), dtype=self.image_data_generator.dtype
            )
            batch_y = np.zeros(
                tuple([len(index_array)] + list(self.y.shape)[1:]), dtype=self.image_data_generator.dtype
            )
            for i, j in enumerate(index_array):
                x = self.x[j]
                y = self.y[j]

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
                batch_y[i] = y
            
            batch_x2 = batch_x
            batch_y2 = batch_y

        if self.image_data_generator.max_iter != None:
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
                
        batch_x_miscs = [xx[index_array] for xx in self.x_misc]
        output = (batch_x2 if not batch_x_miscs else [batch_x2] + batch_x_miscs,)
        if self.y is None:
            return output[0]

        output += (np.asarray(batch_y2),)
        if self.sample_weight is not None:
            output += (self.sample_weight[index_array],)
        return output
