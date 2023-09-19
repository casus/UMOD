import io
import os
import pathlib
import warnings

import numpy as np
import tensorflow.keras.backend as backend

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

def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def _iter_valid_files(directory, white_list_formats, follow_links):
    """Iterates on files with extension.

    Args:
        directory: Absolute path to the directory
            containing files to be counted
        white_list_formats: Set of strings containing allowed extensions for
            the files to be counted.
        follow_links: Boolean, follow symbolic links to subdirectories.
    Yields:
        Tuple of (root, filename) with extension in `white_list_formats`.
    """

    def _recursive_list(subpath):
        return sorted(
            os.walk(subpath, followlinks=follow_links), key=lambda x: x[0]
        )

    for root, _, files in _recursive_list(directory):
        for fname in sorted(files):
            if fname.lower().endswith(".tiff"):
                warnings.warn(
                    'Using ".tiff" files with multiple bands '
                    "will cause distortion. Please verify your output."
                )
            if fname.lower().endswith(white_list_formats):
                yield root, fname


def _list_valid_filenames_in_directory(
    directory, white_list_formats, split, class_indices, follow_links
):
    """Lists paths of files in `subdir` with extensions in `white_list_formats`.

    Args:
        directory: absolute path to a directory containing the files to list.
            The directory name is used as class label
            and must be a key of `class_indices`.
        white_list_formats: set of strings containing allowed extensions for
            the files to be counted.
        split: tuple of floats (e.g. `(0.2, 0.6)`) to only take into
            account a certain fraction of files in each directory.
            E.g.: `segment=(0.6, 1.0)` would only account for last 40 percent
            of images in each directory.
        class_indices: dictionary mapping a class name to its index.
        follow_links: boolean, follow symbolic links to subdirectories.

    Returns:
         classes: a list of class indices
         filenames: the path of valid files in `directory`, relative from
             `directory`'s parent (e.g., if `directory` is "dataset/class1",
            the filenames will be
            `["class1/file1.jpg", "class1/file2.jpg", ...]`).
    """
    dirname = os.path.basename(directory)
    if split:
        all_files = list(
            _iter_valid_files(directory, white_list_formats, follow_links)
        )
        num_files = len(all_files)
        start, stop = int(split[0] * num_files), int(split[1] * num_files)
        valid_files = all_files[start:stop]
    else:
        valid_files = _iter_valid_files(
            directory, white_list_formats, follow_links
        )
    classes = []
    filenames = []
    for root, fname in valid_files:
        classes.append(class_indices[dirname])
        absolute_path = os.path.join(root, fname)
        relative_path = os.path.join(
            dirname, os.path.relpath(absolute_path, directory)
        )
        filenames.append(relative_path)

    return classes, filenames


def load_img(
    path,
    grayscale=False,
    color_mode="rgb",
    target_size=None,
    interpolation="nearest",
    keep_aspect_ratio=False,
    fold_resolution=0.5
):
    """Loads an image into PIL format.

    Usage:

    ```
    image = tf.keras.preprocessing.image.load_img(image_path)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = model.predict(input_arr)
    ```

    Args:
        path: Path to image file.
        grayscale: DEPRECATED use `color_mode="grayscale"`.
        color_mode: One of `"grayscale"`, `"rgb"`, `"rgba"`. Default: `"rgb"`.
          The desired image format.
        target_size: Either `None` (default to original size) or tuple of ints
          `(img_height, img_width)`.
        interpolation: Interpolation method used to resample the image if the
          target size is different from that of the loaded image. Supported
          methods are `"nearest"`, `"bilinear"`, and `"bicubic"`. If PIL version
          1.1.3 or newer is installed, `"lanczos"` is also supported. If PIL
          version 3.4.0 or newer is installed, `"box"` and `"hamming"` are also
          supported. By default, `"nearest"` is used.
        keep_aspect_ratio: Boolean, whether to resize images to a target
                size without aspect ratio distortion. The image is cropped in
                the center with target aspect ratio before resizing.

    Returns:
        A PIL Image instance.

    Raises:
        ImportError: if PIL is not available.
        ValueError: if interpolation method is not supported.
    """
    if grayscale:
        warnings.warn(
            "grayscale is deprecated. Please use " 'color_mode = "grayscale"'
        )
        color_mode = "grayscale"
    if pil_image is None:
        raise ImportError(
            "Could not import PIL.Image. " "The use of `load_img` requires PIL."
        )
    if isinstance(path, io.BytesIO):
        img = pil_image.open(path)
    elif isinstance(path, (pathlib.Path, bytes, str)):
        if isinstance(path, pathlib.Path):
            path = str(path.resolve())
        with open(path, "rb") as f:
            img = pil_image.open(io.BytesIO(f.read()))
    else:
        raise TypeError(
            "path should be path-like or io.BytesIO"
            ", not {}".format(type(path))
        )

    if color_mode == "grayscale":
        # if image is not already an 8-bit, 16-bit or 32-bit grayscale image
        # convert it to an 8-bit grayscale image.
        if img.mode not in ("L", "I;16", "I"):
            img = img.convert("L")
    elif color_mode == "rgba":
        if img.mode != "RGBA":
            img = img.convert("RGBA")
    elif color_mode == "rgb":
        if img.mode != "RGB":
            img = img.convert("RGB")
    else:
        raise ValueError('color_mode must be "grayscale", "rgb", or "rgba"')

    if fold_resolution != 1:
        width, height = img.size
        img = img.resize((int(width*fold_resolution), int(height*fold_resolution)))

    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img.size != width_height_tuple:
            if interpolation not in _PIL_INTERPOLATION_METHODS:
                raise ValueError(
                    "Invalid interpolation method {} specified. Supported "
                    "methods are {}".format(
                        interpolation,
                        ", ".join(_PIL_INTERPOLATION_METHODS.keys()),
                    )
                )
            resample = _PIL_INTERPOLATION_METHODS[interpolation]

            if keep_aspect_ratio:
                width, height = img.size
                target_width, target_height = width_height_tuple

                crop_height = (width * target_height) // target_width
                crop_width = (height * target_width) // target_height

                # Set back to input height / width
                # if crop_height / crop_width is not smaller.
                crop_height = min(height, crop_height)
                crop_width = min(width, crop_width)

                crop_box_hstart = (height - crop_height) // 2
                crop_box_wstart = (width - crop_width) // 2
                crop_box_wend = crop_box_wstart + crop_width
                crop_box_hend = crop_box_hstart + crop_height
                crop_box = [
                    crop_box_wstart,
                    crop_box_hstart,
                    crop_box_wend,
                    crop_box_hend,
                ]
                img = img.resize(width_height_tuple, resample, box=crop_box)
            else:
                img = img.resize(width_height_tuple, resample)
    return img


def img_to_array(img, data_format=None, dtype=None):
    """Converts a PIL Image instance to a Numpy array.

    Usage:

    ```python
    from PIL import Image
    img_data = np.random.random(size=(100, 100, 3))
    img = tf.keras.preprocessing.image.array_to_img(img_data)
    array = tf.keras.preprocessing.image.img_to_array(img)
    ```


    Args:
        img: Input PIL Image instance.
        data_format: Image data format, can be either `"channels_first"` or
          `"channels_last"`. Defaults to `None`, in which case the global
          setting `tf.keras.backend.image_data_format()` is used (unless you
          changed it, it defaults to `"channels_last"`).
        dtype: Dtype to use. Default to `None`, in which case the global setting
          `tf.keras.backend.floatx()` is used (unless you changed it, it
          defaults to `"float32"`).

    Returns:
        A 3D Numpy array.

    Raises:
        ValueError: if invalid `img` or `data_format` is passed.
    """

    if data_format is None:
        data_format = backend.image_data_format()
    if dtype is None:
        dtype = backend.floatx()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(f"Unknown data_format: {data_format}")
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=dtype)
    if len(x.shape) == 3:
        if data_format == "channels_first":
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == "channels_first":
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError(f"Unsupported image shape: {x.shape}")
    return x


def array_to_img(x, data_format=None, scale=True, dtype=None):
    """Converts a 3D Numpy array to a PIL Image instance.

    Usage:

    ```python
    from PIL import Image
    img = np.random.random(size=(100, 100, 3))
    pil_img = tf.keras.preprocessing.image.array_to_img(img)
    ```


    Args:
        x: Input data, in any form that can be converted to a Numpy array.
        data_format: Image data format, can be either `"channels_first"` or
          `"channels_last"`. Defaults to `None`, in which case the global
          setting `tf.keras.backend.image_data_format()` is used (unless you
          changed it, it defaults to `"channels_last"`).
        scale: Whether to rescale the image such that minimum and maximum values
          are 0 and 255 respectively. Defaults to `True`.
        dtype: Dtype to use. Default to `None`, in which case the global setting
          `tf.keras.backend.floatx()` is used (unless you changed it, it
          defaults to `"float32"`)

    Returns:
        A PIL Image instance.

    Raises:
        ImportError: if PIL is not available.
        ValueError: if invalid `x` or `data_format` is passed.
    """

    if data_format is None:
        data_format = backend.image_data_format()
    if dtype is None:
        dtype = backend.floatx()
    if pil_image is None:
        raise ImportError(
            "Could not import PIL.Image. "
            "The use of `array_to_img` requires PIL."
        )
    x = np.asarray(x, dtype=dtype)
    if x.ndim != 3:
        raise ValueError(
            "Expected image array to have rank 3 (single image). "
            f"Got array with shape: {x.shape}"
        )

    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(f"Invalid data_format: {data_format}")

    # Original Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but target PIL image has format (width, height, channel)
    if data_format == "channels_first":
        x = x.transpose(1, 2, 0)
    if scale:
        x = x - np.min(x)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    if x.shape[2] == 4:
        # RGBA
        return pil_image.fromarray(x.astype("uint8"), "RGBA")
    elif x.shape[2] == 3:
        # RGB
        return pil_image.fromarray(x.astype("uint8"), "RGB")
    elif x.shape[2] == 1:
        # grayscale
        if np.max(x) > 255:
            # 32-bit signed integer grayscale image. PIL mode "I"
            return pil_image.fromarray(x[:, :, 0].astype("int32"), "I")
        return pil_image.fromarray(x[:, :, 0].astype("uint8"), "L")
    else:
        raise ValueError(f"Unsupported channel number: {x.shape[2]}")

def apply_random_patch_gen(img, rand_x, rand_y, target_width, target_height):
    img_patch = np.ndarray((target_height, target_width, 1), dtype=np.float64)

    if img.ndim != 2:
        img = np.squeeze(img)

    img = img[rand_x : min(rand_x+target_width,img.shape[0]), rand_y : min(rand_y+target_width,img.shape[1])]

    #padding img to fit the network
    pad_x = (target_width - img.shape[0])/2
    pad_y = (target_height - img.shape[1])/2

    if int(pad_x) > pad_x or int(pad_x) < pad_x :
        if int(pad_y) > pad_y or int(pad_y) < pad_y :
            pad_x1 = int(np.floor(pad_x))
            pad_x2 = int(np.ceil(pad_x))
            pad_y1 = int(np.floor(pad_y))
            pad_y2 = int(np.ceil(pad_y))
        else:
            pad_x1 = int(np.floor(pad_x))
            pad_x2 = int(np.ceil(pad_x))
            pad_y1 = int(pad_y)
            pad_y2 = int(pad_y)

    else:
        if int(pad_y) > pad_y or int(pad_y) < pad_y :
            pad_x1 = int(pad_x)
            pad_x2 = int(pad_x)
            pad_y1 = int(np.floor(pad_y))
            pad_y2 = int(np.ceil(pad_y))
        else:
            pad_x1 = int(pad_x)
            pad_x2 = int(pad_x)
            pad_y1 = int(pad_y)
            pad_y2 = int(pad_y)

    img = np.pad(img, ((pad_x1, pad_x2), (pad_y1, pad_y2)),
                            mode='constant', constant_values=0.5)
    #change to a new type of padding next

    img = np.expand_dims(img, axis=-1)
    img_patch = img
    return img_patch
