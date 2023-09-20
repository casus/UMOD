import numpy as np

from src.keras_patch_generator.customImageDataGenerator import customImageDataGenerator

def image_mask_generator_from_npz(
    data_path: str,
    image_mask_datagen,
    batch_size,
    seed
    ):
    with np.load(data_path) as data:
        imgs = (data['all_images'] / 255.).astype('float32')
        imgs = np.where(imgs == 0, 0.5, imgs)
        masks = np.rint(data['all_masks'] / 255.).astype('float32')

    print(imgs.shape)
    print(masks.shape)

    image_mask_generator = image_mask_datagen.flow(
        x=imgs,
        y=masks,
        batch_size=batch_size,
        seed=seed,
    )

    return image_mask_generator

def image_mask_generator_from_directory(
    img_mask_gen_args, max_iter, thresh_obj_perc, target_width, target_height, image_tf, mask_tf,
    img_dir, mask_dir, img_rows, img_cols, batch_size_test, seed, resampling_number, fold_resolution
    ):
    img_mask_gen_args.update({
        "max_iter": max_iter,
        "thresh_obj_perc": thresh_obj_perc,
        "target_width": target_width,
        "target_height": target_height,
        "image_aug_check": image_tf,
        "mask_aug_check": mask_tf
    })
    image_mask_datagen = customImageDataGenerator(
        **img_mask_gen_args)

    image_mask_generator = image_mask_datagen.flow_from_directory(
        directory_x=img_dir,
        directory_y=mask_dir,
        target_size=(img_rows, img_cols),
        color_mode="grayscale",
        classes=None,
        class_mode=None,
        batch_size=batch_size_test,
        seed=seed,
        keep_aspect_ratio=True,
        resampling_number=resampling_number,
        fold_resolution=fold_resolution,
    )
    return image_mask_generator
