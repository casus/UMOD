from src.unet7 import Unet
from tensorflow.keras.optimizers import Adam
from src.utils.losses import dice_bce_loss, dice_coeff

def load_unet_weights(parameters, checkpoint_file_path):
    model = Unet(
        parameters['target_height'],
        parameters['target_width'],
        filters=parameters['bottleneck_size'],
        nclasses=1,
        do=parameters['dropout'],
        l2=parameters['l2_regularization'],
        )
    model.load_weights(checkpoint_file_path)
    model.compile(optimizer=Adam(learning_rate=parameters['start_lr']),
                    loss=dice_bce_loss, metrics=[dice_coeff])

    return model
