import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy


def dice_coeff(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.) -> tf.Tensor:
    """ Calculate dice coefficient.

    Args:
        y_true (tf.Tensor): Ground truth values. shape = [batch_size, d0, ..., dN]
        y_pred (tf.Tensor): Predicted values. shape = [batch_size, d0, ..., dN]
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        tf.Tensor: Dice coefficient value. shape = [batch_size, d0, ..., dN-1]
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.) -> tf.Tensor:
    """ Calculate dice loss.

    Args:
        y_true (tf.Tensor): Ground truth values. shape = [batch_size, d0, ..., dN]
        y_pred (tf.Tensor): Predicted values. shape = [batch_size, d0, ..., dN]
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        tf.Tensor: Dice loss value. shape = [batch_size, d0, ..., dN-1]
    """
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def jaccard_distance(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: int = 1) -> tf.Tensor:
    """ Calculate jaccard distance.

    Args:
        y_true (tf.Tensor): Ground truth values. shape = [batch_size, d0, ..., dN]
        y_pred (tf.Tensor): Predicted values. shape = [batch_size, d0, ..., dN]
        smooth (int): Smoothing factor to avoid division by zero.

    Returns:
        tf.Tensor: Jaccard distance value. shape = [batch_size, d0, ..., dN-1]
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    return (intersection + smooth) / (sum_ - intersection + smooth)

def dice_bce_loss(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.) -> tf.Tensor:
    """ Calculate combined dice and bce losses.

    Args:
        y_true (tf.Tensor): Ground truth values. shape = [batch_size, d0, ..., dN]
        y_pred (tf.Tensor): Predicted values. shape = [batch_size, d0, ..., dN]
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        tf.Tensor: Dice bce loss value. shape = [batch_size, d0, ..., dN-1]
    """
    loss = (binary_crossentropy(y_true, y_pred)
    + (1*dice_loss(y_true, y_pred, smooth=smooth)))

    return loss
