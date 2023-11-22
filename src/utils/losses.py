import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy

def iou(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """ Calculate intersection over union.

    Args:
        y_true (tf.Tensor): Ground truth values. shape = [batch_size, d0, ..., dN]
        y_pred (tf.Tensor): Predicted values. shape = [batch_size, d0, ..., dN]

    Returns:
        tf.Tensor: IoU value. shape = [batch_size, d0, ..., dN-1]
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = intersection / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score

def precision(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """ Calculate precision.

    Args:
        y_true (tf.Tensor): Ground truth values. shape = [batch_size, d0, ..., dN]
        y_pred (tf.Tensor): Predicted values. shape = [batch_size, d0, ..., dN]

    Returns:
        tf.Tensor: Precision value. shape = [batch_size, d0, ..., dN-1]
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    tp = K.sum(y_true_f * y_pred_f)
    t2 = tf.where( tf.equal( 0, y_true_f ), -1 * tf.ones_like( y_true_f ), y_true_f )
    temp = tf.compat.v1.Variable(np.zeros(shape=y_true_f.shape, dtype=np.float32))
    temp.assign(t2)
    t3 = tf.where( tf.equal( 1, temp ), 0 * tf.ones_like( temp ), temp )
    temp.assign(t3)
    t4 = tf.where( tf.equal( -1, temp ), 1 * tf.ones_like( temp ), temp )
    temp.assign(t4)

    fp = K.sum(temp * y_pred_f)
    score = tp/(tp + fp)
    return score

def recall(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """ Calculate recall.

    Args:
        y_true (tf.Tensor): Ground truth values. shape = [batch_size, d0, ..., dN]
        y_pred (tf.Tensor): Predicted values. shape = [batch_size, d0, ..., dN]

    Returns:
        tf.Tensor: Recall value. shape = [batch_size, d0, ..., dN-1]
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    tp = K.sum(y_true_f * y_pred_f)
    t2 = tf.where( tf.equal( 0, y_pred_f ), -1 * tf.ones_like( y_pred_f ), y_pred_f )
    temp = tf.compat.v1.Variable(np.zeros(shape=y_pred_f.shape, dtype=np.float32))
    temp.assign(t2)
    t3 = tf.where( tf.equal( 1, temp ), 0 * tf.ones_like( temp ), temp )
    temp.assign(t3)
    t4 = tf.where( tf.equal( -1, temp ), 1 * tf.ones_like( temp ), temp )
    temp.assign(t4)

    fn = K.sum(y_true_f * temp)
    score = tp/(tp + fn)
    return score

def auc(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """ Calculate area under receiver operating characteristic curve.

    Args:
        y_true (tf.Tensor): Ground truth values. shape = [batch_size, d0, ..., dN]
        y_pred (tf.Tensor): Predicted values. shape = [batch_size, d0, ..., dN]

    Returns:
        tf.Tensor: ROC-AUC value. shape = [batch_size, d0, ..., dN-1]
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return roc_auc_score(y_true_f,y_pred_f)

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
