from __future__ import division
from keras import backend as K
import tensorflow as tf
# from gaussian_prior import LearningPrior

def null_loss(y_true, y_pred):
    return K.variable(0.0) #,dtype=float32


    
# KL-Divergence Loss
def kl_divergence(y_true, y_pred):
    # max_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(y_pred, axis=1), axis=1)), shape_r_out, axis=1)), shape_c_out, axis=2)
    max_y_pred = K.expand_dims(K.repeat_elements(
        K.expand_dims(K.repeat_elements(K.expand_dims(K.max(y_pred, axis=[2, 3, 4])), y_pred.shape[2], axis=2)),
        y_pred.shape[3], axis=3))
    y_pred /= (max_y_pred + K.epsilon())

    max_y_true = K.expand_dims(K.repeat_elements(
        K.expand_dims(K.repeat_elements(K.expand_dims(K.max(y_true, axis=[2, 3, 4])), y_pred.shape[2], axis=2)),
        y_pred.shape[3], axis=3))
    y_bool = K.cast(K.greater(max_y_true, 0.1), 'float32')

    sum_y_true = K.expand_dims(K.repeat_elements(
        K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(y_true, axis=[2, 3, 4])), y_pred.shape[2], axis=2)),
        y_pred.shape[3], axis=3))
    sum_y_pred = K.expand_dims(K.repeat_elements(
        K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(y_pred, axis=[2, 3, 4])), y_pred.shape[2], axis=2)),
        y_pred.shape[3], axis=3))
    y_true /= (sum_y_true + K.epsilon())
    y_pred /= (sum_y_pred + K.epsilon())
    return K.sum(y_bool * y_true * K.log((y_true / (y_pred + K.epsilon())) + K.epsilon()))


# Correlation Coefficient Loss
def correlation_coefficient(y_true, y_pred):
    max_y_pred = K.expand_dims(K.repeat_elements(
        K.expand_dims(K.repeat_elements(K.expand_dims(K.max(y_pred, axis=[2, 3, 4])), y_pred.shape[2], axis=2)),
        y_pred.shape[3], axis=3))
    y_pred /= (max_y_pred + K.epsilon())

    # max_y_true = K.expand_dims(K.repeat_elements(
    #     K.expand_dims(K.repeat_elements(K.expand_dims(K.max(y_true, axis=[2, 3, 4])), shape_r_out, axis=2)),
    #     shape_c_out, axis=3))
    max_y_true = K.max(y_true, axis=[2, 3, 4])
    y_bool = K.cast(K.greater(max_y_true, 0.1), 'float32')

    sum_y_true = K.expand_dims(K.repeat_elements(
        K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(y_true, axis=[2, 3, 4])), y_pred.shape[2], axis=2)),
        y_pred.shape[3], axis=3))
    sum_y_pred = K.expand_dims(K.repeat_elements(
        K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(y_pred, axis=[2, 3, 4])), y_pred.shape[2], axis=2)),
        y_pred.shape[3], axis=3))

    y_true /= (sum_y_true + K.epsilon())
    y_pred /= (sum_y_pred + K.epsilon())

    N = y_pred._shape_as_list()[2] * y_pred._shape_as_list()[3]
    sum_prod = K.sum(y_true * y_pred, axis=[2, 3, 4])
    sum_x = K.sum(y_true, axis=[2, 3, 4])
    sum_y = K.sum(y_pred, axis=[2, 3, 4])
    sum_x_square = K.sum(K.square(y_true), axis=[2, 3, 4])+ K.epsilon()
    sum_y_square = K.sum(K.square(y_pred), axis=[2, 3, 4])+ K.epsilon()

    num = sum_prod - ((sum_x * sum_y) / N)
    den = K.sqrt((sum_x_square - K.square(sum_x) / N) * (sum_y_square - K.square(sum_y) / N))

    return -K.sum(y_bool*(num/(den + K.epsilon())))#


# Normalized Scanpath Saliency Loss
def nss(y_true, y_pred):
    max_y_pred = K.expand_dims(K.repeat_elements(
        K.expand_dims(K.repeat_elements(K.expand_dims(K.max(y_pred, axis=[2, 3, 4])), y_pred.shape[2], axis=2)),
        y_pred.shape[3], axis=3))
    y_pred /= (max_y_pred + K.epsilon())
    # y_pred_flatten = K.batch_flatten(y_pred)

    # max_y_true = K.expand_dims(K.repeat_elements(
    #     K.expand_dims(K.repeat_elements(K.expand_dims(K.max(y_true, axis=[2, 3, 4])), shape_r_out, axis=2)),
    #     shape_c_out, axis=3))
    max_y_true = K.max(y_true, axis=[2, 3, 4])
    y_bool = K.cast(K.greater(max_y_true, 0.1), 'float32')

    y_mean = K.mean(y_pred, axis=[2, 3, 4])
    y_mean = K.expand_dims(K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(y_mean),
                           y_pred.shape[2], axis=2)), y_pred.shape[3], axis=3))

    y_std = K.std(y_pred, axis=[2, 3, 4])
    y_std = K.expand_dims(K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(y_std),
                           y_pred.shape[2], axis=2)), y_pred.shape[3], axis=3))

    y_pred = (y_pred - y_mean) / (y_std + K.epsilon())

    return -K.sum(y_bool*((K.sum(y_true * y_pred, axis=[2, 3, 4])) / (K.sum(y_true, axis=[2, 3, 4])+K.epsilon())))


# Similarity
def sim(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
    y_true = tf.clip_by_value(y_true, K.epsilon(), 1 - K.epsilon())

    max_y_pred = K.expand_dims(K.repeat_elements(
        K.expand_dims(K.repeat_elements(K.expand_dims(K.max(y_pred, axis=[2, 3, 4])), y_pred.shape[2], axis=2)),
        y_pred.shape[3], axis=3))
        
    max_y_true = K.max(y_true, axis=[2, 3, 4])    
    y_bool = K.cast(K.greater(max_y_true, 0.1), 'float32')    
    y_pred /= (max_y_pred + K.epsilon())

    sum_y_true = K.expand_dims(K.repeat_elements(
        K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(y_true, axis=[2, 3, 4])), y_pred.shape[2], axis=2)),
        y_pred.shape[3], axis=3))
    sum_y_pred = K.expand_dims(K.repeat_elements(
        K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(y_pred, axis=[2, 3, 4])), y_pred.shape[2], axis=2)),
        y_pred.shape[3], axis=3))

    y_true /= (sum_y_true + K.epsilon())
    y_pred /= (sum_y_pred + K.epsilon())

    return -y_bool*K.sum(K.minimum(y_true, y_pred), axis=[2, 3, 4])
