import tensorflow as tf
def get_weight(keypoints, mask=None, ng_w=0.01, ps_w=1.0):
    is_background = tf.equal(keypoints, 0)
    ones = tf.to_float(tf.ones_like(is_background))
    weights = tf.where(is_background, ones * ng_w, ones*ps_w)
    # if mask is not None:
    #     weights *= tf.to_float(mask)

    return weights