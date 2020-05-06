import tensorflow as tf
# import keras.backend as K
# random_seed = 24
# tf.set_random_seed(random_seed)
@tf.function
def get_mode_mask(modes):
    mode_equal = tf.equal(tf.expand_dims(modes, 0), tf.expand_dims(modes, 1))
    i_equal_j = tf.expand_dims(mode_equal, 2)
    i_equal_k = tf.expand_dims(mode_equal, 1)

    # ------------- Atleast two As
    # j_equal_k = tf.expand_dims(mode_equal, 0)
    # ij = tf.logical_and(i_equal_j,tf.equal(tf.expand_dims(modes,1),1))
    # ik = tf.logical_and(i_equal_k,tf.equal(tf.expand_dims(modes,0),1))
    # jk = tf.logical_and(j_equal_k,tf.equal(modes,1))
    # mode_mask = tf.logical_and(distinct_indices,tf.logical_or(jk,tf.logical_or(ij,ij)))

    #------------ Negative imgnet anchor ---------------------------
    # mode_equal = tf.equal(tf.expand_dims(modes, 0), tf.expand_dims(modes, 1))
    # i_equal_j = tf.expand_dims(mode_equal, 2)
    # i_equal_k = tf.expand_dims(mode_equal, 1)
    # ij = tf.logical_and(i_equal_j,tf.equal(tf.expand_dims(modes,1),1))
    # mode_mask = tf.logical_and(ij,tf.logical_not(i_equal_k))
    # mode_mask = tf.logical_and(distinct_indices,mode_mask)

    #-------------- Seen and Unseen ------------------------------
    mode_equal = tf.equal(tf.expand_dims(modes, 0), tf.expand_dims(modes, 1))
    i_equal_j = tf.expand_dims(mode_equal, 2)
    i_equal_k = tf.expand_dims(mode_equal, 1)
    ij = tf.logical_and(i_equal_j,tf.equal(tf.expand_dims(modes,1),1))
    ijk = tf.logical_and(ij, i_equal_k)

    k_equal_1 = tf.logical_and(tf.cast(tf.ones(shape=tf.shape(ijk)), tf.bool), tf.expand_dims(tf.equal(modes,1),0))
    mode_mask = tf.logical_and(tf.logical_not(ijk), k_equal_1)
    
    #------------------------------------------------------------------
    # valid_modes = tf.logical_and(tf.logical_not(i_equal_j), tf.logical_not(i_equal_k))
    # mode_mask = tf.logical_and(distinct_indices,valid_modes)
    
    # ------------- All except (I,I,I)
    # valid_modes = tf.logical_and(i_equal_j,i_equal_k)
    # mode_mask = tf.logical_not(tf.logical_and(valid_modes,tf.expand_dims(tf.expand_dims(tf.equal(modes,0),1),1)))

    # mode_mask = tf.logical_and(distinct_indices, mode_mask)
    return mode_mask

@tf.function
def get_label_mask(labels):

    a = tf.eye(tf.shape(labels)[0])
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)


    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    # Combine the two masks
    mask = tf.logical_and(distinct_indices, valid_labels)
    return mask

@tf.function
def _pairwise_distances(embeddings, squared=True):
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    square_norm = tf.linalg.diag_part(dot_product)

    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        mask = tf.cast(tf.equal(distances, 0.0),tf.float32)
        distances = distances + mask * 1e-16
        distances = tf.sqrt(distances)
        distances = distances * (1.0 - mask)
    return distances

@tf.function
def batch_all_triplet_loss(data):
    # shape (batch_size, batch_size, 1)
    labels, modes, embeddings = data
    margin=0.8
    squared=True
    labels = tf.reshape(labels,[-1])
    modes = tf.reshape(modes,[-1])
    pairwise_dist = _pairwise_distances(embeddings)
    anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
    assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
    # shape (batch_size, 1, batch_size)
    anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)
    assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin
    mask, mode_mask = get_label_mask(labels), get_mode_mask(modes)
    mask = tf.cast(mask, tf.float32)
    mode_mask = tf.cast(mode_mask, tf.float32)
    mask = tf.multiply(mask,mode_mask)
    triplet_loss = tf.multiply(mask, triplet_loss)

    triplet_loss = tf.maximum(triplet_loss, 0.0)

    valid_triplets = tf.cast(tf.greater(triplet_loss, 1e-16),tf.float32)
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    num_valid_triplets = tf.reduce_sum(mask)

    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)
    triplet_loss = tf.reduce_sum(triplet_loss)/ (num_positive_triplets + 1e-16)
    # print(triplet_loss.eval())

    return triplet_loss

@tf.function
def identity_loss(y_true, y_pred):
    return tf.keras.backend.mean(y_pred)