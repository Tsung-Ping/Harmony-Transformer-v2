import tensorflow as tf # version=1.8.0
import math
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.contrib.rnn import LSTMCell, DropoutWrapper
import numpy as np

# Disables AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_absolute_position_encoding(length, hidden_size, min_timescale=1.0, max_timescale=1e4, start_index=0):
    '''https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py'''
    """Return positional encoding.
    Calculates the position encoding as a mix of sine and cosine functions with
    geometrically increasing wavelengths.
    Defined and formulized in Attention is All You Need, section 3.5.
    Args:
      length: Sequence length.
      hidden_size: Size of the
      min_timescale: Minimum scale that will be applied at each position
      max_timescale: Maximum scale that will be applied at each position
    Returns:
      Tensor with shape [length, hidden_size]
    """
    position = tf.cast(tf.range(length) + start_index, tf.float32)
    num_timescales = hidden_size // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / tf.maximum(tf.cast(num_timescales, tf.float32) - 1, 1))
    inv_timescales = min_timescale * tf.exp(tf.cast(tf.range(num_timescales), tf.float32) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    # Please note that this slightly differs from the published paper.
    # See a discussion here: https://github.com/tensorflow/tensor2tensor/pull/177
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(hidden_size, 2)]])
    signal = tf.reshape(signal, [1, length, hidden_size])
    return signal

def get_relative_position_encoding(n_steps, n_units=128, max_dist=10, name='relative_position_encodings'):
    '''https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py'''
    def _generate_relative_positions_matrix(length_q, length_k, max_relative_position):
        """Generates matrix of relative positions between inputs."""
        if length_q == length_k:
            range_vec_q = range_vec_k = tf.range(length_q)
        else:
            range_vec_k = tf.range(length_k)
            range_vec_q = range_vec_k[-length_q:]
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = tf.clip_by_value(distance_mat, -max_relative_position, max_relative_position)
        # Shift values to be >= 0. Each integer still uniquely identifies a relative position difference.
        final_mat = distance_mat_clipped + max_relative_position
        return final_mat

    with tf.variable_scope(name):
        relative_positions_matrix = _generate_relative_positions_matrix(n_steps, n_steps, max_dist)
        vocab_size = max_dist * 2 + 1
        # Generates embedding for each relative position of dimension depth.
        # embeddings_table = tf.get_variable("rel_pos_embeddings", [vocab_size, n_units]) # learnable pos embeddings
        embeddings_table = tf.squeeze(get_absolute_position_encoding(vocab_size, n_units), axis=0) # absolute pos embeddings, [vocab_size, depth]
        embeddings = tf.gather(embeddings_table, relative_positions_matrix)
        return embeddings # [n_steps, n_steps, n_units]

def normalize(inputs, axis=[-1], epsilon=1e-6, scope="ln", reuse=None):
    '''https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py'''
    '''Applies layer normalization.
    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
        beta = tf.get_variable("beta_bias", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        mean = tf.reduce_mean(inputs, axis=axis, keepdims=True)
        variance = tf.reduce_mean(tf.squared_difference(inputs, mean), axis=axis, keepdims=True)
        normalized = (inputs - mean) * tf.rsqrt(variance + epsilon)
        outputs = gamma * normalized + beta
    return outputs

def MHA(queries, keys, values=None, n_units=None, n_heads=8, key_mask=None, forward=False, backward=False,
        relative_position=False, max_dist=4, positional_attention=False, attention_map=False,
        dropout_rate=0, is_training=True, scope="MHA", reuse=None):
    '''Applies multihead attention.
    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      n_units: A scalar. Attentio +-n size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      n_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for n_units
        if n_units is None:
            n_units = queries.get_shape().as_list[-1]

        # Absolute positional encoding
        q = queries
        k = keys
        v = keys if values is None else values

        # Linear projections
        Q = tf.layers.dense(q, n_units, name='dense_Q')
        K = tf.layers.dense(k, n_units, name='dense_K')
        V = tf.layers.dense(v, n_units, name='dense_V')
        Q = tf.layers.dropout(Q, rate=dropout_rate, training=is_training)
        K = tf.layers.dropout(K, rate=dropout_rate, training=is_training)
        V = tf.layers.dropout(V, rate=dropout_rate, training=is_training)

        # Compute attention matrix
        if not relative_position:
            # Split and concat (multihead)
            Q_ = tf.concat(tf.split(Q, n_heads, axis=2), axis=0) # [h*N, T_q, C/h]
            K_ = tf.concat(tf.split(K, n_heads, axis=2), axis=0) # [h*N, T_k, C/h]
            V_ = tf.concat(tf.split(V, n_heads, axis=2), axis=0) # [h*N, T_k, C/h]

            # Multiplication (Compute dot similarity)
            outputs = tf.matmul(Q_, K_, transpose_b=True) # [h*N, T_q, T_k]

        else: # Using relative position encodings
            """only for self attention"""
            '''see Transformer-XL: Attentive language models beyond a fixed-length context'''
            # Split and concat (multihead)
            K_ = tf.concat(tf.split(K, n_heads, axis=2), axis=0) # [h*N, T_k, C/h]
            V_ = tf.concat(tf.split(V, n_heads, axis=2), axis=0) # [h*N, T_k, C/h]
            R_u = tf.get_variable('pe_u', dtype=tf.float32, shape=[n_units], initializer=tf.zeros_initializer()) # [1, 1, C]
            R_v = tf.get_variable('pe_v', dtype=tf.float32, shape=[n_units], initializer=tf.zeros_initializer()) # [1, 1, C]
            ac = Q + R_u # [N, T_q, C]
            ac = tf.concat(tf.split(ac, n_heads, axis=2), axis=0) # [h*N, T_q, C/h]
            ac = tf.matmul(ac, K_, transpose_b=True) # [h*N, T_q, T_k]

            # Get relative positional encodings
            _, T_q, _ = Q.get_shape().as_list()
            _, T_k, _ = K.get_shape().as_list()
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            rel_pe = get_relative_position_encoding(n_steps=T_q, n_units=n_units, max_dist=max_dist) # relative positional encodings, [T_q, T_k, C]
            rel_pe = tf.layers.dense(rel_pe, n_units, name='dense_rel_pe') # [T_q, T_k, C]
            rel_pe = tf.layers.dropout(rel_pe, rate=dropout_rate, training=is_training)
            rel_pe = tf.concat(tf.split(rel_pe[None, :, :, :], n_heads, axis=3), axis=0) # [h, T_q, T_k, C/h]
            bd = Q + R_v # [N, T_q, C]
            bd = tf.concat(tf.split(bd[None, :, :, :], n_heads, axis=3), axis=0) # [h, N, T_q, C/h]
            bd = tf.transpose(bd, [0, 2, 1, 3]) # [h, T_q, N, C/h]
            bd = tf.matmul(bd, rel_pe, transpose_b=True) # [h, T_q, N, T_k]
            bd = tf.transpose(bd, [0, 2, 1, 3]) # [h, N, T_q, T_k]
            bd = tf.reshape(bd, [-1, T_q, T_k]) # [h*N, T_q, T_k]
            outputs = ac + bd
            # rel_pe = get_relative_position_encoding(n_steps=T_q, n_units=n_units, max_dist=max_dist) # relative positional encodings, [T, T, C]
            # rel_pe = tf.layers.dense(rel_pe, n_units//n_heads, name='dense_rel_pe') # [T_q, T_k, C/h]
            # rel_pe = tf.layers.dropout(rel_pe, rate=dropout_rate, training=is_training)
            # bd = Q + R_v # [N, T_q, C]
            # bd = tf.concat(tf.split(bd, n_heads, axis=2), axis=0) # [h*N, T_q, C/h]
            # bd = tf.transpose(bd, [1,0,2]) # [T_q, h*N, C/h]
            # bd = tf.matmul(bd, rel_pe, transpose_b=True) # [T_q, h*N, T_k]
            # bd = tf.transpose(bd, [1,0,2]) # [h*N, T_q, T_k]
            # outputs = ac + bd

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1]**0.5)

        # Key Masking
        if key_mask is not None:
            key_mask = tf.tile(tf.expand_dims(key_mask, 1), [n_heads, tf.shape(queries)[1], 1]) # [h*N, T_q, T_k]
            paddings = tf.ones_like(outputs) * (-2**32 +1) # set padded cells to a value close to -Infinity, so that their contributions are just negligible.
            outputs = tf.where(tf.equal(key_mask, 0), paddings, outputs) # [h*N, T_q, T_k]

        # Foward/Backward Masking
        """only for self attention"""
        if forward:
            diag_vals = tf.ones_like(outputs[0, :, :]) # [T_q, T_k]
            triu = tf.linalg.band_part(diag_vals, 0, -1) # upper triangular mask, [T_q, T_k]
            triu = tf.tile(tf.expand_dims(triu, 0), [tf.shape(outputs)[0], 1, 1]) # [h*N, T_q, T_k]
            paddings = tf.ones_like(triu) * (-2**32+1)
            outputs = tf.where(tf.equal(triu, 0), paddings, outputs) # [h*N, T_q, T_k]

        elif backward:
            diag_vals = tf.ones_like(outputs[0, :, :]) # [T_q, T_k]
            tril = tf.linalg.band_part(diag_vals, -1, 0) # lower triangular mask, [T_q, T_k]
            tril = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # [h*N, T_q, T_k]
            paddings = tf.ones_like(tril) * (-2**32+1)
            outputs = tf.where(tf.equal(tril, 0), paddings, outputs) # [h*N, T_q, T_k]

        # Activation
        outputs = tf.nn.softmax(outputs, axis=2) # softmax(QK^T/sqrt(k_d)), [h*N, T_q, T_k]
        if attention_map:
            attn_map = tf.concat(tf.split(outputs, n_heads, axis=0), axis=2) # [N, T_q, h*T_k]

        # Weighted sum
        outputs = tf.matmul(outputs, V_) # softmax(QK^T/sqrt(k_d))V, [h*N, T_q, C/h]

        # Restore shape
        outputs = tf.concat(tf.split(outputs, n_heads, axis=0), axis=2) # Concat(head 1 , ..., head h ), [N, T_q, C]

        # Output projection
        outputs = tf.layers.dense(outputs, n_units, name='dense_O') # Concat(head 1 , ..., head h )W_O
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=is_training) # [N, T_q, C]

        # Residual connection
        if not positional_attention:
            outputs += queries
        else:
            outputs += values

        # Normalization
        outputs = normalize(outputs) # [N, T_q, C]

    if not attention_map:
        return outputs
    else:
        return outputs, attn_map

def FFN(inputs, n_units=[512, 128], kernel_size=1, activation_function=tf.nn.relu, dropout_rate=0.0, is_training=False, scope="FFN", reuse=None):
    '''Point-wise feed forward net.
    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      n_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        outputs = tf.layers.conv1d(inputs=inputs, filters=n_units[0], kernel_size=kernel_size, activation=activation_function, use_bias=True, padding='same')
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=is_training)
        outputs = tf.layers.conv1d(inputs=outputs, filters=n_units[1], kernel_size=kernel_size, activation=None, use_bias=True, padding='same')
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=is_training)
        # Residual connection
        outputs += inputs
        # Normalization
        outputs = normalize(outputs)
    return outputs

def convFFN(inputs, n_units=[128, 128], activation_function=tf.nn.relu, dropout_rate=0, is_training=True, scope="convFFN", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        outputs = tf.layers.conv1d(inputs=inputs, filters=n_units[0], kernel_size=3, activation=activation_function, use_bias=True, padding='same')
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=is_training)
        outputs = tf.layers.conv1d(inputs=outputs, filters=n_units[1], kernel_size=3, activation=activation_function, use_bias=True, padding='same')
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=is_training)
        # Residual connection
        outputs += inputs
        # Normalization
        outputs = normalize(outputs)
    return outputs

def binaryRound(x, cast_to_int=False):
    """
    Rounds a tensor whose values are in [0,1] to a tensor with values in {0, 1},
    using the straight through estimator for the gradient.
    """
    g = tf.get_default_graph()

    with ops.name_scope("BinaryRound") as name:
        # with g.gradient_override_map({"Round": "Identity"}):
        #     return tf.round(x, name=name)
        if cast_to_int:
            with g.gradient_override_map({"Round": "Identity", "Cast": "Identity"}):
                return tf.cast(tf.round(x), tf.int32, name=name)
        else:
            with g.gradient_override_map({"Round": "Identity"}):
                return tf.round(x, name=name)

def chord_block_compression(hidden_states, chord_changes, compression='mean'):
    """compress hidden states according to chord changes"""
    if compression not in ['mean', 'sum']:
        print('Invalid compression method.')
        exit(1)

    block_ids = tf.cumsum(chord_changes, axis=1)
    change_at_start = tf.cast(tf.equal(chord_changes[:, 0], 1), tf.int32)
    block_ids = block_ids - (tf.ones_like(block_ids) * change_at_start[:, tf.newaxis]) # set 1st id to 0
    num_blocks = tf.reduce_max(block_ids, axis=1) + 1 # batched number of blocks
    max_steps = tf.reduce_max(num_blocks) # max number of blocks

    if compression == 'mean':
        segment_compress_and_pad = lambda x: tf.pad(tf.segment_mean(data=x[0], segment_ids=x[1]), paddings=[[0, max_steps - x[2]], [0, 0]], constant_values=0.0)
    else:  # 'compression == sum'
        segment_compress_and_pad = lambda x: tf.pad(tf.segment_sum(data=x[0], segment_ids=x[1]), paddings=[[0, max_steps - x[2]], [0, 0]], constant_values=0.0)

    chord_blocks = tf.map_fn(segment_compress_and_pad, (hidden_states, block_ids, num_blocks), dtype=tf.float32)

    return chord_blocks, block_ids, num_blocks

def decode_compressed_sequences(compressed_sequences, block_ids):
    # Decode chord sequences according to chords_pred and block_ids
    gather_chords = lambda x: tf.gather(params=x[0], indices=x[1])
    chords_decode = tf.map_fn(gather_chords, (compressed_sequences, block_ids), dtype=compressed_sequences.dtype)
    return chords_decode

def BTC(x, source_mask, dropout_rate, is_training, hyperparameters):
    '''Bi-directional Transformer for Chord Recognition (BTC)'''
    input = x
    with tf.variable_scope("encoder_input_embedding"):
        input_embed = tf.layers.dense(input, hyperparameters.input_embed_size) # [n_batches, n_steps, input_embedding_size]

    with tf.name_scope("encoder_positional_encoding"):
        input_embed += get_absolute_position_encoding(tf.shape(input_embed)[1], tf.shape(input_embed)[2])
        input_embed = tf.layers.dropout(input_embed, rate=dropout_rate, training=is_training)

    with tf.variable_scope("forward_encoding"):
        input_embed_fw = normalize(input_embed)
        for i in range(hyperparameters.n_layers):
            with tf.variable_scope("layer_{}".format(i)):
                # Multihead Attention (self-attention)
                input_embed_fw = MHA(queries=input_embed_fw,
                                     keys=input_embed_fw,
                                     n_units=hyperparameters.input_embed_size,
                                     n_heads=hyperparameters.n_heads,
                                     key_mask=source_mask,
                                     forward=True,
                                     dropout_rate=dropout_rate,
                                     is_training=is_training,
                                     scope="foward_self_attention")
                # Feed Forward
                input_embed_fw = convFFN(input_embed_fw, n_units=[hyperparameters.input_embed_size, hyperparameters.input_embed_size], dropout_rate=dropout_rate, is_training=is_training)

    with tf.variable_scope("backward_encoding"):
        input_embed_bw = normalize(input_embed)
        for i in range(hyperparameters.n_layers):
            with tf.variable_scope("layer_{}".format(i)):
                # Multihead Attention (self-attention)
                input_embed_bw = MHA(queries=input_embed_bw,
                                     keys=input_embed_bw,
                                     n_units=hyperparameters.input_embed_size,
                                     n_heads=hyperparameters.n_heads,
                                     key_mask=source_mask,
                                     backward=True,
                                     dropout_rate=dropout_rate,
                                     is_training=is_training,
                                     scope="backward_self_attention")
                # Feed Forward
                input_embed_bw = convFFN(input_embed_bw, n_units=[hyperparameters.input_embed_size, hyperparameters.input_embed_size], dropout_rate=dropout_rate, is_training=is_training)


    input_embed = tf.concat([input_embed_fw, input_embed_bw], axis=2)
    input_embed = tf.layers.dense(input_embed, 2*hyperparameters.input_embed_size)
    input_embed = normalize(input_embed)
    return input_embed

def HT(x, source_mask, target_mask, slope, dropout_rate, is_training, hyperparameters):
    input = x # [n_batches, n_steps, n_features]
    with tf.variable_scope("encoder_input_embedding"):
        enc_input_embed = tf.layers.dense(input, hyperparameters.input_embed_size) # [n_batches, n_steps, input_embedding_size]

    with tf.name_scope("encoder_positional_encoding"):
        enc_input_embed += get_absolute_position_encoding(tf.shape(enc_input_embed)[1], tf.shape(enc_input_embed)[2])
        enc_input_embed = tf.layers.dropout(enc_input_embed, rate=dropout_rate, training=is_training)

    with tf.variable_scope("encoder"):
        enc_weights = tf.nn.softmax(tf.get_variable('enc_weights_of_layers', dtype=tf.float32, shape=[hyperparameters.n_layers + 1], initializer=tf.initializers.zeros())) # [n_layers]
        enc_weighted_hidden = enc_weights[0] * enc_input_embed
        for i in range(1, hyperparameters.n_layers + 1):
            with tf.variable_scope("layer_{}".format(i)):
                # Multihead Attention (self-attention)
                enc_input_embed = MHA(queries=enc_input_embed,
                                      keys=enc_input_embed,
                                      n_units=hyperparameters.input_embed_size,
                                      n_heads=hyperparameters.n_heads,
                                      key_mask=source_mask,
                                      dropout_rate=dropout_rate,
                                      is_training=is_training,
                                      scope="enc_self_attention")
                # Feed Forward
                enc_input_embed = FFN(enc_input_embed, n_units=[hyperparameters.input_embed_size*4, hyperparameters.input_embed_size], dropout_rate=dropout_rate, is_training=is_training)
                # Weighted sum of hidden states
                enc_weighted_hidden += (enc_weights[i] * enc_input_embed)
        enc_input_embed = enc_weighted_hidden # [n_batches, n_steps, input_embedding_size]

    with tf.name_scope("chord_change_prediction"):
        chord_change_logits = tf.squeeze(tf.layers.dense(enc_input_embed, 1), axis=-1) # [n_batches, n_steps]
        chord_change_prob = tf.sigmoid(slope*chord_change_logits) # [n_batches, n_steps]
        chord_change_prediction = binaryRound(chord_change_prob, cast_to_int=True) # Binarization of chord change prediction, [n_batches, n_steps]

    with tf.variable_scope("decoder_input_embedding"):
        dec_input_embed = tf.layers.dense(input, hyperparameters.input_embed_size) # [n_batches, n_steps, input_embed_size]

    with tf.variable_scope("input_embedding_regionalization"):
        dec_input_embed_reg, block_ids, num_blocks = chord_block_compression(dec_input_embed, chord_change_prediction)
        dec_input_embed_reg = decode_compressed_sequences(dec_input_embed_reg, block_ids) # shape = [n_batches, n_steps, input_embed_size]
        dec_input_embed_reg.set_shape([None, hyperparameters.n_steps, hyperparameters.input_embed_size])
        dec_input_embed = dec_input_embed + dec_input_embed_reg + enc_input_embed # [n_batches, input_embed_size]

    with tf.name_scope("decoder_positional_encoding"):
        dec_input_embed += get_absolute_position_encoding(tf.shape(dec_input_embed)[1], tf.shape(dec_input_embed)[2])
        dec_input_embed = tf.layers.dropout(dec_input_embed, rate=dropout_rate, training=is_training)

    with tf.variable_scope("decoder"):
        dec_weights = tf.nn.softmax(tf.get_variable('dec_weights_of_layers', dtype=tf.float32, shape=[hyperparameters.n_layers + 1], initializer=tf.initializers.zeros())) # [n_layers]
        dec_weighted_hidden = dec_weights[0] * dec_input_embed
        for i in range(1, hyperparameters.n_layers + 1):
            with tf.variable_scope("layer_{}".format(i)):
                # Multihead Attention (self-attention)
                dec_input_embed = MHA(queries=dec_input_embed,
                                      keys=dec_input_embed,
                                      n_units=hyperparameters.input_embed_size,
                                      n_heads=hyperparameters.n_heads,
                                      key_mask=target_mask,
                                      dropout_rate=dropout_rate,
                                      is_training=is_training,
                                      scope="dec_self_attention")
                # Multihead Attention (seq2seq attention)
                dec_input_embed = MHA(queries=dec_input_embed,
                                      keys=enc_input_embed,
                                      n_units=hyperparameters.input_embed_size,
                                      n_heads=hyperparameters.n_heads,
                                      key_mask=source_mask,
                                      dropout_rate=dropout_rate,
                                      is_training=is_training,
                                      scope="enc_dec_attention")
                # Feed Forward
                dec_input_embed = FFN(dec_input_embed, n_units=[hyperparameters.input_embed_size*4, hyperparameters.input_embed_size], dropout_rate=dropout_rate, is_training=is_training)
                # Weighted sum of hidden states
                dec_weighted_hidden += (dec_weights[i] * dec_input_embed )
        dec_input_embed = dec_weighted_hidden
    return chord_change_logits, dec_input_embed, enc_weights, dec_weights

def HTv2(x, source_mask, target_mask, slope, dropout_rate, is_training, hyperparameters):
    input = x
    with tf.variable_scope("encoder_input_embedding"):
        enc_input_embed = tf.layers.dense(input, hyperparameters.input_embed_size) # [n_batches, n_steps, input_embedding_size]
        enc_input_embed = tf.layers.dropout(enc_input_embed, rate=dropout_rate, training=is_training)
        enc_input_embed = intra_block_MHA(inputs=enc_input_embed,
                                          n_blocks=hyperparameters.n_steps//4,
                                          n_heads=hyperparameters.n_heads,
                                          n_units=hyperparameters.input_embed_size,
                                          mask=source_mask,
                                          dropout_rate=dropout_rate,
                                          is_training=is_training)

    with tf.name_scope("encoder_positional_encoding"):
        enc_input_embed += get_absolute_position_encoding(tf.shape(enc_input_embed)[1], tf.shape(enc_input_embed)[2])
        enc_input_embed = tf.layers.dropout(enc_input_embed, rate=dropout_rate, training=is_training)

    with tf.variable_scope("encoder"):
        enc_weights = tf.nn.softmax(tf.get_variable('enc_weights_of_layers', dtype=tf.float32, shape=[hyperparameters.n_layers+1], initializer=tf.initializers.zeros())) # [n_layers]
        enc_weighted_hidden = enc_weights[0] * enc_input_embed
        for i in range(1, hyperparameters.n_layers+1):
            with tf.variable_scope("layer_{}".format(i)):
                # Multihead Attention (self-attention)
                enc_input_embed = MHA(queries=enc_input_embed,
                                      keys=enc_input_embed,
                                      n_units=hyperparameters.input_embed_size,
                                      n_heads=hyperparameters.n_heads,
                                      relative_position=True,
                                      max_dist=hyperparameters.n_steps-1,
                                      key_mask=source_mask,
                                      dropout_rate=dropout_rate,
                                      is_training=is_training,
                                      scope="enc_self_attention")
                # Feed Forward
                enc_input_embed = convFFN(enc_input_embed, n_units=[hyperparameters.input_embed_size, hyperparameters.input_embed_size],
                                          dropout_rate=dropout_rate, is_training=is_training)
                # Weighted sum of hidden states
                enc_weighted_hidden += (enc_weights[i] * enc_input_embed)
        enc_input_embed = enc_weighted_hidden # [n_batches, n_steps, input_embedding_size]

    with tf.name_scope("chord_change_prediction"):
        chord_change_logits = tf.squeeze(tf.layers.dense(enc_input_embed, 1), axis=-1) # shape = [n_batches, n_steps]
        chord_change_prob = tf.sigmoid(slope*chord_change_logits) # shape = [n_batches, n_steps]
        chord_change_prediction = binaryRound(chord_change_prob, cast_to_int=True) # Binarization of chord change prediction, [n_batches, n_steps]

    with tf.variable_scope("decoder_input_embedding"):
        dec_input_embed = tf.layers.dense(input, hyperparameters.input_embed_size) # [n_batches, n_steps, input_embed_size]
        dec_input_embed = tf.layers.dropout(dec_input_embed, rate=dropout_rate, training=is_training)
        dec_input_embed = intra_block_MHA(inputs=dec_input_embed,
                                          n_blocks=hyperparameters.n_steps//4,
                                          n_heads=hyperparameters.n_heads,
                                          n_units=hyperparameters.input_embed_size,
                                          mask=target_mask,
                                          dropout_rate=dropout_rate,
                                          is_training=is_training)

    with tf.variable_scope("input_embedding_regionalization"):
        dec_input_embed_reg, block_ids, num_blocks = chord_block_compression(dec_input_embed, chord_change_prediction)
        dec_input_embed_reg = decode_compressed_sequences(dec_input_embed_reg, block_ids) # [n_batches, n_steps, input_embed_size]
        dec_input_embed_reg.set_shape([None, hyperparameters.n_steps, hyperparameters.input_embed_size])
        dec_input_embed = dec_input_embed + dec_input_embed_reg + enc_input_embed # [n_batches, n_steps, input_embed_size]

    with tf.name_scope("decoder_positional_encoding"):
        dec_pe = get_absolute_position_encoding(hyperparameters.n_steps, hyperparameters.input_embed_size)
        dec_pe_batch = tf.tile(dec_pe, [tf.shape(dec_input_embed)[0],1,1])
        dec_input_embed += dec_pe_batch
        dec_input_embed = tf.layers.dropout(dec_input_embed, rate=dropout_rate, training=is_training)

    with tf.variable_scope("decoder"):
        dec_weights = tf.nn.softmax(tf.get_variable('dec_weights_of_layers', dtype=tf.float32, shape=[hyperparameters.n_layers+1], initializer=tf.initializers.zeros())) # [n_layers]
        dec_weighted_hidden = dec_weights[0] * dec_input_embed
        self_attn_map_list = []
        attn_map_list = []
        for i in range(1, hyperparameters.n_layers+1):
            with tf.variable_scope("layer_{}".format(i)):
                # Multihead Attention (self-attention)
                dec_input_embed, self_attn_map = MHA(queries=dec_input_embed,
                                                     keys=dec_input_embed,
                                                     n_units=hyperparameters.input_embed_size,
                                                     n_heads=hyperparameters.n_heads,
                                                     key_mask=target_mask,
                                                     relative_position=True,
                                                     max_dist=hyperparameters.n_steps-1,
                                                     dropout_rate=dropout_rate,
                                                     is_training=is_training,
                                                     attention_map=True,
                                                     scope="dec_self_attention")
                # Positional Attention (self-attention)
                dec_input_embed = MHA(queries=dec_pe_batch,
                                      keys=dec_pe_batch,
                                      values=dec_input_embed,
                                      n_units=hyperparameters.input_embed_size,
                                      n_heads=hyperparameters.n_heads,
                                      key_mask=target_mask,
                                      relative_position=True,
                                      max_dist=hyperparameters.n_steps-1,
                                      positional_attention=True,
                                      dropout_rate=dropout_rate,
                                      is_training=is_training,
                                      scope="position_attention")
                # Multihead Attention (seq2seq attention)
                dec_input_embed, attn_map = MHA(queries=dec_input_embed,
                                                keys=enc_input_embed,
                                                n_units=hyperparameters.input_embed_size,
                                                n_heads=hyperparameters.n_heads,
                                                key_mask=source_mask,
                                                relative_position=True,
                                                max_dist=hyperparameters.n_steps-1,
                                                dropout_rate=dropout_rate,
                                                is_training=is_training,
                                                attention_map=True,
                                                scope="enc_dec_attention")
                self_attn_map_list.append(self_attn_map)
                attn_map_list.append(attn_map)
                # Feed Forward
                dec_input_embed = convFFN(dec_input_embed, n_units=[hyperparameters.input_embed_size, hyperparameters.input_embed_size],
                                          dropout_rate=dropout_rate, is_training=is_training)
                # Weighted sum of all layers
                dec_weighted_hidden += (dec_weights[i] * dec_input_embed )
        dec_input_embed = dec_weighted_hidden
    return chord_change_logits, dec_input_embed, enc_weights, dec_weights, self_attn_map_list, attn_map_list

def intra_block_MHA(inputs, n_blocks, n_heads, n_units, mask, dropout_rate, is_training, scope='intra_block_MHA'):
    '''input shape = [N, T, C], mask shape = [N, T], where N = batch size, T = time steps, C = feature size'''
    # Split
    blocks_embed = tf.concat(tf.split(inputs, n_blocks, axis=1), axis=0) # [b*N, T/b, C]
    mask_reshape = tf.concat(tf.split(mask, n_blocks, axis=1), axis=0) # [b*N, T/b]
    # blocks_embed += get_absolute_position_encoding(tf.shape(blocks_embed)[1], tf.shape(blocks_embed)[2])
    with tf.variable_scope(scope):
        # Multihead attention
        blocks_embed = MHA(queries=blocks_embed,
                           keys=blocks_embed,
                           n_units=n_units,
                           n_heads=n_heads,
                           relative_position=True,
                           max_dist=3,
                           key_mask=mask_reshape,
                           dropout_rate=dropout_rate,
                           is_training=is_training,
                           scope="intra_block_MHA")
        # Feed Forward
        blocks_embed = convFFN(blocks_embed, n_units=[n_units, n_units], dropout_rate=dropout_rate, is_training=is_training)
    # Restore shape
    blocks_embed = tf.concat(tf.split(blocks_embed, n_blocks, axis=0), axis=1) # [N, T, C]
    return blocks_embed

def CRNN(x, x_len, dropout_rate, is_training, hyperparameters):
    '''https://github.com/Belval/CRNN/blob/master/CRNN/crnn.py'''
    with tf.variable_scope("encoder_input_embedding"):
        input_embed = tf.layers.dense(x, hyperparameters.input_embed_size) # [n_batches, n_steps, input_embedding_size]
        input_embed = tf.layers.dropout(input_embed, rate=dropout_rate, training=is_training)

    with tf.variable_scope("CNN"):
        for i in range(5):
            with tf.variable_scope("layer_{}".format(i)):
                input_embed = tf.layers.conv1d(inputs=input_embed, filters=hyperparameters.input_embed_size, kernel_size=9, activation=tf.nn.relu, use_bias=True, padding='same')
                input_embed = tf.layers.conv1d(inputs=input_embed, filters=hyperparameters.input_embed_size, kernel_size=9, activation=tf.nn.relu, use_bias=True, padding='same')
                input_embed = tf.layers.batch_normalization(inputs=input_embed, training=is_training)

    with tf.variable_scope("BLSTM_RNN"):
        with tf.name_scope('LSTM_cells'):
            cell_fw = LSTMCell(num_units=hyperparameters.input_embed_size, name='cell_fw')
            cell_bw = LSTMCell(num_units=hyperparameters.input_embed_size, name='cell_bw')
            cell_fw = DropoutWrapper(cell_fw, input_keep_prob=1 - dropout_rate)
            cell_bw = DropoutWrapper(cell_bw, input_keep_prob=1 - dropout_rate)

        with tf.name_scope('RNN'):
            (output_fw, output_bw), (_, _) = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                                             cell_bw=cell_bw,
                                                                             inputs=input_embed,
                                                                             sequence_length=x_len,
                                                                             dtype=tf.float32,
                                                                             time_major=False)
            input_embed = tf.concat((output_fw, output_bw), axis=-1)
    return input_embed

def segmentation_quality(reference, estimated, x_len):
    def directional_hamming_distance(ref_seq, est_seq):
        ref_seg_idx = np.concatenate([[0], np.reshape(np.where(np.not_equal(ref_seq[1:], ref_seq[:-1])), [-1]) + 1, [np.size(ref_seq)]], axis=0)
        est_seg_idx = np.concatenate([[0], np.reshape(np.where(np.not_equal(est_seq[1:], est_seq[:-1])), [-1]) + 1, [np.size(estimated)]], axis=0)
        seg = 0
        for start, end in zip(ref_seg_idx[:-1], ref_seg_idx[1:]):
            dur = end - start
            between_start_end = est_seg_idx[(est_seg_idx >= start) & (est_seg_idx < end)]
            seg_ts = np.hstack([start, between_start_end, end])
            seg += dur - np.diff(seg_ts).max()
        return seg / (ref_seg_idx[-1] - ref_seg_idx[0])

    sq = []
    for ref_seq, est_seq, l in zip(reference, estimated, x_len):
        ref2est_dist = directional_hamming_distance(ref_seq[:l], est_seq[:l])
        est2ref_dist = directional_hamming_distance(est_seq[:l], ref_seq[:l])
        score = 1 - max(ref2est_dist, est2ref_dist)
        sq.append(score)
    return np.mean(sq)

