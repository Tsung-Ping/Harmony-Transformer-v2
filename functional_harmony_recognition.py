import numpy as np
import tensorflow as tf # version = 1.8.0
import time
import random
import math
import pickle
from collections import Counter, namedtuple
import chord_recognition_models as crm

# Disables AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Mappings of functional harmony
'''key: 7 degrees * 3 accidentals * 2 modes + 1 padding= 43'''
key_dict = {}
for i_a, accidental in enumerate(['', '#', 'b']):
    for i_t, tonic in enumerate(['C', 'D', 'E', 'F', 'G', 'A', 'B', 'c', 'd', 'e', 'f', 'g', 'a', 'b']):
        key_dict[tonic + accidental] = i_a * 14 + i_t
        if accidental == '#':
            key_dict[tonic + '+'] = i_a * 14 + i_t
        elif accidental == 'b':
            key_dict[tonic + '-'] = i_a * 14 + i_t
key_dict['pad'] = 42

'''degree1: 10 (['1', '2', '3', '4', '5', '6', '7', '-2', '-7', 'pad'])'''
degree1_dict = {d1: i for i, d1 in enumerate(['1', '2', '3', '4', '5', '6', '7', '-2', '-7', 'pad'])}

'''degree2: 15 ['1', '2', '3', '4', '5', '6', '7', '+1', '+3', '+4', '-2', '-3', '-6', '-7', 'pad'])'''
degree2_dict = {d2: i for i, d2 in enumerate(['1', '2', '3', '4', '5', '6', '7', '+1', '+3', '+4', '-2', '-3', '-6', '-7', 'pad'])}

'''quality: 11 (['M', 'm', 'a', 'd', 'M7', 'm7', 'D7', 'd7', 'h7', 'a6', 'pad'])'''
quality_dict = {q: i for i, q in enumerate(['M', 'm', 'a', 'd', 'M7', 'm7', 'D7', 'd7', 'h7', 'a6', 'pad'])}
quality_dict['a7'] = [v for k, v in quality_dict.items() if k == 'a'][0]

def load_data_functional(dir, test_set_id=1, sequence_with_overlap=True):
    if test_set_id not in [1, 2, 3, 4]:
        print('Invalid testing_set_id.')
        exit(1)

    print("Load functional harmony data ...")
    print('test_set_id =', test_set_id)
    with open(dir, 'rb') as file:
        corpus_aug_reshape = pickle.load(file)
    print('keys in corpus_aug_reshape[\'shift_id\'][\'op\'] =', corpus_aug_reshape['shift_0']['1'].keys())

    shift_list = sorted(corpus_aug_reshape.keys())
    number_of_pieces = len(corpus_aug_reshape['shift_0'].keys())
    train_op_list = [str(i + 1) for i in range(number_of_pieces) if i % 4 + 1 != test_set_id]
    test_op_list = [str(i + 1) for i in range(number_of_pieces) if i % 4 + 1 == test_set_id]
    print('shift_list =', shift_list)
    print('train_op_list =', train_op_list)
    print('test_op_list =', test_op_list)

    overlap = int(sequence_with_overlap)

    # Training set
    train_data = {'pianoroll': np.concatenate([corpus_aug_reshape[shift_id][op]['pianoroll'][overlap] for shift_id in shift_list for op in train_op_list], axis=0),
                  'tonal_centroid': np.concatenate([corpus_aug_reshape[shift_id][op]['tonal_centroid'][overlap] for shift_id in shift_list for op in train_op_list], axis=0),
                  'len': np.concatenate([corpus_aug_reshape[shift_id][op]['len'][overlap] for shift_id in shift_list for op in train_op_list], axis=0),
                  'label': np.concatenate([corpus_aug_reshape[shift_id][op]['label'][overlap] for shift_id in shift_list for op in train_op_list], axis=0)}

    train_data_label_key = np.zeros_like(train_data['label'], dtype=np.int32)
    train_data_label_degree1 = np.zeros_like(train_data['label'], dtype=np.int32)
    train_data_label_degree2 = np.zeros_like(train_data['label'], dtype=np.int32)
    train_data_label_quality = np.zeros_like(train_data['label'], dtype=np.int32)
    train_data_label_inversion = train_data['label']['inversion']

    # Functional harmony labels
    '''key: 42'''
    for k, v in key_dict.items():
        train_data_label_key[train_data['label']['key'] == k] = v
    '''degree1: 9'''
    for k, v in degree1_dict.items():
        train_data_label_degree1[train_data['label']['degree1'] == k] = v
    '''degree2: 14'''
    for k, v in degree2_dict.items():
        train_data_label_degree2[train_data['label']['degree2'] == k] = v
    '''quality: 10'''
    for k, v in quality_dict.items():
        train_data_label_quality[train_data['label']['quality'] == k] = v
    '''inversion: 4'''
    train_data_label_inversion[train_data_label_inversion == -1] = 4
    '''roman numeral: (degree1, degree2, quality, inversion)'''
    train_data_label_roman = train_data_label_degree1 * 14 * 10 * 4 + train_data_label_degree2 * 10 * 4 + train_data_label_quality * 4 + train_data_label_inversion
    train_data_label_roman[train_data['label']['key'] == 'pad'] = 9 * 14 * 10 * 4

    train_data['key'] = train_data_label_key
    train_data['degree1'] = train_data_label_degree1
    train_data['degree2'] = train_data_label_degree2
    train_data['quality'] = train_data_label_quality
    train_data['inversion'] = train_data_label_inversion
    train_data['roman'] = train_data_label_roman

    # Test set
    test_data = {'pianoroll': np.concatenate([corpus_aug_reshape['shift_0'][op]['pianoroll'][0] for op in test_op_list], axis=0),
                 'tonal_centroid': np.concatenate([corpus_aug_reshape['shift_0'][op]['tonal_centroid'][0] for op in test_op_list], axis=0),
                 'len': np.concatenate([corpus_aug_reshape['shift_0'][op]['len'][0] for op in test_op_list], axis=0),
                 'label': np.concatenate([corpus_aug_reshape['shift_0'][op]['label'][0] for op in test_op_list], axis=0)}

    test_data_label_key = np.zeros_like(test_data['label'], dtype=np.int32)
    test_data_label_degree1 = np.zeros_like(test_data['label'], dtype=np.int32)
    test_data_label_degree2 = np.zeros_like(test_data['label'], dtype=np.int32)
    test_data_label_quality = np.zeros_like(test_data['label'], dtype=np.int32)
    test_data_label_inversion = test_data['label']['inversion']

    # Functional harmony labels
    '''key: 42'''
    for k, v in key_dict.items():
        test_data_label_key[test_data['label']['key'] == k] = v
    '''degree1: 9'''
    for k, v in degree1_dict.items():
        test_data_label_degree1[test_data['label']['degree1'] == k] = v
    '''degree2: 14'''
    for k, v in degree2_dict.items():
        test_data_label_degree2[test_data['label']['degree2'] == k] = v
    '''quality: 10'''
    for k, v in quality_dict.items():
        test_data_label_quality[test_data['label']['quality'] == k] = v
    '''inversion: 4'''
    test_data_label_inversion[test_data_label_inversion == -1] = 4
    '''roman numeral'''
    test_data_label_roman = test_data_label_degree1 * 14 * 10 * 4 + test_data_label_degree2 * 10 * 4 + test_data_label_quality * 4 + test_data_label_inversion
    test_data_label_roman[test_data['label']['key'] == 'pad'] = 9 * 14 * 10 * 4

    test_data['key'] = test_data_label_key
    test_data['degree1'] = test_data_label_degree1
    test_data['degree2'] = test_data_label_degree2
    test_data['quality'] = test_data_label_quality
    test_data['inversion'] = test_data_label_inversion
    test_data['roman'] = test_data_label_roman

    print('keys in train/test_data =', train_data.keys())
    return train_data, test_data

def compute_pre_PRF(predicted, actual):
    predicted = tf.cast(predicted, tf.float32)
    actual = tf.cast(actual, tf.float32)
    TP = tf.count_nonzero(predicted * actual, dtype=tf.float32)
    # TN = tf.count_nonzero((predicted - 1) * (actual - 1), dtype=tf.float32)
    FP = tf.count_nonzero(predicted * (actual - 1), dtype=tf.float32)
    FN = tf.count_nonzero((predicted - 1) * actual, dtype=tf.float32)
    return TP, FP, FN

def comput_PRF_with_pre(TP, FP, FN):
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * precision * recall / (precision + recall)
    precision = tf.cond(tf.is_nan(precision), lambda: tf.constant(0.0), lambda: precision)
    recall = tf.cond(tf.is_nan(recall), lambda: tf.constant(0.0), lambda: recall)
    F1 = tf.cond(tf.is_nan(F1), lambda: tf.constant(0.0), lambda: F1)
    return precision, recall, F1

def train_HT():
    print('Run HT functional harmony recognition on %s-%d...' % (hp.dataset, hp.test_set_id))

    # Load training and testing data
    train_data, test_data = load_data_functional(dir=hp.dataset + '_preprocessed_data_MIREX_Mm.pickle', test_set_id=hp.test_set_id, sequence_with_overlap=hp.train_sequence_with_overlap)
    n_train_sequences = train_data['pianoroll'].shape[0]
    n_test_sequences = test_data['pianoroll'].shape[0]
    n_iterations_per_epoch = int(math.ceil(n_train_sequences/hp.n_batches))
    print('n_train_sequences =', n_train_sequences)
    print('n_test_sequences =', n_test_sequences)
    print('n_iterations_per_epoch =', n_iterations_per_epoch)
    print(hp)

    with tf.name_scope('placeholder'):
        x_p = tf.placeholder(tf.int32, [None, hp.n_steps, 88], name="pianoroll")
        x_len = tf.placeholder(tf.int32, [None], name="seq_lens")
        y_k = tf.placeholder(tf.int32, [None, hp.n_steps], name="key") # 7 degrees * 3 accidentals * 2 modes = 42
        y_r = tf.placeholder(tf.int32, [None, hp.n_steps], name="roman_numeral")
        y_cc = tf.placeholder(tf.int32, [None, hp.n_steps], name="chord_change")
        dropout = tf.placeholder(dtype=tf.float32, name="dropout_rate")
        is_training = tf.placeholder(dtype=tf.bool, name="is_training")
        global_step = tf.placeholder(dtype=tf.int32, name='global_step')
        slope = tf.placeholder(dtype=tf.float32, name='annealing_slope')

    with tf.name_scope('model'):
        x_in = tf.cast(x_p, tf.float32)
        source_mask = tf.sequence_mask(lengths=x_len, maxlen=hp.n_steps, dtype=tf.float32) # [n_batches, n_steps]
        target_mask = source_mask
        # chord_change_logits, dec_input_embed, enc_weights, dec_weights = crm.HT(x_in, source_mask, target_mask, slope, dropout, is_training, hp)
        chord_change_logits, dec_input_embed, enc_weights, dec_weights, _, _ = crm.HTv2(x_in, source_mask, target_mask, slope, dropout, is_training, hp)

    with tf.variable_scope("output_projection"):
        n_key_classes = 42 + 1
        n_roman_classes = 9 * 14 * 10 * 4 + 1
        dec_input_embed = tf.layers.dropout(dec_input_embed, rate=dropout, training=is_training)
        key_logits = tf.layers.dense(dec_input_embed, n_key_classes)
        roman_logits = tf.layers.dense(dec_input_embed, n_roman_classes)

    with tf.name_scope('loss'):
        # Chord change
        loss_cc = 4 * tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.cast(y_cc, tf.float32), logits=slope*chord_change_logits, weights=source_mask)
        # Key
        loss_k = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(y_k, n_key_classes), logits=key_logits, weights=target_mask, label_smoothing=0.01)
        # Roman numeral
        loss_r = 0.5 * tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(y_r, n_roman_classes), logits=roman_logits, weights=target_mask, label_smoothing=0.0)
        # Total loss
        loss = loss_cc + loss_k + loss_r
    valid = tf.reduce_sum(target_mask)
    summary_loss = tf.Variable([0.0 for _ in range(4)], trainable=False, dtype=tf.float32)
    summary_valid = tf.Variable(0.0, trainable=False, dtype=tf.float32)
    update_loss = tf.assign(summary_loss, summary_loss + valid * [loss, loss_cc, loss_k, loss_r])
    update_valid = tf.assign(summary_valid, summary_valid + valid)
    mean_loss = tf.assign(summary_loss, summary_loss / summary_valid)
    clr_summary_loss = summary_loss.initializer
    clr_summary_valid = summary_valid.initializer
    tf.summary.scalar('Loss_total', summary_loss[0])
    tf.summary.scalar('Loss_chord_change', summary_loss[1])
    tf.summary.scalar('Loss_key', summary_loss[2])
    tf.summary.scalar('Loss_roman', summary_loss[3])

    with tf.name_scope('evaluation'):
        eval_mask = tf.cast(target_mask, tf.bool)
        # Chord change
        pred_cc = tf.cast(tf.round(tf.sigmoid(slope*chord_change_logits)), tf.int32)
        pred_cc_mask = tf.boolean_mask(pred_cc, tf.cast(source_mask, tf.bool))
        y_cc_mask = tf.boolean_mask(y_cc, tf.cast(source_mask, tf.bool))
        TP_cc, FP_cc, FN_cc = compute_pre_PRF(pred_cc_mask, y_cc_mask)
        # Key
        pred_k = tf.argmax(key_logits, axis=2, output_type=tf.int32)
        pred_k_correct = tf.equal(pred_k, y_k)
        pred_k_correct_mask = tf.boolean_mask(tensor=pred_k_correct, mask=eval_mask)
        n_correct_k = tf.reduce_sum(tf.cast(pred_k_correct_mask, tf.float32))
        # Roman numeral
        pred_r = tf.argmax(roman_logits, axis=2, output_type=tf.int32)
        pred_r_correct = tf.equal(pred_r, y_r)
        pred_r_correct_mask = tf.boolean_mask(tensor=pred_r_correct, mask=eval_mask)
        n_correct_r = tf.reduce_sum(tf.cast(pred_r_correct_mask, tf.float32))
        n_total = tf.cast(tf.size(pred_r_correct_mask), tf.float32)
    summary_count = tf.Variable([0.0 for _ in range(6)], trainable=False, dtype=tf.float32)
    summary_score = tf.Variable([0.0 for _ in range(5)], trainable=False, dtype=tf.float32)
    update_count = tf.assign(summary_count, summary_count + [n_correct_k, n_correct_r, n_total, TP_cc, FP_cc, FN_cc])
    acc_k = summary_count[0] / summary_count[2]
    acc_r = summary_count[1] / summary_count[2]
    P_cc, R_cc, F1_cc = comput_PRF_with_pre(summary_count[3], summary_count[4], summary_count[5])
    update_score = tf.assign(summary_score, summary_score + [acc_k, acc_r, P_cc, R_cc, F1_cc])
    clr_summary_count = summary_count.initializer
    clr_summary_score = summary_score.initializer
    tf.summary.scalar('Accuracy_key', summary_score[0])
    tf.summary.scalar('Accuracy_roman', summary_score[1])
    tf.summary.scalar('Precision_cc', summary_score[2])
    tf.summary.scalar('Recall_cc', summary_score[3])
    tf.summary.scalar('F1_cc', summary_score[4])

    with tf.name_scope('optimization'):
        # Apply warn-up learning rate
        warm_up_steps = tf.constant(4000, dtype=tf.float32)
        gstep = tf.cast(global_step, dtype=tf.float32)
        learning_rate = pow(hp.input_embed_size, -0.5) * tf.minimum(tf.pow(gstep, -0.5), gstep * tf.pow(warm_up_steps, -1.5))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           beta1=0.9,
                                           beta2=0.98,
                                           epsilon=1e-9)
        train_op = optimizer.minimize(loss)
    # Graph location and summary writers
    print('Saving graph to: %s' % hp.graph_location)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(hp.graph_location + '\\train')
    test_writer = tf.summary.FileWriter(hp.graph_location + '\\test')
    train_writer.add_graph(tf.get_default_graph())
    test_writer.add_graph(tf.get_default_graph())
    saver = tf.train.Saver(max_to_keep=1)

    # Training
    print('Train the model...')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        startTime = time.time() # start time of training
        best_score = [0.0 for _ in range(6)]
        in_succession = 0
        best_epoch = 0
        annealing_slope = 1.0
        best_slope = 0.0
        for step in range(hp.n_training_steps):
            # Training
            if step == 0:
                indices = range(n_train_sequences)
                batch_indices = [indices[x:x + hp.n_batches] for x in range(0, len(indices), hp.n_batches)]

            if step > 0 and step % n_iterations_per_epoch == 0:
                annealing_slope *= hp.annealing_rate

            if step >= 2*n_iterations_per_epoch and step % n_iterations_per_epoch == 0:
                # Shuffle training data
                indices = random.sample(range(n_train_sequences), n_train_sequences)
                batch_indices = [indices[x:x + hp.n_batches] for x in range(0, len(indices), hp.n_batches)]

            batch = (train_data['pianoroll'][batch_indices[step % len(batch_indices)]],
                     train_data['len'][batch_indices[step % len(batch_indices)]],
                     train_data['label']['chord_change'][batch_indices[step % len(batch_indices)]],
                     train_data['key'][batch_indices[step % len(batch_indices)]],
                     train_data['roman'][batch_indices[step % len(batch_indices)]],
                     train_data['degree1'][batch_indices[step % len(batch_indices)]],
                     train_data['degree2'][batch_indices[step % len(batch_indices)]],
                     train_data['quality'][batch_indices[step % len(batch_indices)]],
                     train_data['inversion'][batch_indices[step % len(batch_indices)]],
                     train_data['label']['key'][batch_indices[step % len(batch_indices)]])

            train_run_list = [train_op, update_valid, update_loss, update_count, loss, loss_cc, loss_k, loss_r, pred_cc, pred_k, pred_r, eval_mask, enc_weights, dec_weights]
            train_feed_fict = {x_p: batch[0],
                               x_len: batch[1],
                               y_cc: batch[2],
                               y_k: batch[3],
                               y_r: batch[4],
                               dropout: hp.drop,
                               is_training: True,
                               global_step: step + 1,
                               slope: annealing_slope}
            _, _, _, _, train_loss, train_loss_cc, train_loss_k, train_loss_r, \
            train_pred_cc, train_pred_k, train_pred_r, train_eval_mask, enc_w, dec_w = sess.run(train_run_list, feed_dict=train_feed_fict)
            if step == 0:
                print('*~ loss_cc %.4f, loss_k %.4f, loss_r %.4f ~*' % (train_loss_cc, train_loss_k, train_loss_r))

            # Display training log & Testing
            if step > 0 and step % n_iterations_per_epoch == 0:
                sess.run([mean_loss, update_score])
                train_summary, train_loss, train_score = sess.run([merged, summary_loss, summary_score])
                sess.run([clr_summary_valid, clr_summary_loss, clr_summary_count, clr_summary_score])
                train_writer.add_summary(train_summary, step)
                print("---- step %d, epoch %d: train_loss: total %.4f (cc %.4f, k %.4f, r %.4f), evaluation: k %.4f, r %.4f, cc (P %.4f, R %.4f, F1 %.4f) ----"
                    % (step, step // n_iterations_per_epoch, train_loss[0], train_loss[1], train_loss[2], train_loss[3],
                       train_score[0], train_score[1], train_score[2], train_score[3], train_score[4]))
                print('enc_w =', enc_w, 'dec_w =', dec_w)
                display_len = 32
                n_just = 5
                print('len =', batch[1][0])
                print('y_k'.ljust(7, ' '), ''.join([b.rjust(n_just, ' ') for b in batch[9][0, :display_len]]))
                print('y_d1'.ljust(7, ' '), ''.join([[k for k, v in degree1_dict.items() if v == b][0].rjust(n_just, ' ') for b in batch[5][0, :display_len]]))
                print('y_d2'.ljust(7, ' '), ''.join([[k for k, v in degree2_dict.items() if v == b][0].rjust(n_just, ' ') for b in batch[6][0, :display_len]]))
                print('y_q'.ljust(7, ' '), ''.join([[k for k, v in quality_dict.items() if v == b][0].rjust(n_just, ' ') for b in batch[7][0, :display_len]]))
                print('y_inv'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in batch[8][0, :display_len]]))
                print('valid'.ljust(7, ' '), ''.join(['y'.rjust(n_just, ' ') if b else 'n'.rjust(n_just, ' ') for b in train_eval_mask[0, :display_len]]))
                print('y_cc'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in batch[2][0, :display_len]]))
                print('pred_cc'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in train_pred_cc[0, :display_len]]))
                print('y_k'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in batch[3][0, :display_len]]))
                print('pred_k'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in train_pred_k[0, :display_len]]))
                print('y_r'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in batch[4][0, :display_len]]))
                print('pred_r'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in train_pred_r[0, :display_len]]))

                # Testing
                test_run_list = [update_valid, update_loss, update_count, pred_cc, pred_k, pred_r, eval_mask]
                test_feed_fict = {x_p: test_data['pianoroll'],
                                  x_len: test_data['len'],
                                  y_cc: test_data['label']['chord_change'],
                                  y_k: test_data['key'],
                                  y_r: test_data['roman'],
                                  dropout: 0.0,
                                  is_training: False,
                                  slope: annealing_slope}
                _, _, _, test_pred_cc, test_pred_k, test_pred_r, test_eval_mask = sess.run(test_run_list, feed_dict=test_feed_fict)
                sess.run([mean_loss, update_score])
                test_summary, test_loss, test_score = sess.run([merged, summary_loss, summary_score])
                sess.run([clr_summary_valid, clr_summary_loss, clr_summary_count, clr_summary_score])
                test_writer.add_summary(test_summary, step)

                sq = crm.segmentation_quality(test_data['roman'], test_pred_r, test_data['len'])
                print("==== step %d, epoch %d: test_loss: total %.4f (cc %.4f, k %.4f, r %.4f), evaluation: k %.4f, r %.4f, cc (P %.4f, R %.4f, F1 %.4f), sq %.4f ===="
                      % (step, step // n_iterations_per_epoch, test_loss[0], test_loss[1], test_loss[2], test_loss[3],
                         test_score[0], test_score[1], test_score[2], test_score[3], test_score[4], sq))
                sample_id = random.randint(0, n_test_sequences - 1)
                print('len =', test_data['len'][sample_id])
                print('y_k'.ljust(7, ' '), ''.join([b.rjust(n_just, ' ') for b in test_data['label']['key'][sample_id, :display_len]]))
                print('y_d1'.ljust(7, ' '), ''.join([[k for k, v in degree1_dict.items() if v == b][0].rjust(n_just, ' ') for b in test_data['degree1'][sample_id, :display_len]]))
                print('y_d2'.ljust(7, ' '), ''.join([[k for k, v in degree2_dict.items() if v == b][0].rjust(n_just, ' ') for b in test_data['degree2'][sample_id, :display_len]]))
                print('y_q'.ljust(7, ' '), ''.join([[k for k, v in quality_dict.items() if v == b][0].rjust(n_just, ' ') for b in test_data['quality'][sample_id, :display_len]]))
                print('y_inv'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in test_data['inversion'][sample_id, :display_len]]))
                print('valid'.ljust(7, ' '), ''.join(['y'.rjust(n_just, ' ') if b else 'n'.rjust(n_just, ' ') for b in test_eval_mask[sample_id, :display_len]]))
                print('y_cc'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in test_data['label']['chord_change'][sample_id, :display_len]]))
                print('pred_cc'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in test_pred_cc[sample_id, :display_len]]))
                print('y_k'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in test_data['key'][sample_id, :display_len]]))
                print('pred_k'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in test_pred_k[sample_id, :display_len]]))
                print('y_r'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in test_data['roman'][sample_id, :display_len]]))
                print('pred_r'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in test_pred_r[sample_id, :display_len]]))

                if step > 0 and sum(test_score[:2]) > sum(best_score[:2]):
                    best_score = np.concatenate([test_score, [sq]], axis=0)
                    best_epoch = step // n_iterations_per_epoch
                    best_slope = annealing_slope
                    in_succession = 0
                    # Save variables of the model
                    print('*saving variables...\n')
                    saver.save(sess, hp.graph_location + '\\HT_functional_harmony_recognition_' + hp.dataset + '_' + str(hp.test_set_id) + '.ckpt')
                else:
                    in_succession += 1
                    if in_succession > hp.n_in_succession:
                        print('Early stopping.')
                        break

        elapsed_time = time.time() - startTime
        print('\nHT functional harmony recognition on %s-%d:' % (hp.dataset, hp.test_set_id))
        print('training time = %.2f hr' % (elapsed_time / 3600))
        print('best epoch = ', best_epoch)
        print('best score =', np.round(best_score, 4))
        print('best slope =', best_slope)

def train_BTC():
    print('Run BTC functional harmony recognition on %s-%d...' % (hp.dataset, hp.test_set_id))

    # Load training and testing data
    train_data, test_data = load_data_functional(dir=hp.dataset + '_preprocessed_data_MIREX_Mm.pickle', test_set_id=hp.test_set_id, sequence_with_overlap=hp.train_sequence_with_overlap)
    n_train_sequences = train_data['pianoroll'].shape[0]
    n_test_sequences = test_data['pianoroll'].shape[0]
    n_iterations_per_epoch = int(math.ceil(n_train_sequences/hp.n_batches))
    print('n_train_sequences =', n_train_sequences)
    print('n_test_sequences =', n_test_sequences)
    print('n_iterations_per_epoch =', n_iterations_per_epoch)
    print(hp)

    with tf.name_scope('placeholder'):
        x_p = tf.placeholder(tf.int32, [None, hp.n_steps, 88], name="pianoroll")
        x_len = tf.placeholder(tf.int32, [None], name="seq_lens")
        y_k = tf.placeholder(tf.int32, [None, hp.n_steps], name="key") # 7 degrees * 3 accidentals * 2 modes = 42
        y_r = tf.placeholder(tf.int32, [None, hp.n_steps], name="roman_numeral")
        dropout = tf.placeholder(dtype=tf.float32, name="dropout_rate")
        is_training = tf.placeholder(dtype=tf.bool, name="is_training")
        global_step = tf.placeholder(dtype=tf.int32, name='global_step')

    with tf.name_scope('model'):
        x_in = tf.cast(x_p, tf.float32)
        source_mask = tf.sequence_mask(lengths=x_len, maxlen=hp.n_steps, dtype=tf.float32) # [n_batches, n_steps]
        target_mask = source_mask
        input_embed = crm.BTC(x_in, source_mask, dropout, is_training, hp)

    with tf.variable_scope("output_projection"):
        n_key_classes = 42 + 1
        n_roman_classes = 9 * 14 * 10 * 4 + 1
        input_embed = tf.layers.dropout(input_embed, rate=dropout, training=is_training)
        key_logits = tf.layers.dense(input_embed, n_key_classes)
        roman_logits = tf.layers.dense(input_embed, n_roman_classes)

    with tf.name_scope('loss'):
        # Key
        loss_k = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(y_k, n_key_classes), logits=key_logits, weights=target_mask)
        # Roman numeral
        loss_r = 0.5 * tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(y_r, n_roman_classes), logits=roman_logits, weights=target_mask)
        # Total loss
        loss = loss_k + loss_r
    valid = tf.reduce_sum(target_mask)
    summary_loss = tf.Variable([0.0 for _ in range(3)], trainable=False, dtype=tf.float32)
    summary_valid = tf.Variable(0.0, trainable=False, dtype=tf.float32)
    update_loss = tf.assign(summary_loss, summary_loss + valid * [loss, loss_k, loss_r])
    update_valid = tf.assign(summary_valid, summary_valid + valid)
    mean_loss = tf.assign(summary_loss, summary_loss / summary_valid)
    clr_summary_loss = summary_loss.initializer
    clr_summary_valid = summary_valid.initializer
    tf.summary.scalar('Loss_total', summary_loss[0])
    tf.summary.scalar('Loss_key', summary_loss[1])
    tf.summary.scalar('Loss_roman', summary_loss[2])

    with tf.name_scope('evaluation'):
        eval_mask = tf.cast(target_mask, tf.bool)
        # Key
        pred_k = tf.argmax(key_logits, axis=2, output_type=tf.int32)
        pred_k_correct = tf.equal(pred_k, y_k)
        pred_k_correct_mask = tf.boolean_mask(tensor=pred_k_correct, mask=eval_mask)
        n_correct_k = tf.reduce_sum(tf.cast(pred_k_correct_mask, tf.float32))
        # Roman numeral
        pred_r = tf.argmax(roman_logits, axis=2, output_type=tf.int32)
        pred_r_correct = tf.equal(pred_r, y_r)
        pred_r_correct_mask = tf.boolean_mask(tensor=pred_r_correct, mask=eval_mask)
        n_correct_r = tf.reduce_sum(tf.cast(pred_r_correct_mask, tf.float32))
        n_total = tf.cast(tf.size(pred_r_correct_mask), tf.float32)
    summary_count = tf.Variable([0.0 for _ in range(3)], trainable=False, dtype=tf.float32)
    summary_score = tf.Variable([0.0 for _ in range(2)], trainable=False, dtype=tf.float32)
    update_count = tf.assign(summary_count, summary_count + [n_correct_k, n_correct_r, n_total])
    acc_k = summary_count[0] / summary_count[2]
    acc_r = summary_count[1] / summary_count[2]
    update_score = tf.assign(summary_score, summary_score + [acc_k, acc_r])
    clr_summary_count = summary_count.initializer
    clr_summary_score = summary_score.initializer
    tf.summary.scalar('Accuracy_key', summary_score[0])
    tf.summary.scalar('Accuracy_roman', summary_score[1])

    with tf.name_scope('optimization'):
        # Apply warn-up learning rate
        warm_up_steps = tf.constant(4000, dtype=tf.float32)
        gstep = tf.cast(global_step, dtype=tf.float32)
        learning_rate = pow(hp.input_embed_size, -0.5) * tf.minimum(tf.pow(gstep, -0.5), gstep * tf.pow(warm_up_steps, -1.5))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           beta1=0.9,
                                           beta2=0.98,
                                           epsilon=1e-9)
        train_op = optimizer.minimize(loss)
    # Graph location and summary writers
    print('Saving graph to: %s' % hp.graph_location)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(hp.graph_location + '\\train')
    test_writer = tf.summary.FileWriter(hp.graph_location + '\\test')
    train_writer.add_graph(tf.get_default_graph())
    test_writer.add_graph(tf.get_default_graph())
    saver = tf.train.Saver(max_to_keep=1)

    # Training
    print('Train the model...')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        startTime = time.time() # start time of training
        best_score = [0.0 for _ in range(3)]
        in_succession = 0
        best_epoch = 0
        for step in range(hp.n_training_steps):
            # Training
            if step == 0:
                indices = range(n_train_sequences)
                batch_indices = [indices[x:x + hp.n_batches] for x in range(0, len(indices), hp.n_batches)]

            if step >= 2*n_iterations_per_epoch and step % n_iterations_per_epoch == 0:
                # Shuffle training data
                indices = random.sample(range(n_train_sequences), n_train_sequences)
                batch_indices = [indices[x:x + hp.n_batches] for x in range(0, len(indices), hp.n_batches)]

            batch = (train_data['pianoroll'][batch_indices[step % len(batch_indices)]],
                     train_data['len'][batch_indices[step % len(batch_indices)]],
                     train_data['key'][batch_indices[step % len(batch_indices)]],
                     train_data['roman'][batch_indices[step % len(batch_indices)]],
                     train_data['degree1'][batch_indices[step % len(batch_indices)]],
                     train_data['degree2'][batch_indices[step % len(batch_indices)]],
                     train_data['quality'][batch_indices[step % len(batch_indices)]],
                     train_data['inversion'][batch_indices[step % len(batch_indices)]],
                     train_data['label']['key'][batch_indices[step % len(batch_indices)]])

            train_run_list = [train_op, update_valid, update_loss, update_count, loss, loss_k, loss_r, pred_k, pred_r, eval_mask]
            train_feed_fict = {x_p: batch[0],
                               x_len: batch[1],
                               y_k: batch[2],
                               y_r: batch[3],
                               dropout: hp.drop,
                               is_training: True,
                               global_step: step + 1}
            _, _, _, _, train_loss, train_loss_k, train_loss_r,\
            train_pred_k, train_pred_r, train_eval_mask = sess.run(train_run_list, feed_dict=train_feed_fict)
            if step == 0:
                print('*~ loss_k %.4f, loss_r %.4f ~*' % (train_loss_k, train_loss_r))

            # Display training log & Testing
            if step > 0 and step % n_iterations_per_epoch == 0:
                sess.run([mean_loss, update_score])
                train_summary, train_loss, train_score = sess.run([merged, summary_loss, summary_score])
                sess.run([clr_summary_valid, clr_summary_loss, clr_summary_count, clr_summary_score])
                train_writer.add_summary(train_summary, step)
                print("---- step %d, epoch %d: train_loss: total %.4f (k %.4f, r %.4f), evaluation: k %.4f, r %.4f ----"
                    % (step, step // n_iterations_per_epoch, train_loss[0], train_loss[1], train_loss[2], train_score[0], train_score[1]))
                display_len = 32
                n_just = 5
                print('len =', batch[1][0])
                print('y_k'.ljust(7, ' '), ''.join([b.rjust(n_just, ' ') for b in batch[8][0, :display_len]]))
                print('y_d1'.ljust(7, ' '), ''.join([[k for k, v in degree1_dict.items() if v == b][0].rjust(n_just, ' ') for b in batch[4][0, :display_len]]))
                print('y_d2'.ljust(7, ' '), ''.join([[k for k, v in degree2_dict.items() if v == b][0].rjust(n_just, ' ') for b in batch[5][0, :display_len]]))
                print('y_q'.ljust(7, ' '), ''.join([[k for k, v in quality_dict.items() if v == b][0].rjust(n_just, ' ') for b in batch[6][0, :display_len]]))
                print('y_inv'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in batch[7][0, :display_len]]))
                print('valid'.ljust(7, ' '), ''.join(['y'.rjust(n_just, ' ') if b else 'n'.rjust(n_just, ' ') for b in train_eval_mask[0, :display_len]]))
                print('y_k'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in batch[2][0, :display_len]]))
                print('pred_k'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in train_pred_k[0, :display_len]]))
                print('y_r'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in batch[3][0, :display_len]]))
                print('pred_r'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in train_pred_r[0, :display_len]]))

                # Testing
                test_run_list = [update_valid, update_loss, update_count, pred_k, pred_r, eval_mask]
                test_feed_fict = {x_p: test_data['pianoroll'],
                                  x_len: test_data['len'],
                                  y_k: test_data['key'],
                                  y_r: test_data['roman'],
                                  dropout: 0.0,
                                  is_training: False}
                _, _, _, test_pred_k, test_pred_r, test_eval_mask = sess.run(test_run_list, feed_dict=test_feed_fict)
                sess.run([mean_loss, update_score])
                test_summary, test_loss, test_score = sess.run([merged, summary_loss, summary_score])
                sess.run([clr_summary_valid, clr_summary_loss, clr_summary_count, clr_summary_score])
                test_writer.add_summary(test_summary, step)

                sq = crm.segmentation_quality(test_data['roman'], test_pred_r, test_data['len'])
                print("==== step %d, epoch %d: test_loss: total %.4f (k %.4f, r %.4f), evaluation: k %.4f, r %.4f, sq %.4f ===="
                      % (step, step // n_iterations_per_epoch, test_loss[0], test_loss[1], test_loss[2], test_score[0], test_score[1], sq))
                sample_id = random.randint(0, n_test_sequences - 1)
                print('len =', test_data['len'][sample_id])
                print('y_k'.ljust(7, ' '), ''.join([b.rjust(n_just, ' ') for b in test_data['label']['key'][sample_id, :display_len]]))
                print('y_d1'.ljust(7, ' '), ''.join([[k for k, v in degree1_dict.items() if v == b][0].rjust(n_just, ' ') for b in test_data['degree1'][sample_id, :display_len]]))
                print('y_d2'.ljust(7, ' '), ''.join([[k for k, v in degree2_dict.items() if v == b][0].rjust(n_just, ' ') for b in test_data['degree2'][sample_id, :display_len]]))
                print('y_q'.ljust(7, ' '), ''.join([[k for k, v in quality_dict.items() if v == b][0].rjust(n_just, ' ') for b in test_data['quality'][sample_id, :display_len]]))
                print('y_inv'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in test_data['inversion'][sample_id, :display_len]]))
                print('valid'.ljust(7, ' '), ''.join(['y'.rjust(n_just, ' ') if b else 'n'.rjust(n_just, ' ') for b in test_eval_mask[sample_id, :display_len]]))
                print('y_k'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in test_data['key'][sample_id, :display_len]]))
                print('pred_k'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in test_pred_k[sample_id, :display_len]]))
                print('y_r'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in test_data['roman'][sample_id, :display_len]]))
                print('pred_r'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in test_pred_r[sample_id, :display_len]]))

                if step > 0 and sum(test_score[:2]) > sum(best_score[:2]):
                    best_score = np.concatenate([test_score, [sq]], axis=0)
                    best_epoch = step // n_iterations_per_epoch
                    in_succession = 0
                    # Save variables of the model
                    print('*saving variables...\n')
                    saver.save(sess, hp.graph_location + '\\BTC_functional_harmony_recognition_' + hp.dataset + '_' + str(hp.test_set_id) + '.ckpt')
                else:
                    in_succession += 1
                    if in_succession > hp.n_in_succession:
                        print('Early stopping.')
                        break

        elapsed_time = time.time() - startTime
        print('\nBTC functional harmony recognition on %s-%d:' % (hp.dataset, hp.test_set_id))
        print('training time = %.2f hr' % (elapsed_time / 3600))
        print('best epoch = ', best_epoch)
        print('best score =', np.round(best_score, 4))

def train_CRNN():
    print('Run CRNN functional harmony recognition on %s-%d...' % (hp.dataset, hp.test_set_id))

    # Load training and testing data
    train_data, test_data = load_data_functional(dir=hp.dataset + '_preprocessed_data_MIREX_Mm.pickle', test_set_id=hp.test_set_id, sequence_with_overlap=hp.train_sequence_with_overlap)
    n_train_sequences = train_data['pianoroll'].shape[0]
    n_test_sequences = test_data['pianoroll'].shape[0]
    n_iterations_per_epoch = int(math.ceil(n_train_sequences/hp.n_batches))
    print('n_train_sequences =', n_train_sequences)
    print('n_test_sequences =', n_test_sequences)
    print('n_iterations_per_epoch =', n_iterations_per_epoch)
    print(hp)

    with tf.name_scope('placeholder'):
        x_p = tf.placeholder(tf.int32, [None, hp.n_steps, 88], name="pianoroll")
        x_len = tf.placeholder(tf.int32, [None], name="seq_lens")
        y_k = tf.placeholder(tf.int32, [None, hp.n_steps], name="key") # 7 degrees * 3 accidentals * 2 modes = 42
        y_r = tf.placeholder(tf.int32, [None, hp.n_steps], name="roman_numeral")
        dropout = tf.placeholder(dtype=tf.float32, name="dropout_rate")
        is_training = tf.placeholder(dtype=tf.bool, name="is_training")
        global_step = tf.placeholder(dtype=tf.int32, name='global_step')

    with tf.name_scope('model'):
        x_in = tf.cast(x_p, tf.float32)
        source_mask = tf.sequence_mask(lengths=x_len, maxlen=hp.n_steps, dtype=tf.float32) # [n_batches, n_steps]
        target_mask = source_mask
        input_embed = crm.CRNN(x_in, x_len, dropout, is_training, hp)

    with tf.variable_scope("output_projection"):
        n_key_classes = 42 + 1
        n_roman_classes = 9 * 14 * 10 * 4 + 1
        input_embed = tf.layers.dropout(input_embed, rate=dropout, training=is_training)
        key_logits = tf.layers.dense(input_embed, n_key_classes)
        roman_logits = tf.layers.dense(input_embed, n_roman_classes)

    with tf.name_scope('loss'):
        # Key
        loss_k = 0.8 * tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(y_k, n_key_classes), logits=key_logits, weights=target_mask)
        # Roman numeral
        loss_r = 0.5 * tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(y_r, n_roman_classes), logits=roman_logits, weights=target_mask)
        # Total loss
        loss = loss_k + loss_r
    valid = tf.reduce_sum(target_mask)
    summary_loss = tf.Variable([0.0 for _ in range(3)], trainable=False, dtype=tf.float32)
    summary_valid = tf.Variable(0.0, trainable=False, dtype=tf.float32)
    update_loss = tf.assign(summary_loss, summary_loss + valid * [loss, loss_k, loss_r])
    update_valid = tf.assign(summary_valid, summary_valid + valid)
    mean_loss = tf.assign(summary_loss, summary_loss / summary_valid)
    clr_summary_loss = summary_loss.initializer
    clr_summary_valid = summary_valid.initializer
    tf.summary.scalar('Loss_total', summary_loss[0])
    tf.summary.scalar('Loss_key', summary_loss[1])
    tf.summary.scalar('Loss_roman', summary_loss[2])

    with tf.name_scope('evaluation'):
        eval_mask = tf.cast(target_mask, tf.bool)
        # Key
        pred_k = tf.argmax(key_logits, axis=2, output_type=tf.int32)
        pred_k_correct = tf.equal(pred_k, y_k)
        pred_k_correct_mask = tf.boolean_mask(tensor=pred_k_correct, mask=eval_mask)
        n_correct_k = tf.reduce_sum(tf.cast(pred_k_correct_mask, tf.float32))
        # Roman numeral
        pred_r = tf.argmax(roman_logits, axis=2, output_type=tf.int32)
        pred_r_correct = tf.equal(pred_r, y_r)
        pred_r_correct_mask = tf.boolean_mask(tensor=pred_r_correct, mask=eval_mask)
        n_correct_r = tf.reduce_sum(tf.cast(pred_r_correct_mask, tf.float32))
        n_total = tf.cast(tf.size(pred_r_correct_mask), tf.float32)
    summary_count = tf.Variable([0.0 for _ in range(3)], trainable=False, dtype=tf.float32)
    summary_score = tf.Variable([0.0 for _ in range(2)], trainable=False, dtype=tf.float32)
    update_count = tf.assign(summary_count, summary_count + [n_correct_k, n_correct_r, n_total])
    acc_k = summary_count[0] / summary_count[2]
    acc_r = summary_count[1] / summary_count[2]
    update_score = tf.assign(summary_score, summary_score + [acc_k, acc_r])
    clr_summary_count = summary_count.initializer
    clr_summary_score = summary_score.initializer
    tf.summary.scalar('Accuracy_key', summary_score[0])
    tf.summary.scalar('Accuracy_roman', summary_score[1])

    with tf.name_scope('optimization'):
        # Apply warn-up learning rate
        warm_up_steps = tf.constant(4000, dtype=tf.float32)
        gstep = tf.cast(global_step, dtype=tf.float32)
        learning_rate = pow(hp.input_embed_size, -0.5) * tf.minimum(tf.pow(gstep, -0.5), gstep * tf.pow(warm_up_steps, -1.5))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           beta1=0.9,
                                           beta2=0.98,
                                           epsilon=1e-9)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # update moving_mean and moving_variance of batch normalization
        train_op = optimizer.minimize(loss)
        train_op = tf.group([train_op, update_ops])
    # Graph location and summary writers
    print('Saving graph to: %s' % hp.graph_location)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(hp.graph_location + '\\train')
    test_writer = tf.summary.FileWriter(hp.graph_location + '\\test')
    train_writer.add_graph(tf.get_default_graph())
    test_writer.add_graph(tf.get_default_graph())
    saver = tf.train.Saver(max_to_keep=1)

    # Training
    print('Train the model...')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        startTime = time.time() # start time of training
        best_score = [0.0 for _ in range(3)]
        in_succession = 0
        best_epoch = 0
        for step in range(hp.n_training_steps):
            # Training
            if step == 0:
                indices = range(n_train_sequences)
                batch_indices = [indices[x:x + hp.n_batches] for x in range(0, len(indices), hp.n_batches)]

            if step >= 2*n_iterations_per_epoch and step % n_iterations_per_epoch == 0:
                # Shuffle training data
                indices = random.sample(range(n_train_sequences), n_train_sequences)
                batch_indices = [indices[x:x + hp.n_batches] for x in range(0, len(indices), hp.n_batches)]

            batch = (train_data['pianoroll'][batch_indices[step % len(batch_indices)]],
                     train_data['len'][batch_indices[step % len(batch_indices)]],
                     train_data['key'][batch_indices[step % len(batch_indices)]],
                     train_data['roman'][batch_indices[step % len(batch_indices)]],
                     train_data['degree1'][batch_indices[step % len(batch_indices)]],
                     train_data['degree2'][batch_indices[step % len(batch_indices)]],
                     train_data['quality'][batch_indices[step % len(batch_indices)]],
                     train_data['inversion'][batch_indices[step % len(batch_indices)]],
                     train_data['label']['key'][batch_indices[step % len(batch_indices)]])

            train_run_list = [train_op, update_valid, update_loss, update_count, loss, loss_k, loss_r, pred_k, pred_r, eval_mask]
            train_feed_fict = {x_p: batch[0],
                               x_len: batch[1],
                               y_k: batch[2],
                               y_r: batch[3],
                               dropout: hp.drop,
                               is_training: True,
                               global_step: step + 1}
            _, _, _, _, train_loss, train_loss_k, train_loss_r, \
            train_pred_k, train_pred_r, train_eval_mask = sess.run(train_run_list, feed_dict=train_feed_fict)
            if step == 0:
                print('*~ loss_k %.4f, loss_r %.4f ~*' % (train_loss_k, train_loss_r))

            # Display training log & Testing
            if step > 0 and step % n_iterations_per_epoch == 0:
                sess.run([mean_loss, update_score])
                train_summary, train_loss, train_score = sess.run([merged, summary_loss, summary_score])
                sess.run([clr_summary_valid, clr_summary_loss, clr_summary_count, clr_summary_score])
                train_writer.add_summary(train_summary, step)
                print("---- step %d, epoch %d: train_loss: total %.4f (k %.4f, r %.4f), evaluation: k %.4f, r %.4f ----"
                    % (step, step // n_iterations_per_epoch, train_loss[0], train_loss[1], train_loss[2], train_score[0], train_score[1]))
                display_len = 32
                n_just = 5
                print('len =', batch[1][0])
                print('y_k'.ljust(7, ' '), ''.join([b.rjust(n_just, ' ') for b in batch[8][0, :display_len]]))
                print('y_d1'.ljust(7, ' '), ''.join([[k for k, v in degree1_dict.items() if v == b][0].rjust(n_just, ' ') for b in batch[4][0, :display_len]]))
                print('y_d2'.ljust(7, ' '), ''.join([[k for k, v in degree2_dict.items() if v == b][0].rjust(n_just, ' ') for b in batch[5][0, :display_len]]))
                print('y_q'.ljust(7, ' '), ''.join([[k for k, v in quality_dict.items() if v == b][0].rjust(n_just, ' ') for b in batch[6][0, :display_len]]))
                print('y_inv'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in batch[7][0, :display_len]]))
                print('valid'.ljust(7, ' '), ''.join(['y'.rjust(n_just, ' ') if b else 'n'.rjust(n_just, ' ') for b in train_eval_mask[0, :display_len]]))
                print('y_k'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in batch[2][0, :display_len]]))
                print('pred_k'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in train_pred_k[0, :display_len]]))
                print('y_r'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in batch[3][0, :display_len]]))
                print('pred_r'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in train_pred_r[0, :display_len]]))

                # Testing
                test_run_list = [update_valid, update_loss, update_count, pred_k, pred_r, eval_mask]
                test_feed_fict = {x_p: test_data['pianoroll'],
                                  x_len: test_data['len'],
                                  y_k: test_data['key'],
                                  y_r: test_data['roman'],
                                  dropout: 0.0,
                                  is_training: False}
                _, _, _, test_pred_k, test_pred_r, test_eval_mask = sess.run(test_run_list, feed_dict=test_feed_fict)
                sess.run([mean_loss, update_score])
                test_summary, test_loss, test_score = sess.run([merged, summary_loss, summary_score])
                sess.run([clr_summary_valid, clr_summary_loss, clr_summary_count, clr_summary_score])
                test_writer.add_summary(test_summary, step)

                sq = crm.segmentation_quality(test_data['roman'], test_pred_r, test_data['len'])
                print("==== step %d, epoch %d: test_loss: total %.4f (k %.4f, r %.4f), evaluation: k %.4f, r %.4f, sq %.4f ===="
                      % (step, step // n_iterations_per_epoch, test_loss[0], test_loss[1], test_loss[2], test_score[0], test_score[1], sq))
                sample_id = random.randint(0, n_test_sequences - 1)
                print('len =', test_data['len'][sample_id])
                print('y_k'.ljust(7, ' '), ''.join([b.rjust(n_just, ' ') for b in test_data['label']['key'][sample_id, :display_len]]))
                print('y_d1'.ljust(7, ' '), ''.join([[k for k, v in degree1_dict.items() if v == b][0].rjust(n_just, ' ') for b in test_data['degree1'][sample_id, :display_len]]))
                print('y_d2'.ljust(7, ' '), ''.join([[k for k, v in degree2_dict.items() if v == b][0].rjust(n_just, ' ') for b in test_data['degree2'][sample_id, :display_len]]))
                print('y_q'.ljust(7, ' '), ''.join([[k for k, v in quality_dict.items() if v == b][0].rjust(n_just, ' ') for b in test_data['quality'][sample_id, :display_len]]))
                print('y_inv'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in test_data['inversion'][sample_id, :display_len]]))
                print('valid'.ljust(7, ' '), ''.join(['y'.rjust(n_just, ' ') if b else 'n'.rjust(n_just, ' ') for b in test_eval_mask[sample_id, :display_len]]))
                print('y_k'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in test_data['key'][sample_id, :display_len]]))
                print('pred_k'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in test_pred_k[sample_id, :display_len]]))
                print('y_r'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in test_data['roman'][sample_id, :display_len]]))
                print('pred_r'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in test_pred_r[sample_id, :display_len]]))

                if step > 0 and sum(test_score[:2]) > sum(best_score[:2]):
                    best_score = np.concatenate([test_score, [sq]], axis=0)
                    best_epoch = step // n_iterations_per_epoch
                    in_succession = 0
                    # Save variables of the model
                    print('*saving variables...\n')
                    saver.save(sess, hp.graph_location + '\\CRNN_functional_harmony_recognition_' + hp.dataset + '_' + str(hp.test_set_id) + '.ckpt')
                else:
                    in_succession += 1
                    if in_succession > hp.n_in_succession:
                        print('Early stopping.')
                        break

        elapsed_time = time.time() - startTime
        print('\nCRNN functional harmony recognition on %s-%d:' % (hp.dataset, hp.test_set_id))
        print('training time = %.2f hr' % (elapsed_time / 3600))
        print('best epoch = ', best_epoch)
        print('best score =', np.round(best_score, 4))

def main():
    # Functional harmony recognition
    train_HT() # Harmony Transformer
    # train_BTC() # Bi-directional Transformer for Chord Recognition
    # train_CRNN() # Convolutional Recurrent Neural Network

if __name__ == '__main__':
    # Hyperparameters
    hyperparameters = namedtuple('hyperparameters',
                                 ['dataset',
                                  'test_set_id',
                                  'graph_location',
                                  'n_steps',
                                  'input_embed_size',
                                  'n_layers',
                                  'n_heads',
                                  'train_sequence_with_overlap',
                                  'initial_learning_rate',
                                  'drop',
                                  'n_batches',
                                  'n_training_steps',
                                  'n_in_succession',
                                  'annealing_rate'])

    hp = hyperparameters(dataset='Preludes', # {'BPS_FH', 'Preludes'}
                         test_set_id=1,
                         graph_location='model',
                         n_steps=128,
                         input_embed_size=128,
                         n_layers=2,
                         n_heads=4,
                         train_sequence_with_overlap=True,
                         initial_learning_rate=1e-4,
                         drop=0.1,
                         n_batches=40,
                         n_training_steps=100000,
                         n_in_succession=10,
                         annealing_rate=1.1)

    main()

