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

def load_data_symbol(dir, test_set_id=1, sequence_with_overlap=True):
    if test_set_id not in [1,2,3,4]:
        print('Invalid testing_set_id.')
        exit(1)

    print("Load chord symbol data...")
    print('test_set_id =', test_set_id)
    with open(dir, 'rb') as file:
        corpus_aug_reshape = pickle.load(file)
    print('keys in corpus_aug_reshape[\'shift_id\'][\'op\'] =', corpus_aug_reshape['shift_0']['1'].keys())

    shift_list = sorted(corpus_aug_reshape.keys())
    number_of_pieces = len(corpus_aug_reshape['shift_0'].keys())
    train_op_list = [str(i+1) for i in range(number_of_pieces) if i % 4 + 1 != test_set_id]
    test_op_list = [str(i+1) for i in range(number_of_pieces) if i % 4 + 1 == test_set_id]
    print('shift_list =', shift_list)
    print('train_op_list =', train_op_list)
    print('test_op_list =', test_op_list)

    overlap = int(sequence_with_overlap)

    # Training set
    train_data = {'pianoroll': np.concatenate([corpus_aug_reshape[shift_id][op]['pianoroll'][overlap] for shift_id in shift_list for op in train_op_list], axis=0),
                  'tonal_centroid': np.concatenate([corpus_aug_reshape[shift_id][op]['tonal_centroid'][overlap] for shift_id in shift_list for op in train_op_list], axis=0),
                  'len': np.concatenate([corpus_aug_reshape[shift_id][op]['len'][overlap] for shift_id in shift_list for op in train_op_list], axis=0),
                  'label': np.concatenate([corpus_aug_reshape[shift_id][op]['label'][overlap] for shift_id in shift_list for op in train_op_list], axis=0)}

    train_data_label_root = np.zeros_like(train_data['label'], dtype=np.int32)
    train_data_label_tquality = np.zeros_like(train_data['label'], dtype=np.int32)

    for k, v in root_dict.items():
        train_data_label_root[train_data['label']['root'] == k] = v
    for k, v in tquality_dict.items():
        train_data_label_tquality[train_data['label']['tquality'] == k] = v
    train_data['root'] = train_data_label_root
    train_data['tquality'] = train_data_label_tquality
    train_data['tchord'] = train_data['tquality'] * 12 + train_data['root']
    if 'O' not in tquality_dict.keys():
        train_data['tchord'][train_data['root'] == root_dict['pad']] = tquality_dict['pad'] * 12
    else:
        train_data['tchord'][train_data['tquality'] == tquality_dict['O']] = tquality_dict['O'] * 12
        train_data['tchord'][train_data['root'] == root_dict['pad']] = tquality_dict['O'] * 12 + 1

    # Testing set
    test_data = {'pianoroll': np.concatenate([corpus_aug_reshape['shift_0'][op]['pianoroll'][0] for op in test_op_list], axis=0),
                 'tonal_centroid': np.concatenate([corpus_aug_reshape['shift_0'][op]['tonal_centroid'][0] for op in test_op_list], axis=0),
                 'len': np.concatenate([corpus_aug_reshape['shift_0'][op]['len'][0] for op in test_op_list], axis=0),
                 'label': np.concatenate([corpus_aug_reshape['shift_0'][op]['label'][0] for op in test_op_list], axis=0)}

    test_data_label_root = np.zeros_like(test_data['label'], dtype=np.int32)
    test_data_label_tquality = np.zeros_like(test_data['label'], dtype=np.int32)
    for k, v in root_dict.items():
        test_data_label_root[test_data['label']['root'] == k] = v
    for k, v in tquality_dict.items():
        test_data_label_tquality[test_data['label']['tquality'] == k] = v
    test_data['root'] = test_data_label_root
    test_data['tquality'] = test_data_label_tquality
    test_data['tchord'] = test_data['tquality'] * 12 + test_data['root']
    if 'O' not in tquality_dict.keys():
        test_data['tchord'][test_data['root'] == root_dict['pad']] = tquality_dict['pad'] * 12
    else:
        test_data['tchord'][test_data['tquality'] == tquality_dict['O']] = tquality_dict['O'] * 12
        test_data['tchord'][test_data['root'] == root_dict['pad']] = tquality_dict['O'] * 12 + 1

    print('train_data: ', [(k, v.shape)for k, v in train_data.items()])
    print('test_data: ', [(k, v.shape) for k, v in test_data.items()])
    print('label fields:', test_data['label'].dtype)
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
    print('Run HT chord recognition on %s-%d...' % (hp.dataset, hp.test_set_id))

    # Load training and testing data
    train_data, test_data = load_data_symbol(dir=hp.dataset + '_preprocessed_data_MIREX_Mm.pickle', test_set_id=hp.test_set_id, sequence_with_overlap=hp.train_sequence_with_overlap)
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
        y_tc = tf.placeholder(tf.int32, [None, hp.n_steps], name="tchord")
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
        dec_input_embed = tf.layers.dropout(dec_input_embed, rate=dropout, training=is_training)
        chord_logits = tf.layers.dense(dec_input_embed, hp.n_chord_classes, name='output_dense')

    with tf.name_scope('loss'):
        # Chord change
        loss_cc = 1.5 * tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.cast(y_cc, tf.float32), logits=slope*chord_change_logits, weights=source_mask)

        # Chord symbol
        loss_tc = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(y_tc, hp.n_chord_classes), logits=chord_logits, weights=target_mask)

        # Total loss
        loss = loss_cc + loss_tc
    valid = tf.reduce_sum(target_mask)
    summary_loss = tf.Variable([0.0, 0.0, 0.0], trainable=False, dtype=tf.float32)
    summary_valid = tf.Variable(0, trainable=False, dtype=tf.float32)
    update_loss = tf.assign(summary_loss, summary_loss + valid * [loss, loss_cc, loss_tc])
    update_valid = tf.assign(summary_valid, summary_valid + valid)
    mean_loss = tf.assign(summary_loss, summary_loss / summary_valid)
    clr_summary_loss = summary_loss.initializer
    clr_summary_valid = summary_valid.initializer
    tf.summary.scalar('Loss_total', summary_loss[0])
    tf.summary.scalar('Loss_chord_change', summary_loss[1])
    tf.summary.scalar('Loss_chord', summary_loss[2])

    with tf.name_scope('evaluation'):
        chord_mask = tf.cast(target_mask, tf.bool)
        chord_mask = tf.logical_and(chord_mask, tf.less(y_tc, tquality_dict['O'] * 12))

        # Chord change
        pred_cc = tf.cast(tf.round(tf.sigmoid(slope*chord_change_logits)), tf.int32)
        pred_cc_mask = tf.boolean_mask(pred_cc, tf.cast(source_mask, tf.bool))
        y_cc_mask = tf.boolean_mask(y_cc, tf.cast(source_mask, tf.bool))
        TP_cc, FP_cc, FN_cc = compute_pre_PRF(pred_cc_mask, y_cc_mask)

        # Chord
        pred_tc = tf.argmax(chord_logits, axis=2, output_type=tf.int32)
        pred_tc_correct = tf.equal(pred_tc, y_tc)
        pred_tc_correct_mask = tf.boolean_mask(tensor=pred_tc_correct, mask=chord_mask)
        correct = tf.reduce_sum(tf.cast(pred_tc_correct_mask, tf.float32))
        total = tf.cast(tf.size(pred_tc_correct_mask), tf.float32)
    summary_count = tf.Variable([0.0 for _ in range(5)], trainable=False, dtype=tf.float32)
    summary_score = tf.Variable([0.0 for _ in range(4)], trainable=False, dtype=tf.float32)
    update_count = tf.assign(summary_count, summary_count + [correct, total, TP_cc, FP_cc, FN_cc])
    acc_tc = summary_count[0] / summary_count[1]
    P_cc, R_cc, F1_cc = comput_PRF_with_pre(summary_count[2], summary_count[3], summary_count[4])
    update_score = tf.assign(summary_score, summary_score + [acc_tc, P_cc, R_cc, F1_cc,])
    clr_summary_count = summary_count.initializer
    clr_summary_score = summary_score.initializer
    tf.summary.scalar('Accuracy_tchord', summary_score[0])
    tf.summary.scalar('Precision_chord_change', summary_score[1])
    tf.summary.scalar('Recall_chord_change', summary_score[2])
    tf.summary.scalar('F1_chord_change', summary_score[3])

    with tf.name_scope('optimization'):
        # Apply warm-up learning rate
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
        startTime = time.time()
        best_score = [0.0 for _ in range(5)]
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

            if step >= n_iterations_per_epoch and step % n_iterations_per_epoch == 0:
                # Shuffle training data
                indices = random.sample(range(n_train_sequences), n_train_sequences)
                batch_indices = [indices[x:x + hp.n_batches] for x in range(0, len(indices), hp.n_batches)]

            batch = (train_data['pianoroll'][batch_indices[step % len(batch_indices)]],
                     train_data['len'][batch_indices[step % len(batch_indices)]],
                     train_data['label']['chord_change'][batch_indices[step % len(batch_indices)]],
                     train_data['tchord'][batch_indices[step % len(batch_indices)]],
                     train_data['root'][batch_indices[step % len(batch_indices)]],
                     train_data['tquality'][batch_indices[step % len(batch_indices)]])

            train_run_list = [train_op, update_valid, update_loss, update_count, loss, loss_cc, loss_tc, pred_cc, pred_tc, chord_mask, enc_weights, dec_weights]
            train_feed_fict = {x_p: batch[0],
                               x_len: batch[1],
                               y_cc: batch[2],
                               y_tc: batch[3],
                               dropout: hp.drop,
                               is_training: True,
                               global_step: step + 1,
                               slope: annealing_slope}
            _, _, _, _, train_loss, train_loss_cc, train_loss_tc, train_pred_cc, train_pred_tc, train_chord_mask, enc_w, dec_w = sess.run(train_run_list, feed_dict=train_feed_fict)
            if step == 0:
                print('*~ loss_cc %.4f, loss_tc %.4f ~*' % (train_loss_cc, train_loss_tc))

            # Display training log & Testing
            if step > 0 and step % n_iterations_per_epoch == 0:
                sess.run([mean_loss, update_score])
                train_summary, train_loss, train_score = sess.run([merged, summary_loss, summary_score])
                sess.run([clr_summary_valid, clr_summary_loss, clr_summary_count, clr_summary_score])
                train_writer.add_summary(train_summary, step)
                print("---- step %d, epoch %d: train_loss: total %.4f, cc %.4f, tc %.4f, evaluation: tc %.4f, cc (P %.4f, R %.4f, F1 %.4f) ----"
                    % (step, step // n_iterations_per_epoch, train_loss[0], train_loss[1], train_loss[2], train_score[0], train_score[1], train_score[2], train_score[3]))
                print('enc_w =', enc_w, 'dec_w =', dec_w)
                display_len = 64
                print('len =', batch[1][0])
                print('y_root'.ljust(7, ' '), ''.join([[k for k, v in root_dict.items() if v == b][0].rjust(3, ' ') for b in batch[4][0, :display_len]]))
                print('y_tq'.ljust(7, ' '), ''.join([[k for k, v in tquality_dict.items() if v == b][0].rjust(3, ' ') for b in batch[5][0, :display_len]]))
                print('valid'.ljust(7, ' '), ''.join(['y'.rjust(3, ' ') if b else 'n'.rjust(3, ' ') for b in train_chord_mask[0, :display_len]]))
                print('y_cc'.ljust(7, ' '), ''.join([str(b).rjust(3, ' ') for b in batch[2][0, :display_len]]))
                print('pred_cc'.ljust(7, ' '), ''.join([str(b).rjust(3, ' ') for b in train_pred_cc[0, :display_len]]))
                print('y_tc'.ljust(7, ' '), ''.join([str(b).rjust(3, ' ') for b in batch[3][0, :display_len]]))
                print('pred_tc'.ljust(7, ' '), ''.join([str(b).rjust(3, ' ') for b in train_pred_tc[0, :display_len]]))

                # Testing
                test_run_list = [update_valid, update_loss, update_count, pred_cc, pred_tc, chord_mask]
                test_feed_fict = {x_p: test_data['pianoroll'],
                                  x_len: test_data['len'],
                                  y_cc: test_data['label']['chord_change'],
                                  y_tc: test_data['tchord'],
                                  dropout: 0.0,
                                  is_training: False,
                                  slope: annealing_slope}
                _, _, _, test_pred_cc, test_pred_tc, test_chord_mask = sess.run(test_run_list, feed_dict=test_feed_fict)
                sess.run([mean_loss, update_score])
                test_summary, test_loss, test_score = sess.run([merged, summary_loss, summary_score])
                sess.run([clr_summary_valid, clr_summary_loss, clr_summary_count, clr_summary_score])
                test_writer.add_summary(test_summary, step)

                sq = crm.segmentation_quality(test_data['tchord'], test_pred_tc, test_data['len'])
                print("==== step %d, epoch %d: test_loss: total %.4f, cc %.4f, tc %.4f, evaluation: tc %.4f, cc (P %.4f, R %.4f, F1 %.4f), sq %.4f ===="
                      % (step, step // n_iterations_per_epoch, test_loss[0], test_loss[1], test_loss[2], test_score[0], test_score[1], test_score[2], test_score[3], sq))
                sample_id = random.randint(0, n_test_sequences - 1)
                print('len =', test_data['len'][sample_id])
                print('y_root'.ljust(7, ' '), ''.join([[k for k, v in root_dict.items() if v == b][0].rjust(3, ' ') for b in test_data['root'][sample_id, :display_len]]))
                print('y_tq'.ljust(7, ' '), ''.join([[k for k, v in tquality_dict.items() if v == b][0].rjust(3, ' ') for b in test_data['tquality'][sample_id, :display_len]]))
                print('valid'.ljust(7, ' '), ''.join(['y'.rjust(3, ' ') if b else 'n'.rjust(3, ' ') for b in test_chord_mask[sample_id, :display_len]]))
                print('y_cc'.ljust(7, ' '), ''.join([str(b).rjust(3, ' ') for b in test_data['label']['chord_change'][sample_id, :display_len]]))
                print('pred_cc'.ljust(7, ' '), ''.join([str(b).rjust(3, ' ') for b in test_pred_cc[sample_id, :display_len]]))
                print('y_tc'.ljust(7, ' '), ''.join([str(b).rjust(3, ' ') for b in test_data['tchord'][sample_id, :display_len]]))
                print('pred_tc'.ljust(7, ' '), ''.join([str(b).rjust(3, ' ') for b in test_pred_tc[sample_id, :display_len]]))

                if step > 0 and (test_score[0] + sq) > (best_score[0] + best_score[-1]):
                    best_score = np.concatenate([test_score, [sq]], axis=0)
                    best_epoch = step // n_iterations_per_epoch
                    best_slope = annealing_slope
                    in_succession = 0

                    # Save variables of the model
                    print('*saving variables...\n')
                    saver.save(sess, hp.graph_location + '\\HT_chord_recognition_' + hp.dataset + '_' + str(hp.test_set_id) + '.ckpt')
                else:
                    in_succession += 1
                    if in_succession > hp.n_in_succession:
                        print('Early stopping.')
                        break

        # saver.save(sess, hp.graph_location + '\\HT_chord_recognition_train_model.ckpt')
        elapsed_time = time.time() - startTime
        print('\nHT chord symbol recognition on %s-%d:' % (hp.dataset, hp.test_set_id))
        print('training time = %.2f hr' % (elapsed_time / 3600))
        print('best epoch = ', best_epoch)
        print('best score =', np.round(best_score, 4))
        print('best slope =', best_slope)

def train_BTC():
    print('Run BTC chord recognition on %s-%d...' % (hp.dataset, hp.test_set_id))

    # Load training and testing data
    train_data, test_data = load_data_symbol(dir=hp.dataset + '_preprocessed_data_MIREX_Mm.pickle', test_set_id=hp.test_set_id, sequence_with_overlap=hp.train_sequence_with_overlap)
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
        y_tc = tf.placeholder(tf.int32, [None, hp.n_steps], name="tchord")
        dropout = tf.placeholder(dtype=tf.float32, name="dropout_rate")
        is_training = tf.placeholder(dtype=tf.bool, name="is_training")
        global_step = tf.placeholder(dtype=tf.int32, name='global_step')

    with tf.name_scope('model'):
        x_in = tf.cast(x_p, tf.float32)
        source_mask = tf.sequence_mask(lengths=x_len, maxlen=hp.n_steps, dtype=tf.float32) # [n_batches, n_steps]
        input_embed = crm.BTC(x_in, source_mask, dropout, is_training, hp)

    with tf.variable_scope("output_projection"):
        input_embed = tf.layers.dropout(input_embed, rate=dropout, training=is_training)
        chord_logits = tf.layers.dense(input_embed, hp.n_chord_classes)

    with tf.name_scope('loss'):
        loss = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(y_tc, hp.n_chord_classes), logits=chord_logits, weights=source_mask)
    valid = tf.reduce_sum(source_mask)
    summary_loss = tf.Variable(0.0, trainable=False, dtype=tf.float32)
    summary_valid = tf.Variable(0, trainable=False, dtype=tf.float32)
    update_loss = tf.assign(summary_loss, summary_loss + valid * loss)
    update_valid = tf.assign(summary_valid, summary_valid + valid)
    mean_loss = tf.assign(summary_loss, summary_loss / summary_valid)
    clr_summary_loss = summary_loss.initializer
    clr_summary_valid = summary_valid.initializer
    tf.summary.scalar('Loss', summary_loss)

    with tf.name_scope('evaluation'):
        chord_mask = tf.cast(source_mask, tf.bool)
        chord_mask = tf.logical_and(chord_mask, tf.less(y_tc, tquality_dict['O'] * 12))
        pred_tc = tf.argmax(chord_logits, axis=2, output_type=tf.int32)
        pred_tc_correct = tf.equal(pred_tc, y_tc)
        pred_tc_correct_mask = tf.boolean_mask(tensor=pred_tc_correct, mask=chord_mask)
        correct = tf.reduce_sum(tf.cast(pred_tc_correct_mask, tf.float32))
        total = tf.cast(tf.size(pred_tc_correct_mask), tf.float32)
    summary_count = tf.Variable([0.0 for _ in range(2)], trainable=False, dtype=tf.float32)
    summary_score = tf.Variable(0.0, trainable=False, dtype=tf.float32)
    update_count = tf.assign(summary_count, summary_count + [correct, total])
    acc_tc = summary_count[0] / summary_count[1]
    compute_score = tf.assign(summary_score, summary_score + acc_tc)
    clr_summary_count = summary_count.initializer
    clr_summary_score = summary_score.initializer
    tf.summary.scalar('Accuracy_tchord', summary_score)

    with tf.name_scope('optimization'):
        # Apply warm-up learning rate
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
        best_score = [0.0, 0.0]
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
                     train_data['tchord'][batch_indices[step % len(batch_indices)]],
                     train_data['root'][batch_indices[step % len(batch_indices)]],
                     train_data['tquality'][batch_indices[step % len(batch_indices)]])

            train_run_list = [train_op, update_valid, update_loss, update_count, loss, pred_tc, chord_mask]
            train_feed_fict = {x_p: batch[0],
                               x_len: batch[1],
                               y_tc: batch[2],
                               dropout: hp.drop,
                               is_training: True,
                               global_step: step + 1}
            _, _, _, _, train_loss, train_pred_tc, train_chord_mask = sess.run(train_run_list, feed_dict=train_feed_fict)
            if step == 0:
                print('*~ loss %.4f ~*' % (train_loss))

            # Display training log & Testing
            if step > 0 and step % n_iterations_per_epoch == 0:
                sess.run([mean_loss, compute_score])
                train_summary, train_loss, train_score = sess.run([merged, summary_loss, summary_score])
                sess.run([clr_summary_valid, clr_summary_loss, clr_summary_count, clr_summary_score])
                train_writer.add_summary(train_summary, step)
                print("---- step %d, epoch %d: train_loss: %.4f, evaluation: tc %.4f ----"
                    % (step, step // n_iterations_per_epoch, train_loss, train_score))
                display_len = 64
                print('len =', batch[1][0])
                print('y_root'.ljust(7, ' '), ''.join([[k for k, v in root_dict.items() if v == b][0].rjust(3, ' ') for b in batch[3][0, :display_len]]))
                print('y_tq'.ljust(7, ' '), ''.join([[k for k, v in tquality_dict.items() if v == b][0].rjust(3, ' ') for b in batch[4][0, :display_len]]))
                print('valid'.ljust(7, ' '), ''.join(['y'.rjust(3, ' ') if b else 'n'.rjust(3, ' ') for b in train_chord_mask[0, :display_len]]))
                print('y_tc'.ljust(7, ' '), ''.join([str(b).rjust(3, ' ') for b in batch[2][0, :display_len]]))
                print('pred_tc'.ljust(7, ' '), ''.join([str(b).rjust(3, ' ') for b in train_pred_tc[0, :display_len]]))

                # Testing
                test_run_list = [update_valid, update_loss, update_count, pred_tc, chord_mask]
                test_feed_fict = {x_p: test_data['pianoroll'],
                                  x_len: test_data['len'],
                                  y_tc: test_data['tchord'],
                                  dropout: 0.0,
                                  is_training: False}
                _, _, _, test_pred_tc, test_chord_mask = sess.run(test_run_list, feed_dict=test_feed_fict)
                sess.run([mean_loss, compute_score])
                test_summary, test_loss, test_score = sess.run([merged, summary_loss, summary_score])
                sess.run([clr_summary_valid, clr_summary_loss, clr_summary_count, clr_summary_score])
                test_writer.add_summary(test_summary, step)

                sq = crm.segmentation_quality(test_data['tchord'], test_pred_tc, test_data['len'])
                print("==== step %d, epoch %d: test_loss: %.4f, evaluation: tc %.4f, sq %.4f ===="
                      % (step, step // n_iterations_per_epoch, test_loss, test_score, sq))
                sample_id = random.randint(0, n_test_sequences - 1)
                print('len =', test_data['len'][sample_id])
                print('y_root'.ljust(7, ' '), ''.join([[k for k, v in root_dict.items() if v == b][0].rjust(3, ' ') for b in test_data['root'][sample_id, :display_len]]))
                print('y_tq'.ljust(7, ' '), ''.join([[k for k, v in tquality_dict.items() if v == b][0].rjust(3, ' ') for b in test_data['tquality'][sample_id, :display_len]]))
                print('valid'.ljust(7, ' '), ''.join(['y'.rjust(3, ' ') if b else 'n'.rjust(3, ' ') for b in test_chord_mask[sample_id, :display_len]]))
                print('y_tc'.ljust(7, ' '), ''.join([str(b).rjust(3, ' ') for b in test_data['tchord'][sample_id, :display_len]]))
                print('pred_tc'.ljust(7, ' '), ''.join([str(b).rjust(3, ' ') for b in test_pred_tc[sample_id, :display_len]]))

                if step > 0 and (test_score + sq) > sum(best_score):
                    best_score = [test_score, sq]
                    best_epoch = step // n_iterations_per_epoch
                    in_succession = 0
                    # Save variables of the model
                    print('*saving variables...\n')
                    saver.save(sess, hp.graph_location + '\\BTC_chord_recognition_' + hp.dataset + '_' + str(hp.test_set_id) + '.ckpt')
                else:
                    in_succession += 1
                    if in_succession > hp.n_in_succession:
                        print('Early stopping.')
                        break

        elapsed_time = time.time() - startTime
        print('\nBTC chord symbol recognition on %s-%d:' % (hp.dataset, hp.test_set_id))
        print('training time = %.2f hr' % (elapsed_time / 3600))
        print('best epoch = ', best_epoch)
        print('best score =', np.round(best_score, 4))

def train_CRNN():
    print('Run CRNN chord recognition on %s-%d...' % (hp.dataset, hp.test_set_id))

    # Load training and testing data
    train_data, test_data = load_data_symbol(dir=hp.dataset + '_preprocessed_data_MIREX_Mm.pickle', test_set_id=hp.test_set_id, sequence_with_overlap=hp.train_sequence_with_overlap)
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
        y_tc = tf.placeholder(tf.int32, [None, hp.n_steps], name="tchord")
        dropout = tf.placeholder(dtype=tf.float32, name="dropout_rate")
        is_training = tf.placeholder(dtype=tf.bool, name="is_training")
        global_step = tf.placeholder(dtype=tf.int32, name='global_step')

    with tf.name_scope('model'):
        x_in = tf.cast(x_p, tf.float32)
        source_mask = tf.sequence_mask(lengths=x_len, maxlen=hp.n_steps, dtype=tf.float32) # [n_batches, n_steps]
        input_embed = crm.CRNN(x_in, x_len, dropout, is_training, hp)

    with tf.variable_scope("output_projection"):
        input_embed = tf.layers.dropout(input_embed, rate=dropout, training=is_training)
        chord_logits = tf.layers.dense(input_embed, hp.n_chord_classes)

    with tf.name_scope('loss'):
        loss = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(y_tc, hp.n_chord_classes), logits=chord_logits, weights=source_mask)
    valid = tf.reduce_sum(source_mask)
    summary_loss = tf.Variable(0.0, trainable=False, dtype=tf.float32)
    summary_valid = tf.Variable(0, trainable=False, dtype=tf.float32)
    update_loss = tf.assign(summary_loss, summary_loss + valid * loss)
    update_valid = tf.assign(summary_valid, summary_valid + valid)
    mean_loss = tf.assign(summary_loss, summary_loss / summary_valid)
    clr_summary_loss = summary_loss.initializer
    clr_summary_valid = summary_valid.initializer
    tf.summary.scalar('Loss_total', summary_loss)

    with tf.name_scope('evaluation'):
        chord_mask = tf.cast(source_mask, tf.bool)
        chord_mask = tf.logical_and(chord_mask, tf.less(y_tc, tquality_dict['O'] * 12))
        pred_tc = tf.argmax(chord_logits, axis=2, output_type=tf.int32)
        pred_tc_correct = tf.equal(pred_tc, y_tc)
        pred_tc_correct_mask = tf.boolean_mask(tensor=pred_tc_correct, mask=chord_mask)
        correct = tf.reduce_sum(tf.cast(pred_tc_correct_mask, tf.float32))
        total = tf.cast(tf.size(pred_tc_correct_mask), tf.float32)
    summary_count = tf.Variable([0.0 for _ in range(2)], trainable=False, dtype=tf.float32)
    summary_score = tf.Variable(0.0, trainable=False, dtype=tf.float32)
    update_count = tf.assign(summary_count, summary_count + [correct, total])
    acc_tc = summary_count[0] / summary_count[1]
    compute_score = tf.assign(summary_score, summary_score + acc_tc)
    clr_summary_count = summary_count.initializer
    clr_summary_score = summary_score.initializer
    tf.summary.scalar('Accuracy_tchord', summary_score)

    with tf.name_scope('optimization'):
        # Apply warm-up learning rate
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
        best_score = [0.0, 0.0]
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
                     train_data['tchord'][batch_indices[step % len(batch_indices)]],
                     train_data['root'][batch_indices[step % len(batch_indices)]],
                     train_data['tquality'][batch_indices[step % len(batch_indices)]])

            train_run_list = [train_op, update_valid, update_loss, update_count, loss, pred_tc, chord_mask]
            train_feed_fict = {x_p: batch[0],
                               x_len: batch[1],
                               y_tc: batch[2],
                               dropout: hp.drop,
                               is_training: True,
                               global_step: step + 1}
            _, _, _, _, train_loss, train_pred_tc, train_chord_mask = sess.run(train_run_list, feed_dict=train_feed_fict)
            if step == 0:
                print('*~ loss %.4f ~*' % (train_loss))

            # Display training log & Testing
            if step > 0 and step % n_iterations_per_epoch == 0:
                sess.run([mean_loss, compute_score])
                train_summary, train_loss, train_score = sess.run([merged, summary_loss, summary_score])
                sess.run([clr_summary_valid, clr_summary_loss, clr_summary_count, clr_summary_score])
                train_writer.add_summary(train_summary, step)
                print("---- step %d, epoch %d: train_loss: %.4f, evaluation: tc %.4f ----"
                    % (step, step // n_iterations_per_epoch, train_loss, train_score))
                display_len = 64
                print('len =', batch[1][0])
                print('y_root'.ljust(7, ' '), ''.join([[k for k, v in root_dict.items() if v == b][0].rjust(3, ' ') for b in batch[3][0, :display_len]]))
                print('y_tq'.ljust(7, ' '), ''.join([[k for k, v in tquality_dict.items() if v == b][0].rjust(3, ' ') for b in batch[4][0, :display_len]]))
                print('valid'.ljust(7, ' '), ''.join(['y'.rjust(3, ' ') if b else 'n'.rjust(3, ' ') for b in train_chord_mask[0, :display_len]]))
                print('y_tc'.ljust(7, ' '), ''.join([str(b).rjust(3, ' ') for b in batch[2][0, :display_len]]))
                print('pred_tc'.ljust(7, ' '), ''.join([str(b).rjust(3, ' ') for b in train_pred_tc[0, :display_len]]))

                # Testing
                test_run_list = [update_valid, update_loss, update_count, pred_tc, chord_mask]
                test_feed_fict = {x_p: test_data['pianoroll'],
                                  x_len: test_data['len'],
                                  y_tc: test_data['tchord'],
                                  dropout: 0.0,
                                  is_training: False}
                _, _, _, test_pred_tc, test_chord_mask = sess.run(test_run_list, feed_dict=test_feed_fict)
                sess.run([mean_loss, compute_score])
                test_summary, test_loss, test_score = sess.run([merged, summary_loss, summary_score])
                sess.run([clr_summary_valid, clr_summary_loss, clr_summary_count, clr_summary_score])
                test_writer.add_summary(test_summary, step)

                sq = crm.segmentation_quality(test_data['tchord'], test_pred_tc, test_data['len'])
                print("==== step %d, epoch %d: test_loss: %.4f, evaluation: tc %.4f, sq %.4f ===="
                      % (step, step // n_iterations_per_epoch, test_loss, test_score, sq))
                sample_id = random.randint(0, n_test_sequences - 1)
                print('len =', test_data['len'][sample_id])
                print('y_root'.ljust(7, ' '), ''.join([[k for k, v in root_dict.items() if v == b][0].rjust(3, ' ') for b in test_data['root'][sample_id, :display_len]]))
                print('y_tq'.ljust(7, ' '), ''.join([[k for k, v in tquality_dict.items() if v == b][0].rjust(3, ' ') for b in test_data['tquality'][sample_id, :display_len]]))
                print('valid'.ljust(7, ' '), ''.join(['y'.rjust(3, ' ') if b else 'n'.rjust(3, ' ') for b in test_chord_mask[sample_id, :display_len]]))
                print('y_tc'.ljust(7, ' '), ''.join([str(b).rjust(3, ' ') for b in test_data['tchord'][sample_id, :display_len]]))
                print('pred_tc'.ljust(7, ' '), ''.join([str(b).rjust(3, ' ') for b in test_pred_tc[sample_id, :display_len]]))

                if step > 0 and test_score + sq > sum(best_score):
                    best_score = [test_score, sq]
                    best_epoch = step // n_iterations_per_epoch
                    in_succession = 0
                    # Save variables of the model
                    print('*saving variables...\n')
                    saver.save(sess, hp.graph_location + '\\CRNN_chord_recognition_' + hp.dataset + '_' + str(hp.test_set_id) + '.ckpt')
                else:
                    in_succession += 1
                    if in_succession > hp.n_in_succession:
                        print('Early stopping.')
                        break

        elapsed_time = time.time() - startTime
        print('\nCRNN chord symbol recognition on %s-%d:' % (hp.dataset, hp.test_set_id))
        print('training time = %.2f hr' % (elapsed_time / 3600))
        print('best epoch = ', best_epoch)
        print('best score =', np.round(best_score, 4))

def main():
    # Chord symbol recognition
    train_HT() # Harmony Transformer
    # train_BTC() # Bi-directional Transformer for Chord Recognition
    # train_CRNN() # Convolutional Recurrent Neural Network

if __name__ == '__main__':
    root_dict = {'C': 0, 'C+': 1, 'D': 2, 'D+': 3, 'E': 4, 'F': 5, 'F+': 6, 'G': 7, 'G+': 8, 'A': 9, 'A+': 10, 'B': 11, 'pad': 12}
    tquality_dict = {'M': 0, 'm': 1, 'O': 2, 'pad': 3}  # 'O' stands for 'others'
    n_chord_classes = 24 + 1  # 24 major-minor modes plus 1 others
    
    # Hyperparameters
    hyperparameters = namedtuple('hyperparameters',
                                 ['dataset',
                                  'test_set_id',
                                  'graph_location',
                                  'n_root_classes',
                                  'n_tquality_classes',
                                  'n_chord_classes',
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
                         test_set_id=1, # {1, 2, 3, 4}
                         graph_location='model',
                         n_root_classes=len(root_dict.keys()),
                         n_tquality_classes=len(tquality_dict.keys()),
                         n_chord_classes=n_chord_classes,
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

