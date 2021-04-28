import numpy as np
import itertools
import numpy.lib.recfunctions as rfn
import math
import matplotlib.pyplot as plt
import xlrd
import copy
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import euclidean
import pickle
from collections import Counter, OrderedDict
import os

def strided_axis1(a, window, hop):
    n_pad = window // 2
    b = np.lib.pad(a, ((0, 0), (n_pad, n_pad)), 'constant', constant_values=0)
    # Length of 3D output array along its axis=1
    nd1 = int((b.shape[1] - window) / hop) + 1
    # Store shape and strides info
    m, n = b.shape
    s0, s1 = b.strides
    # Finally use strides to get the 3D array view
    return np.lib.stride_tricks.as_strided(b, shape=(nd1, m, window), strides=(s1 * hop, s0, s1))

def pianoroll2chromagram(pianoRoll, smoothing=False, window=17):
    """pianoRoll with shape = [88, time]"""
    pianoRoll_T = np.transpose(pianoRoll.astype(np.float32)) # [time, 88]
    pianoRoll_T_pad = np.pad(pianoRoll_T, [(0, 0), (9, 11)], 'constant') # [time, 108]
    pianoRoll_T = np.split(pianoRoll_T_pad, indices_or_sections=(pianoRoll_T_pad.shape[1]//12), axis=1) # [9, time, 12]
    chromagram_T = np.sum(pianoRoll_T, axis=0) # [time,  12]
    if smoothing:
        n_pad = window // 2
        chromagram_T_pad = np.pad(chromagram_T, ((n_pad, n_pad), (0, 0)), 'constant', constant_values=0)
        chromagram_T_smoothed = np.array([np.mean(chromagram_T_pad[(time+n_pad)-window//2:(time+n_pad)+window//2+1, :], axis=0) for time in range(chromagram_T.shape[0])])
        chromagram_T = chromagram_T_smoothed # [time,  12]
    L1_norm = chromagram_T.sum(axis=1) # [time]
    L1_norm[L1_norm == 0] = 1 # replace zeros with ones
    chromagram_T_norm = chromagram_T / L1_norm[:, np.newaxis] # L1 normalization, [time, 12]
    chromagram = np.transpose(chromagram_T_norm) # [12, time]
    return chromagram

def load_pieces(resolution=4):
    """
    :param resolution: time resolution, default = 4 (16th note as 1unit in piano roll)
    :param representType: 'pianoroll' or 'onset_duration'
    :return: pieces, tdeviation
    """
    print('Message: load note data ...')
    dir = os.getcwd() + "\\BPS_FH_Dataset\\"
    dt = [('onset', 'float'), ('pitch', 'int'), ('mPitch', 'int'), ('duration', 'float'), ('staffNum', 'int'), ('measure', 'int')] # datatype
    highest_pitch = 0
    lowest_pitch = 256
    pieces = {str(k): {'pianoroll': None, 'chromagram': None, 'start_time': None} for k in range(1,33)}
    for i in range(1,33):
        fileDir = dir + str(i) + "\\notes.csv"
        notes = np.genfromtxt(fileDir, delimiter=',', dtype=dt) # read notes from .csv file
        total_length = math.ceil((max(notes['onset'] + notes['duration']) - notes[0]['onset']) * resolution) # length of pianoroll
        start_time = notes[0]['onset']
        pianoroll = np.zeros(shape=[88, total_length], dtype=np.int32) # piano range: 21-108 (A0 to C8)
        for note in notes:
            if note['duration'] == 0: # "Ornament"
                continue
            pitch = note['pitch']
            onset = int(math.floor((note['onset'] - start_time)*resolution))
            end = int(math.ceil((note['onset'] + note['duration'] - start_time)*resolution))
            if onset == end:
                print('no', i)
                print('Error: note onset = note end')
                exit(1)
            time = range(onset, end)
            pianoroll[pitch-21, time] = 1 # add note to representation

            if pitch > highest_pitch:
                highest_pitch = pitch
            if pitch < lowest_pitch:
                lowest_pitch = pitch

        pieces[str(i)]['pianoroll'] = pianoroll # [88, time]
        pieces[str(i)]['chromagram'] = pianoroll2chromagram(pianoroll) # [12, time]
        pieces[str(i)]['start_time'] = start_time
    print('lowest pitch =', lowest_pitch, 'highest pitch = ', highest_pitch)
    return pieces


def load_chord_labels(vocabulary='MIREX_Mm'):
    print('Message: load chord labels...')
    dir = os.getcwd() + "\\BPS_FH_Dataset\\"
    dt = [('onset', 'float'), ('duration', 'float'), ('key', '<U10'), ('degree1', '<U10'), ('degree2', '<U10'), ('quality', '<U10'), ('inversion', 'int'), ('rchord', '<U10')] # datatype
    chord_labels = {str(k): None for k in range(1,33)}
    for i in range(1,33):
        fileDir = dir + str(i) + "\\chords.xlsx"
        workbook = xlrd.open_workbook(fileDir)
        sheet = workbook.sheet_by_index(0)
        labels = []
        for i_row in range(sheet.nrows):
            values = sheet.row_values(i_row)
            onset = values[0]
            durarion = values[1] - values[0]
            key = values[2]
            values[3] = str(int(values[3])) if isinstance(values[3], float) else values[3]
            degree1 = '1' if '/' not in values[3] else values[3].split('/')[1]
            degree2 = values[3] if '/' not in values[3] else values[3].split('/')[0]
            quality = values[4]
            inversion = int(values[5])
            rchord = values[6]
            labels.append((onset, durarion, key, degree1, degree2, quality, inversion, rchord))
        labels = np.array(labels, dtype=dt) # convert to structured array
        chord_labels[str(i)] = derive_chordSymbol_from_romanNumeral(labels, vocabulary) # translate rchords to tchords
    return chord_labels

def get_framewise_labels(pieces, chord_labels, resolution=4):
    """
    :param pieces:
    :param chord_labels:
    :param resolution: time resolution, default=4 (16th note as 1 unit of a pianoroll)
    :return: images, image_labels
    """
    print("Message: get framewise labels ...")
    dt = [('op', '<U10'), ('onset', 'float'), ('key', '<U10'), ('degree1', '<U10'), ('degree2', '<U10'), ('quality', '<U10'), ('inversion', 'int'), ('rchord', '<U10'), ('root', '<U10'), ('tquality', '<U10')] # label datatype
    for p in range(1,33):
        # Split Piano Roll into frames of the same size (88, wsize)
        pianoroll = pieces[str(p)]['pianoroll'] # [88, time]
        labels = chord_labels[str(p)]
        start_time = pieces[str(p)]['start_time']
        n_frames = pianoroll.shape[1]

        frame_labels = []
        for n in range(n_frames):
            frame_time = n*(1/resolution) + start_time
            try:
                label = labels[(labels['onset'] <= frame_time) & (labels['onset'] + labels['duration'] > frame_time)][0]
                frame_label = tuple([str(p), frame_time] + list(label)[2:])
                frame_labels.append(frame_label)
            except:
                print('Error: cannot get label !')
                print('piece =', p)
                print('frame time =', frame_time)
                exit(1)
        frame_labels = np.array(frame_labels, dtype=dt)
        chord_change = [1] + [0 if frame_labels[n]['root']+frame_labels[n]['tquality'] == frame_labels[n-1]['root']+frame_labels[n-1]['tquality'] else 1 for n in range(1, n_frames)] # chord change labels
        chord_change = np.array([(cc) for cc in chord_change], dtype=[('chord_change', 'int')])
        pieces[str(p)]['label'] = rfn.merge_arrays([frame_labels, chord_change], flatten=True, usemask=False)
    return pieces

def load_dataset(resolution, vocabulary):
    pieces = load_pieces(resolution=resolution) # {'no': {'pianoroll': 2d array, 'chromagram': 2d array, 'start_time': float}...}
    chord_labels = load_chord_labels(vocabulary=vocabulary) # {'no':  array}
    corpus = get_framewise_labels(pieces, chord_labels, resolution=resolution) # {'no': {'pianoroll': 2d array, 'chromagram': 2d array, 'start_time': float, 'label': array, 'chord_change': array},  ...}
    pianoroll_lens = [x['pianoroll'].shape[1] for x in corpus.values()]
    print('max_length =', max(pianoroll_lens))
    print('min_length =', min(pianoroll_lens))
    print('keys in corpus[\'op\'] =', corpus['1'].keys())
    print('label fields = ', corpus['1']['label'].dtype)
    return corpus

def augment_data(corpus):
    print('Running Message: augment data...')
    dt = [('op', '<U10'), ('onset', 'float'), ('key', '<U10'), ('degree1', '<U10'), ('degree2', '<U10'), ('quality', '<U10'), ('inversion', 'int'), ('rchord', '<U10'), ('root', '<U10'), ('tquality', '<U10'), ('chord_change', 'int')] # label datatype
    chroma_scale = ['C', 'C+', 'D', 'D+', 'E', 'F', 'F+', 'G', 'G+', 'A', 'A+', 'B']
    def shift_labels(label, shift):
        def shift_key(key, shift):
            accidental = '' if len(key) == 1 else key[1]
            key_without_accidental = key[0]
            key_id_shift = (chroma_scale.index(key_without_accidental.upper()) + shift) % 12
            if accidental != '':
                key_id_shift = (key_id_shift + 1)%12 if accidental == '+' else (key_id_shift - 1) % 12
            key_shift = chroma_scale[key_id_shift]
            if key_without_accidental.islower():
                key_shift = key_shift.lower()
            return key_shift
        def shift_root(root, shift):
            root_id_shift = (chroma_scale.index(root) + shift) % 12
            return chroma_scale[root_id_shift]
        return (label['op'], label['onset'], shift_key(label['key'], shift), label['degree1'], label['degree2'], label['quality'], label['inversion'], label['rchord'], shift_root(label['root'], shift), label['tquality'], label['chord_change'])

    corpus_aug = {}
    for shift in range(-3,7):
        shift_id = 'shift_' + str(shift)
        corpus_aug[shift_id] = {}
        for op in range(1,33):
            pianoroll_shift = np.roll(corpus[str(op)]['pianoroll'], shift=shift, axis=0)
            chromagram_shift = np.roll(corpus[str(op)]['chromagram'], shift=shift, axis=0)
            tonal_centroid = compute_Tonal_centroids(chromagram_shift)
            start_time = corpus[str(op)]['start_time']
            labels_shift = np.array([shift_labels(l, shift) for l in corpus[str(op)]['label']], dtype=dt)
            corpus_aug[shift_id][str(op)] = {'pianoroll': pianoroll_shift, 'tonal_centroid': tonal_centroid, 'start_time': start_time, 'label': labels_shift}
    print('keys in corpus_aug[\'shift_id\'][\'op\'] =', corpus_aug['shift_0']['1'].keys())
    return corpus_aug

def reshape_data(corpus_aug, n_steps=128, hop_size=16):
    '''n_steps: default = 128 frames (equals to 32 quater notes)
         hop_size: default = 16 frames (equals to 4 quater notes)'''
    print('Running Message: reshape data...')
    corpus_aug_reshape = copy.deepcopy(corpus_aug)
    dt = [('op', '<U10'), ('onset', 'float'), ('key', '<U10'), ('degree1', '<U10'), ('degree2', '<U10'), ('quality', '<U10'), ('inversion', 'int'), ('rchord', '<U10'), ('root', '<U10'), ('tquality', '<U10'), ('chord_change', 'int')]  # label datatype
    for shift_id, op_dict in corpus_aug.items():
        for op,  piece in op_dict.items():
            label_padding = np.array([(op, -1, 'pad', 'pad', 'pad', 'pad', -1, 'pad', 'pad', 'pad', 0)], dtype=dt)
            length = piece['pianoroll'].shape[1]
            n_pad = n_steps - (length % n_steps) if length % n_steps != 0 else 0
            n_sequences = (length + n_pad)//n_steps
            n_overlapped_sequences = (n_sequences - 1) * (n_steps//hop_size) + 1

            # padding
            pianoroll_pad = np.pad(piece['pianoroll'], [(0, 0), (0, n_pad)], 'constant').T # [time, 88]
            tonal_centroid_pad = np.pad(piece['tonal_centroid'], [(0, 0), (0, n_pad)], 'constant').T # [time, 6]
            label_pad = np.pad(piece['label'], (0, n_pad), 'constant', constant_values=label_padding) # [time]

            # segment into sequences without overlap
            corpus_aug_reshape[shift_id][op]['pianoroll'] = [np.reshape(pianoroll_pad, newshape=[-1, n_steps, 88])]
            corpus_aug_reshape[shift_id][op]['tonal_centroid'] = [np.reshape(tonal_centroid_pad, newshape=[-1, n_steps, 6])]
            corpus_aug_reshape[shift_id][op]['label'] = [np.reshape(label_pad, newshape=[-1, n_steps])]
            seq_lens = [n_steps for _ in range(n_sequences - 1)] + [(length % n_steps)] if n_pad != 0 else [n_steps for _ in range(n_sequences)]
            corpus_aug_reshape[shift_id][op]['len'] = [np.array(seq_lens, dtype=np.int32)]

            # segment into sequences with overlap
            corpus_aug_reshape[shift_id][op]['pianoroll'].append(np.stack([pianoroll_pad[i:i+n_steps] for i in range(0,length+n_pad-n_steps+1, hop_size)], axis=0))
            corpus_aug_reshape[shift_id][op]['tonal_centroid'].append(np.stack([tonal_centroid_pad[i:i+n_steps] for i in range(0,length+n_pad-n_steps+1, hop_size)], axis=0))
            corpus_aug_reshape[shift_id][op]['label'].append(np.stack([label_pad[i:i+n_steps] for i in range(0,length+n_pad-n_steps+1, hop_size)], axis=0))
            overlapped_seq_lens = [n_steps for _ in range(n_overlapped_sequences - 1)] + [(length % n_steps)] if n_pad != 0 else [n_steps for _ in range(n_overlapped_sequences)]
            corpus_aug_reshape[shift_id][op]['len'].append(np.array(overlapped_seq_lens, dtype=np.int32))

            '''corpus_aug_reshape[shift_id][op]['key'][0]: non-overlaped sequences
                        corpus_aug_reshape[shift_id][op]['key'][1]: overlapped sequences'''
    print('keys in corpus_aug_reshape[\'shift_id\'][\'op\'] =', corpus_aug_reshape['shift_0']['1'].keys())
    print('sequence_len_non_overlaped =', sorted(set([l for shift_dict in corpus_aug_reshape.values() for op_dict in shift_dict.values() for l in op_dict['len'][0]])))
    print('sequence_len_overlaped =', sorted(set([l for shift_dict in corpus_aug_reshape.values() for op_dict in shift_dict.values() for l in op_dict['len'][1]])))
    return corpus_aug_reshape

def compute_Tonal_centroids(chromagram, filtering=True, sigma=8):
    # define transformation matrix - phi
    Pi = math.pi
    r1, r2, r3 = 1, 1, 0.5
    phi_0 = r1 * np.sin(np.array(range(12)) * 7 * Pi / 6)
    phi_1 = r1 * np.cos(np.array(range(12)) * 7 * Pi / 6)
    phi_2 = r2 * np.sin(np.array(range(12)) * 3 * Pi / 2)
    phi_3 = r2 * np.cos(np.array(range(12)) * 3 * Pi / 2)
    phi_4 = r3 * np.sin(np.array(range(12)) * 2 * Pi / 3)
    phi_5 = r3 * np.cos(np.array(range(12)) * 2 * Pi / 3)
    phi_ = [phi_0, phi_1, phi_2, phi_3, phi_4, phi_5]
    phi = np.concatenate(phi_).reshape(6, 12) # [6, 12]
    phi_T = np.transpose(phi) # [12, 6]

    chromagram_T = np.transpose(chromagram) # [time, 12]
    TC_T = chromagram_T.dot(phi_T) # convert to tonal centiod representations, [time, 6]
    TC = np.transpose(TC_T) # [6, time]
    if filtering: # Gaussian filtering
        TC = gaussian_filter1d(TC, sigma=sigma, axis=1)
    return TC.astype(np.float32) # [6, time]

key_dict = {'C': 0, 'C-': 11, 'C+': 1,
            'D': 2, 'D-': 1, 'D+': 3,
            'E': 4, 'E-': 3, 'E+': 5,
            'F': 5, 'F-': 4, 'F+': 6,
            'G': 7, 'G-': 6, 'G+': 8,
            'A': 9, 'A-': 8, 'A+': 10,
            'B': 11, 'B-': 10, 'B+': 0,
            'c': 12, 'c-': 23, 'c+': 13,
            'd': 14, 'd-': 13, 'd+': 15,
            'e': 16, 'e-': 15, 'e+': 17,
            'f': 17, 'f-': 16, 'f+': 18,
            'g': 19, 'g-': 18, 'g+': 20,
            'a': 21, 'a-': 20, 'a+': 22,
            'b': 23, 'b-': 22, 'b+': 12,
            'PAD': 24}
quality_dict = {'M': 0, 'm': 1, 'a': 2, 'd': 3, 'M7': 4, 'm7': 5, 'D7': 6, 'd7': 7, 'h7': 8, 'a6': 9, 'PAD': 10}

def rlabel_indexing(labels):
    def analyze_label(label):
        def analyze_degree(degree):
            if '/' not in degree:
                pri_degree = 1
                sec_degree = translate_degree(degree)
            else:
                sec_degree = degree.split('/')[0]
                pri_degree = degree.split('/')[1]
                sec_degree = translate_degree(sec_degree)
                pri_degree = translate_degree(pri_degree)
            return pri_degree, sec_degree
        key = label['key']
        degree = label['degree']
        quality = label['quality']
        inversion = label['inversion']

        key_idx = key_dict[key]
        pri_degree, sec_degree = analyze_degree(degree)
        pri_degree_idx = pri_degree - 1
        sec_degree_idx = sec_degree - 1
        quality_idx = quality_dict[quality]
        inversion_idx = inversion
        return (key_idx, pri_degree_idx, sec_degree_idx, quality_idx, inversion_idx)
    dt = [('key', int), ('pri_degree', int), ('sec_degree', int), ('quality', int), ('inversion', int)]
    return np.array([analyze_label(label) for label in labels], dtype=dt)

def split_dataset(input_features, input_TC, input_labels, input_cc_labels, input_lengths, sequence_info):
    print('Running Message: split dataset into training, validation and testing sets ...')

    ## tChord_data_mirex_Mm_new_new
    s1 = [0, 4, 8, 12, 16, 20, 24, 28]
    s2 = [1, 5, 9, 13, 17, 21, 25, 29]
    s3 = [2, 6, 10, 14, 18, 22, 26, 30]
    s4 = [3, 7, 11, 15, 19, 23, 27, 31]
    train_indices = s1 + s2
    valid_indices = s3
    test_indices = s4

    feature_train = np.concatenate([input_features[m][p] for m in range(12) for p in train_indices], axis=0)
    feature_valid = np.concatenate([input_features[m][p] for m in range(12) for p in valid_indices], axis=0)
    feature_test = np.concatenate([input_features[0][p][::2] for p in test_indices], axis=0)

    TC_train = np.concatenate([input_TC[m][p] for m in range(12) for p in train_indices], axis=0)
    TC_valid = np.concatenate([input_TC[m][p] for m in range(12) for p in valid_indices], axis=0)
    TC_test = np.concatenate([input_TC[0][p][::2] for p in test_indices], axis=0)

    labels_train = np.concatenate([input_labels[m][p] for m in range(12) for p in train_indices], axis=0)
    labels_valid = np.concatenate([input_labels[m][p] for m in range(12) for p in valid_indices], axis=0)
    labels_test = np.concatenate([input_labels[0][p][::2] for p in test_indices], axis=0)

    cc_labels_train = np.concatenate([input_cc_labels[p] for m in range(12) for p in train_indices], axis=0)
    cc_labels_valid = np.concatenate([input_cc_labels[p] for m in range(12) for p in valid_indices], axis=0)
    cc_labels_test = np.concatenate([input_cc_labels[p][::2] for p in test_indices], axis=0)

    lens_train = list(itertools.chain.from_iterable([input_lengths[p] for m in range(12) for p in train_indices]))
    lens_valid = list(itertools.chain.from_iterable([input_lengths[p] for m in range(12) for p in valid_indices]))
    lens_test = list(itertools.chain.from_iterable([input_lengths[p][::2] for p in test_indices]))

    split_sets = {}
    split_sets['train'] = [sequence_info[p] for p in train_indices]
    split_sets['valid'] = [sequence_info[p] for p in valid_indices]
    split_sets['test'] = [(sequence_info[p][0], sequence_info[p][1]//2+1)  for p in test_indices]
    return feature_train, feature_valid, feature_test, \
           TC_train, TC_valid, TC_test, \
           labels_train, labels_valid, labels_test, \
           cc_labels_train, cc_labels_valid, cc_labels_test, \
           lens_train, lens_valid, lens_test, \
           split_sets

def derive_chordSymbol_from_romanNumeral(labels, vocabulary):
    # Create scales of all keys
    temp = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    keys = {}
    for i in range(11):
        majtonic = temp[(i * 4) % 7] + int(i / 7) * '+' + int(i % 7 > 5) * '+'
        mintonic = temp[(i * 4 - 2) % 7].lower() + int(i / 7) * '+' + int(i % 7 > 2) * '+'

        scale = list(temp)
        for j in range(i):
            scale[(j + 1) * 4 % 7 - 1] += '+'
        majscale = scale[(i * 4) % 7:] + scale[:(i * 4) % 7]
        minscale = scale[(i * 4 + 5) % 7:] + scale[:(i * 4 + 5) % 7]
        minscale[6] += '+'
        keys[majtonic] = majscale
        keys[mintonic] = minscale

    for i in range(1, 9):
        majtonic = temp[(i * 3) % 7] + int(i / 7) * '-' + int(i % 7 > 1) * '-'
        mintonic = temp[(i * 3 - 2) % 7].lower() + int(i / 7) * '-' + int(i % 7 > 4) * '-'
        scale = list(temp)
        for j in range(i):
            scale[(j + 2) * 3 % 7] += '-'
        majscale = scale[(i * 3) % 7:] + scale[:(i * 3) % 7]
        minscale = scale[(i * 3 + 5) % 7:] + scale[:(i * 3 + 5) % 7]
        if len(minscale[6]) == 1:
            minscale[6] += '+'
        else:
            minscale[6] = minscale[6][:-1]
        keys[majtonic] = majscale
        keys[mintonic] = minscale

    # Translate chords
    tchords = []
    for rchord in labels:
        # print(str(rchord['key'])+': '+str(rchord['degree'])+', '+str(rchord['quality']))
        key = str(rchord['key'])
        degree1 = rchord['degree1']
        degree2 = rchord['degree2']

        if degree1 == '1':  # case: not secondary chord
            if len(degree2) == 1: # case: degree = x
                degree = int(degree2)
                root = keys[key][degree-1]
            else: # case: degree = -x or +x
                if str(rchord['quality']) != 'a6': # case: chromatic chord, -x
                    degree = int(degree2[1])
                    root = keys[key][degree-1]
                    if '+' not in root:
                        root += degree2[0]
                    else:
                        root = root[:-1]
                else:  # case: augmented 6th
                    degree = 6
                    root = keys[key][degree - 1]
                    if str(rchord['key'])[0].isupper():  # case: major key
                        if '+' not in root:
                            root += '-'
                        else:
                            root = root[:-1]

        elif degree1 != '1': # case: secondary chord
            d2 = int(degree2) if degree2 != '+4' else 6
            d1 = int(degree1)
            if d1 > 0:
                key2 = keys[key][d1 - 1] # secondary key
            else:
                key2 = keys[key][abs(d1) - 1]  # secondary key
                if '+' not in key2:
                    key2 += '-'
                else:
                    key2 = key2[:-1]

            root = keys[key2][d2 - 1]
            if degree2 == '+4' :
                if key2.isupper(): # case: major key
                    if '+' not in root:
                        root += '-'
                    else:
                        root = root[:-1]

        # Re-translate root for enharmonic equivalence
        if '++' in root:  # if root = x++
            root = temp[(temp.index(root[0]) + 1) % 7]
        elif '--' in root:  # if root = x--
            root = temp[(temp.index(root[0]) - 1) % 7]

        if '-' in root:  # case: root = x-
            if ('F' not in root) and ('C' not in root):  # case: root = x-, and x != F and C
                root = temp[((temp.index(root[0])) - 1) % 7] + '+'
            else:
                root = temp[((temp.index(root[0])) - 1) % 7]  # case: root = x-, and x == F or C
        elif ('+' in root) and ('E' in root or 'B' in root):  # case: root = x+, and x == E or B
            root = temp[((temp.index(root[0])) + 1) % 7]

        tquality = rchord['quality'] if rchord['quality'] != 'a6' else 'D7' # outputQ[rchord['quality']]

        # tquality mapping
        if vocabulary == 'MIREX_Mm':
            tquality_map_dict = {'M': 'M', 'm': 'm', 'a': 'O', 'd': 'O', 'M7': 'M', 'D7': 'M', 'm7': 'm', 'h7': 'O', 'd7': 'O'} # 'O' stands for 'others'
        elif vocabulary == 'MIREX_7th':
            tquality_map_dict = {'M': 'M', 'm': 'm', 'a': 'O', 'd': 'O', 'M7': 'M7', 'D7': 'D7', 'm7': 'm7', 'h7': 'O', 'd7': 'O'}
        elif vocabulary == 'triad':
            tquality_map_dict = {'M': 'M', 'm': 'm', 'a': 'a', 'd': 'd', 'M7': 'M', 'D7': 'M', 'm7': 'm', 'h7': 'd', 'd7': 'd'}
        elif vocabulary == 'seventh':
            tquality_map_dict = {'M': 'M', 'm': 'm', 'a': 'a', 'd': 'd', 'M7': 'M7', 'D7': 'D7', 'm7': 'm7', 'h7': 'h7', 'd7': 'd7'}
        tquality = tquality_map_dict[tquality]

        tchord = (root, tquality)
        tchords.append(tchord)

    tchords = np.array(tchords, dtype=[('root', '<U10'), ('tquality', '<U10')])
    rtchords = rfn.merge_arrays((labels, tchords), flatten=True, usemask=False) # merge rchords and tchords into one structured array
    return rtchords

def translate_degree(degree_str):
    if ('+' not in degree_str and '-' not in degree_str) or ('+' in degree_str and degree_str[1] == '+'):
        degree_hot = int(degree_str[0])
    elif degree_str[0] == '-':
        degree_hot = int(degree_str[1]) + 14
    elif degree_str[0] == '+':
        degree_hot = int(degree_str[1]) + 7
    return degree_hot

def save_preprocessed_data(data, save_dir):
    with open(save_dir, 'wb') as save_file:
        pickle.dump(data, save_file, protocol=pickle.HIGHEST_PROTOCOL)
    print('Preprocessed data saved.')

def main():
    vocabulary = 'MIREX_Mm'
    corpus = load_dataset(resolution=4, vocabulary=vocabulary) # {'no': {'pianoroll': 2d array, 'chromagram': 2d array, 'start_time': float, 'label': array},  ...}
    corpus_aug = augment_data(corpus) # {'shift_id': {'no': {'pianoroll': 2d array, 'chromagram': 2d array, tonal_centroid': 2d array, 'start_time': float, 'label': 1d array}, ...},  ...}
    corpus_aug_reshape = reshape_data(corpus_aug, n_steps=128, hop_size=16) # {'shift_id': {'no': {'pianoroll': 3d array, 'chromagram': 3d array, 'tonal_centroid': 3d array, 'start_time': float, 'label': 2d array, 'len': 2d array}, ...},  ...}

    # Save processed data
    dir = 'BPS_FH_preprocessed_data_' + vocabulary + '.pickle'
    save_preprocessed_data(corpus_aug_reshape, save_dir=dir)

if __name__ == '__main__':
    main()