import time
import chardet
import numpy as np
import tensorflow as tf

from tensorflow.contrib import rnn
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.seq2seq import sequence_loss


# 해당 파트 구동 확인 완료되었습니다.
def init_data(full_filename, encode=False):
    try:
        read_file = open(full_filename, 'rb').read()
        if encode is True:
            encode_type = chardet.detect(read_file)['encoding']
            print(encode_type)
        else:
            encode_type = None
        # text = open(full_filename, encoding=None).read().lower()
        text = open(full_filename, encoding=encode_type).read()
        return text
    except:
        return False

def slice_data(text):
    print("Word Slicing...")
    char_set = sorted(list(set(text)))    # character split

    char_indices = dict((c, i) for i, c in enumerate(char_set))
    indices_char = dict((i, c) for i, c in enumerate(char_set))
    return char_indices, indices_char

def data_to_idx(text, char_indices):
    idx_text = []
    for i, char in enumerate(text):
        idx_text.append(char_indices[char])
    text = idx_text
    return text

def build_data(text, seq_len, stride = 4):
    print("Generating Number Index...")
    x_data = []
    y_data = []
    for i in range(0, len(text) - seq_len, stride):
        fill_x = seq_len - len(text[i : i+seq_len])
        fill_y = seq_len - len(text[i+1 : i+seq_len - 1])
        
        x_text = text[i : i+seq_len]
        y_text = text[i+1 : i+seq_len - 1]
        
        if fill_x is not 0:
            x_text.extend([0 for i in range(fill_x)])
        elif fill_y is not 0:
            y_text.extend([0 for i in range(fill_y)])

        x_data.append(x_text)
        y_data.append(y_text)
    return x_data, y_data


def lstm_cell(hidden_size):
    # Make a lstm cell with hidden_size (each unit output vector size)
    cell = rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
    return cell

def predict(result, x_data, outputs, char_indices, indices_char):
    output = []
    results = sess.run(outputs, feed_dict={X: x_data})
    for j, result in enumerate(results):
        index = np.argmax(result, axis=1)
        if j is 0:  # print all for the first result to make a sentence
            print(''.join([indices_char[t] for t in index]), end='')
        else:
            print(indices_char[index[-1]], end='')

def elapsed(start, end):
    total = end - start
    m, s = divmod(total, 60)
    h, m = divmod(m, 60)
    
    if m == 0 and h == 0:
        print("Time Elapsed: {:>3d} sec".format(int(s)))
    elif m != 0 and h == 0:
        print("Time Elapsed: {:>3d} min {:>3d} sec".format(int(m), int(s)))
    else:
        print("Time Elapsed: {:>3d} hour {:>3d} min {:>3d} sec".format(int(h), int(m), int(s)))

if __name__ == "__main__":
    filename = "SAMPLE.py"
    seq_len = 40

    # load data
    text = init_data(filename)
    char_indices, indices_char = slice_data(text)
    text = data_to_idx(text, char_indices)
    x_data, y_data = build_data(text, seq_len)

    data_dim = len(char_indices)
    hidden_size = len(char_indices)
    num_classes = len(char_indices)
    learning_rate = 0.1

    print("Text length: %s" % len(text))
    print("Character set length: %s" % data_dim)    # length check
    print("Dataset X length: %s" % len(x_data))    # dataset shape check
    print("Dataset Y length: %s" % len(y_data))

    batch_size = len(x_data)

    X = tf.placeholder(tf.int32, [None, seq_len])
    Y = tf.placeholder(tf.int32, [None, seq_len])

    # One-hot encoding
    X_one_hot = tf.one_hot(X, num_classes)

    # Make a lstm cell with hidden_size (each unit output vector size)
    lstm = lstm_cell(hidden_size)
    multi_cells = rnn.MultiRNNCell([lstm for _ in range(2)], state_is_tuple=True)

    # outputs: unfolding size x hidden size, state = hidden size
    outputs, _states = tf.nn.dynamic_rnn(multi_cells, X_one_hot, dtype=tf.float32)

    # FC layer
    X_for_fc = tf.reshape(outputs, [-1, hidden_size])
    outputs = fully_connected(X_for_fc, num_classes, activation_fn=None)

    # reshape out for sequence_loss
    outputs = tf.reshape(outputs, [batch_size, seq_len, num_classes])

    # All weights are 1 (equal weights)
    weights = tf.ones([batch_size, seq_len])

    loss = sequence_loss(logits=outputs, targets=Y, weights=weights)
    mean_loss = tf.reduce_mean(loss)
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mean_loss)

    sess = tf.Session()
    
    
    
    sess.run(tf.global_variables_initializer())
    print('=' * 20, "{:^20}".format("Training Start"), '=' * 20)

    iter_loss = []
    elapsed_loss = []
    
    start = time.process_time()
    for i in range(10000):
        _, l, results = sess.run(
            [train_op, mean_loss, outputs], feed_dict={X: x_data, Y: y_data})
        for j, result in enumerate(results):
            index = np.argmax(result, axis=1)
            
            if i % 100 == 0 and j == 0:
                print("\n Step", i, "Iter", j, ':',
                      ''.join([indices_char[t] for t in index]))
                print('Loss:', l)
                end = time.process_time()
                elapsed(start, end)
        iter_loss.append([i, l])
        elapsed_loss.append([start-end, l])

    print('=' * 20, "{:^20}".format("Training Complete"), '=' * 20)

    output = predict(result, x_data, outputs, char_indices, indices_char)