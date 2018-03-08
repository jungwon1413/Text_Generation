import numpy as np
import tensorflow as tf
from konlpy.tag import Twitter
from tensorflow.contrib import rnn

# 해당 파트 구동 확인 완료되었습니다.
class Novel_Dataset:
    def __init__(self, full_filename, seq_len):
        try:
            read_file = open(full_filename, 'rb').read()
            encode_type = chardet.detect(read_file)['encoding']
            # text = open(full_filename, encoding=None).read().lower()
            print(encode_type)
            text = open(full_filename, encoding=encode_type).read()
        except:
            return

        print("Word Slicing...")
        char_set = sorted(list(set(text)))    # character split

        self.char_indices = dict((c, i) for i, c in enumerate(char_set))
        self.indices_char = dict((i, c) for i, c in enumerate(char_set))

        idx_text = []
        for i, char in enumerate(text):
            idx_text.append(self.char_indices[char])
        text = idx_text

        print("Generating Number Index...")
        self.x_data = []
        self.y_data = []
        for i in range(0, len(text) - seq_len):
            fill_x = seq_len - len(text[i : i+seq_len])
            fill_y = seq_len - len(text[i+1 : i+seq_len - 1])
            
            x_text = text[i : i+seq_len]
            y_text = text[i+1 : i+seq_len - 1]
            
            if fill_x is not 0:
                x_text.extend([0 for i in range(fill_x)])
            elif fill_y is not 0:
                y_text.extend([0 for i in range(fill_y)])

            # Debug purpose
            if i % 400000 is 0:
                print('Line', i, ':', x_text, '=>', y_text)

            self.x_data.append(x_text)
            self.y_data.append(y_text)
        
if __name__ == "__main__":
    filename = "D:/Paper_txt/PAPER_01001.txt"
    seq_len = 26
    novel = Novel_Dataset(filename, seq_len)

    data_dim = len(novel.characters)
    hidden_size = len(novel.characters)
    num_classes = len(novel.characters)
    learning_rate = 0.001

    print("Character set length: %s" % data_dim)    # length check
    print("Dataset X length: %s" % len(novel.x_data))    # dataset length check
    print("Dataset Y length: %s" % len(novel.y_data))

    batch_size = len(novel.x_data)

    X = tf.placeholder(tf.int32, [None, seq_len])
    Y = tf.placeholder(tf.int32, [None, seq_len])

    # One-hot encoding
    X_one_hot = tf.one_hot(X, num_classes)
    print(X_one_hot)

    # Make a lstm cell with hidden_size (each unit output vector size)
    def lstm_cell():
        cell = rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
        return cell

    multi_cells = rnn.MultiRNNCell([lstm_cell() for _ in range(2)], state_is_tuple=True)

    # outputs: unfolding size x hidden size, state = hidden size
    outputs, _states =  tf.nn.dynamic_rnn(multi_cells, X_one_hot, dtype=tf.float32)

    # FC layer
    X_for_fc = tf.reshape(outputs, [-1, hidden_size])
    outputs = tf.contrib.layers.fully_connected(X_for_fc, num_classes, activation_fn=None)

    # reshape out for sequence_loss
    outputs = tf.reshape(outputs, [batch_size, seq_len, num_classes])

    # All weights are 1 (equal weights)
    weights = tf.ones([batch_size, seq_len])

    sequence_loss = tf.contrib.seq2seq.sequence_loss(
        logits=outputs, targets=Y, weights=weights)
    mean_loss = tf.reduce_mean(sequence_loss)
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mean_loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(500):
        _, l, results = sess.run(
            [train_op, mean_loss, outputs], feed_dict={X: novel.x_data, Y: novel.y_data})
        for j, result in enumerate(results):
            index = np.argmax(result, axis=1)
        if i % 10 is 0:
            print(i, j, ''.join([novel.indices_char[t] for t in index]), l)