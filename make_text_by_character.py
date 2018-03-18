import time
import chardet
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.seq2seq import sequence_loss
import matplotlib.pyplot as plt

# ??? ??? ??
plt.rcParams["figure.figsize"] = (14, 12)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.color'] = 'r'
plt.rcParams['axes.grid'] = True 


class TextGen:
    def __init__(self, full_filename, encode=False):
        try:
            read_file = open(full_filename, 'rb').read()
            if encode is True:
                encode_type = chardet.detect(read_file)['encoding']
                print(encode_type)
            else:
                encode_type = None
            # text = open(full_filename, encoding=None).read().lower()
            text = open(full_filename, encoding=encode_type).read()
            self.text = text
        except:
            self.text = False

    def Slice_Data(self):
        print("Word Slicing...")
        self.char_set = sorted(list(set(self.text)))    # character split
        self.char_indices = dict((c, i) for i, c in enumerate(self.char_set))
        self.indices_char = dict((i, c) for i, c in enumerate(self.char_set))

    def Data2idx(self):
        idx_text = []
        for char in self.text:
            idx_text.append(self.char_indices[char])
        self.text = idx_text

    def Build_Data(self, seq_len, stride = 1):
        print("Generating Number Index...")
        x_data = []
        y_data = []
        for i in range(0, len(self.text) - self.seq_len, stride):
            fill_x = seq_len - len(self.text[i : i+seq_len])
            fill_y = seq_len - len(self.text[i+1 : i+seq_len - 1])
            
            x_text = self.text[i : i+seq_len]
            y_text = self.text[i+1 : i+seq_len - 1]
            
            if fill_x is not 0:
                x_text.extend([0 for i in range(fill_x)])
            elif fill_y is not 0:
                y_text.extend([0 for i in range(fill_y)])

            x_data.append(x_text)
            y_data.append(y_text)
        self.x_data = x_data
        self.y_data = y_data

    def LSTM_Cell(hidden_size):
        # Make a lstm cell with hidden_size (each unit output vector size)
        cell = rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
        return cell

    def Make_Text(self, sess):
        self.results = sess.run(self.outputs, feed_dict={self.X: self.x_data})
        for j, result in enumerate(self.results):
            index = np.argmax(result, axis=1)
            if j is 0:  # print all for the first result to make a sentence
                print(''.join([self.indices_char[t] for t in index]), end='')
            else:
                print(self.indices_char[index[-1]], end='')

    def Elapsed(start, end):
        total = end - start
        m, s = divmod(total, 60)
        h, m = divmod(m, 60)
        
        if m == 0 and h == 0:
            print("Time Elapsed: {:>3d} sec".format(int(s)))
        elif m != 0 and h == 0:
            print("Time Elapsed: {:>3d} min {:>3d} sec".format(int(m), int(s)))
        else:
            print("Time Elapsed: {:>3d} hour {:>3d} min {:>3d} sec".format(int(h), int(m), int(s)))
            
    def Save_Model(saver, sess, filepath, step=1000):
        save_model = saver.save(sess, filepath, global_step=step)
        print("Model saved in path: %s" % save_model)
        
    def Load_Model(saver, sess, filepath):
        saver.restore(sess, filepath)
        print("Model restored.")
        
    def Plot_Iter_Loss(self):
        plt.figure()
        plt.plot(self.iter_loss)
        plt.xlabel('Iter')
        plt.ylabel('Loss')
        plt.title('Loss vs Epoch')
        plt.show()
        
    def Plot_Time_Loss(self):
        plt.figure()
        plt.plot(self.elapsed, self.iter_loss)
        plt.xlabel('Time(sec)')
        plt.ylabel('Loss')
        plt.title('Loss vs Time')
        plt.show()
        
    def Plot_Time_Iter(self):
        plt.figure()
        plt.plot(self.elapsed)
        plt.xlabel('Iter')
        plt.ylabel('Time')
        plt.title('Iter vs Time')
        plt.show()

    def Prepare_Model(self, seq_len, learning_rate=0.1):
        self.seq_len = seq_len

        # load data
        TextGen.Slice_Data(self)
        TextGen.Data2idx(self)
        TextGen.Build_Data(self, seq_len)

        self.data_dim = len(self.char_set)
        self.hidden_size = len(self.char_set)
        self.num_classes = len(self.char_set)
        self.learning_rate = learning_rate

        print("Text length: %s" % len(self.text))
        print("Character set length: %s" % self.data_dim)    # length check
        print("Dataset X length: %s" % len(self.x_data))    # dataset shape check
        print("Dataset Y length: %s" % len(self.y_data))

        self.batch_size = len(self.x_data)

        self.X = tf.placeholder(tf.int32, [None, self.seq_len])
        self.Y = tf.placeholder(tf.int32, [None, self.seq_len])

        # One-hot encoding
        X_one_hot = tf.one_hot(self.X, self.num_classes)

        # Make a lstm cell with hidden_size (each unit output vector size)
        lstm = TextGen.LSTM_Cell(self.hidden_size)
        multi_cells = rnn.MultiRNNCell([lstm for _ in range(2)], state_is_tuple=True)

        # outputs: unfolding size x hidden size, state = hidden size
        outputs, _states = tf.nn.dynamic_rnn(multi_cells, X_one_hot, dtype=tf.float32)

        # FC layer
        X_for_fc = tf.reshape(outputs, [-1, self.hidden_size])
        outputs = fully_connected(X_for_fc, self.num_classes, activation_fn=None)

        # reshape out for sequence_loss
        self.outputs = tf.reshape(outputs, [self.batch_size, self.seq_len, self.num_classes])

        # All weights are 1 (equal weights)
        weights = tf.ones([self.batch_size, self.seq_len])

        loss = sequence_loss(logits=self.outputs, targets=self.Y, weights=weights)
        self.mean_loss = tf.reduce_mean(loss)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.mean_loss)

    def Train(self, sess, epoch, save_at):
        saver = tf.train.Saver()
        
        sess.run(tf.global_variables_initializer())
        print('=' * 20, "{:^20}".format("Training Start"), '=' * 20)

        self.iter_loss = []
        self.elapsed = []
        self.savepath = save_at
        
        start = time.time()
        for i in range(epoch):
            _, l, results = sess.run(
                [self.train_op, self.mean_loss, self.outputs],
                feed_dict={self.X: self.x_data, self.Y: self.y_data})
            for j, result in enumerate(results):
                index = np.argmax(result, axis=1)
                
                if i % 100 == 0 and j == 0:
                    print("\n At step", i, ':',
                        ''.join([self.indices_char[t] for t in index]))
                    print('Loss:', l)
                    end = time.time()
                    TextGen.Elapsed(start, end)
            if i % 1000 == 0:
                TextGen.Save_Model(saver, sess, self.savepath)            
            self.iter_loss.append(l)    # Iteration & Loss
            self.elapsed.append(end-start)    # Elapsed Time & Loss


        print('=' * 20, "{:^20}".format("Training Complete"), '=' * 20)

if __name__ == "__main__":
    code_gen = TextGen("SAMPLE.py")
    learning_rate = 0.1
    code_gen.Prepare_Model(40, learning_rate)

    sess = tf.Session()
    save_at = "C:/Users/jungw/OneDrive/??/GitHub/Text_Generation/rnn_text.ckpt"
    epoch = 10001
    
    code_gen.Train(sess, epoch, save_at)
    
    code_gen.Make_Text(sess)