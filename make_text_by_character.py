import os
import time
import chardet
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class TextGen:

    def __init__(self, filename, learning_rate, num_layers, seq_len,
         epoch, save_point, save_at, encode=False, stride=1):
        self.filename = filename
        self.learning_rate = learning_rate
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.epoch = epoch
        self.save_point = save_point
        self.save_at = save_at
        self.encode = encode
        self.stride = stride

    def read_dataset(self):
        try:
            read_file = open(self.filename, 'rb').read()
            if self.encode is True:
                encode_type = chardet.detect(read_file)['encoding']
                print(encode_type)
            else:
                encode_type = None
            # text = open(full_filename, encoding=None).read().lower()
            text = open(self.filename, encoding=encode_type).read()
        except:
            text = False
        print("Load Complete.")
        return text

    def Slice_Data(self):
        char_set = sorted(list(set(self.text)))    # character split
        num_classes = len(char_set)
        char_indices = dict((c, i) for i, c in enumerate(char_set))
        indices_char = dict((i, c) for i, c in enumerate(char_set))
        print("Slice Complete.")
        return char_set, num_classes, char_indices, indices_char

    def Data2idx(self):
        idx_text = []
        for char in self.text:
            idx_text.append(self.char_indices[char])
        text = idx_text
        print("Indexing Complete.")
        return text

    def Build_Data(self):
        x_data = []
        y_data = []
        for i in range(0, len(self.text) - self.seq_len, self.stride):
            fill_x = self.seq_len - len(self.text[i : i+self.seq_len])
            fill_y = self.seq_len - len(self.text[i+1 : i+self.seq_len - 1])
            
            x_text = self.text[i : i+self.seq_len]
            y_text = self.text[i+1 : i+self.seq_len - 1]
            
            if fill_x is not 0:
                x_text.extend([0 for i in range(fill_x)])
            elif fill_y is not 0:
                y_text.extend([0 for i in range(fill_y)])

            x_data.append(x_text)
            y_data.append(y_text)
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        print("Sequencing Complete.")
        return x_data, y_data

    def Make_Text(self, sess):
        self.results = sess.run(self.outputs, feed_dict={self.X: self.x_data})
        for j, result in enumerate(self.results):
            index = np.argmax(result, axis=1)
            if j is 0:  # print all for the first result to make a sentence
                print(''.join([self.indices_char[t] for t in index]), end='')
            else:
                print(self.indices_char[index[-1]], end='')

    def LSTM(self):
        lstm = tf.contrib.rnn.BasicLSTMCell(self.hidden_size,
                                            state_is_tuple = True)
        return lstm

    def BRNN_Dynamic(self, X):
        # Default activation: Tanh
        # Default: state_is_Tuple=True
        # lstm = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, state_is_tuple = True)
        lstm = tf.nn.rnn_cell.DropoutWrapper(self.LSTM(), output_keep_prob=0.5)    # Dropout 0.5
        # multi_lstm = tf.contrib.rnn.MultiRNNCell([lstm]*2)
        initial_state = lstm.zero_state(self.batch_size, tf.float32)
        # Bidirectional Dynamic RNN
        (fw_output, bw_output), _states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm,
                                                                        cell_bw=lstm,
                                                                        inputs=X,
                                                                        initial_state_fw=initial_state,
                                                                        initial_state_bw=initial_state,
                                                                        dtype=tf.float32)
        outputs = tf.concat([fw_output, bw_output], axis=2)
        return outputs

    def MultiRNN_Dynamic(self, X):
        # lstm = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)
        lstm = tf.nn.rnn_cell.DropoutWrapper(self.LSTM(), output_keep_prob=0.5)
        multi_lstm = tf.contrib.rnn.MultiRNNCell([lstm] * self.num_layers,
                                                 state_is_tuple = True)
        outputs, _states = tf.nn.dynamic_rnn(multi_lstm,
                                             X,
                                             dtype=tf.float32)
        return outputs
  
    def Elapsed(self):
        total = self.end - self.start
        m, s = divmod(total, 60)
        h, m = divmod(m, 60)
        
        if m == 0 and h == 0:
            print("Time Elapsed: {:>3d} sec".format(int(s)))
        elif m != 0 and h == 0:
            print("Time Elapsed: {:>3d} min {:>3d} sec".format(int(m), int(s)))
        else:
            print("Time Elapsed: {:>3d} hour {:>3d} min {:>3d} sec".format(int(h), int(m), int(s)))
            
    def Save_Model(self, saver, sess, iter):
        pathfile, ext = os.path.splitext(self.save_at)
        file = pathfile + str(iter) + ext
        save_model = saver.save(sess, file)
        print("Model saved in path: %s" % save_model)
        
    def Load_Model(saver, sess, filepath):
        saver.restore(sess, filepath)
        print("Model restored.")
        
    def Plot_Loss(self):
        fig = plt.figure(figsize=(14, 12))
        graph_1 = fig.add_subplot(2, 1, 1)
        graph_1.grid()
        graph_2 = fig.add_subplot(2, 1, 2)
        graph_2.grid()
        graph_1.plot(self.iter_loss,
                     label='Loss v. Iter')
        graph_2.plot(self.elapsed,
                     self.iter_loss,
                     label='Loss v. Time')
        
        graph_1.set_xlabel('Iter')
        graph_1.set_ylabel('Loss')
        graph_1.set_title('Loss vs Epoch')
        graph_2.set_xlabel('Time(sec)')
        graph_2.set_ylabel('Loss')
        graph_2.set_title('Loss vs Time')
        plt.show()

    def Prepare_Model(self):
        
        # load data
        self.text = self.read_dataset()
        self.char_set, self.num_classes, self.char_indices, self.indices_char = self.Slice_Data()
        self.text = self.Data2idx()
        self.x_data, self.y_data = self.Build_Data()

        print("Text length: %s" % len(self.text))
        print("Number of characters: {}".format(self.num_classes))    # length check
        print("Dataset X has {} sequences.".format(len(self.x_data)))    # dataset shape check
        print("Dataset Y has {} sequences.".format(len(self.y_data)))
        print("Dataset X has {} shape.".format(self.x_data.shape))

        # batch_size: Mini batch size
        # data_dims: ??? ??? feature? ??? (???, ???, ??)
        # output_dim: ??? ??? feature? ?????
        # seq_len: ??? ? sequence(?? ??)? ???
        self.batch_size = len(self.x_data)
        self.data_dims = len(self.x_data)
        self.output_dim = 1
        self.hidden_size = self.num_classes

        self.X = tf.placeholder(tf.int32, [None, self.seq_len])
        self.Y = tf.placeholder(tf.int32, [None, self.seq_len])

        # One-hot encoding
        X_one_hot = tf.one_hot(self.X, self.num_classes)
        print(X_one_hot)

        # LSTM Cell
        outputs = self.MultiRNN_Dynamic(X_one_hot)
        # weights = tf.truncated_normal([self.hidden_size, self.num_classes], stddev=0.01)
        # bias = tf.constant(0.1, shape=[self.num_classes])
        # weights, bias = tf.Variable((weights, bias))
        
        # Add Softmax Layer
        X_for_softmax = tf.reshape(outputs,
                                   [-1, self.hidden_size])    # FC layer, x2 for BRNN
        self.outputs = tf.contrib.layers.fully_connected(outputs,
                                                        self.num_classes,
                                                        activation_fn=None)
        # self.predict = tf.nn.softmax(tf.matmul(X_for_softmax, weights) + bias)
        # self.outputs = tf.reshape(self.predict,    # Do not apply activation!?
        #                           [-1,
        #                            self.seq_len,
        #                            self.num_classes])

        # Initialize fc weights with Ones
        # If all weights are not same, loss will explode!!
        weights = tf.ones([self.batch_size, self.seq_len])

        # Monitor loss
        loss = tf.contrib.seq2seq.sequence_loss(logits=self.outputs,
                                                targets=self.Y,
                                                weights=weights)
        # self.mean_loss = -tf.reduce_sum(self.Y * tf.log(tf.clip_by_value(self.predict,1e-10,1.0)))
        self.mean_loss = tf.reduce_mean(loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = optimizer.minimize(self.mean_loss)

    def Train(self, sess):
        saver = tf.train.Saver()
        
        sess.run(tf.global_variables_initializer())
        print('=' * 20, "{:^20}".format("Training Start"), '=' * 20)

        self.iter_loss = []
        
        self.start = time.time()
        for i in range(self.epoch):
            _, l, results = sess.run(
                [self.train_op, self.mean_loss, self.outputs],
                feed_dict={self.X: self.x_data, self.Y: self.y_data})
            for j, result in enumerate(results):
                index = np.argmax(result, axis=1)
                
                if i % 100 == 0 and j == 0:
                    print("\n At step", i, ':',
                        ''.join([self.indices_char[t] for t in index]))
                    print('Loss:', l)
                    self.end = time.time()
                    self.Elapsed()
            if i % 1000 == 0:
                self.Save_Model(saver, sess, i)
            self.iter_loss.append(l)    # Iteration & Loss

if __name__ == "__main__":
    filename = "SAMPLE.py"
    learning_rate = 0.1
    num_layers = 1
    seq_len = 30
    epoch = 5001
    save_point = 500
    save_at = "C:/Users/jungw/OneDrive/??/GitHub/Text_Generation/Models/rnn_text.ckpt"
    
    code_gen = TextGen(filename=filename,
                    learning_rate=learning_rate,
                    num_layers=num_layers,
                    seq_len=seq_len,
                    epoch=epoch,
                    save_point=save_point,
                    save_at=save_at)
    
    code_gen.Prepare_Model()
    sess = tf.Session()
    code_gen.Train(sess)
    
    # Generate Text
    code_gen.Make_Text(sess)

    # Monitor Loss, and Time info
    code_gen.Plot_Loss()
    code_gen.Plot_Time_Iter()