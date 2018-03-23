import os
import csv
import time
import chardet
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from konlpy.tag import Twitter
from tensorflow.contrib.tensorboard.plugins import projector

class TextGen:

    def __init__(self, filename, learning_rate, num_layers, seq_len, epoch,
         save_point, save_at, encode=False, chunk_word=False, stride=1):
        self.filename = filename
        self.learning_rate = learning_rate
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.epoch = epoch
        self.save_point = save_point
        self.save_at = save_at
        self.encode = encode
        self.stride = stride
        self.chunk_word = chunk_word

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
        if self.chunk_word is True:
            text.replace(' ', " SPACE ")
            text.replace('\n', " ENTER ")
        print("Load Complete.")
        return text

    def Slice_Data(self):
        if self.chunk_word is True:
            twitter = Twitter()
            self.text = twitter.morphs(self.text)
            self.text = [' ' if word is "SPACE" else word for word in self.text]
            self.text = ['\n' if word is "ENTER" else word for word in self.text]
        
        vocabulary = sorted(list(set(self.text)))    # character split
        vocabulary_size = len(vocabulary)
        vocab_ids = dict((c, i) for i, c in enumerate(vocabulary))
        ids_vocab = dict((i, c) for i, c in enumerate(vocabulary))
        print("Slice Complete.")
        return vocabulary, vocabulary_size, vocab_ids, ids_vocab

    def Data2idx(self):
        idx_text = []
        for element in self.text:
            idx_text.append(self.vocab_ids[element])
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

    def Make_CSV(self, filename):
        with open(filename, 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            for element in self.vocabulary:
                writer.writerow([element])

    def Make_Text(self, sess):
        self.results = sess.run(self.outputs, feed_dict={self.X: self.x_data})
        for j, result in enumerate(self.results):
            index = np.argmax(result, axis=1)
            if j is 0:  # print all for the first result to make a sentence
                print(''.join([self.ids_vocab[t] for t in index]), end='')
            else:
                print(self.ids_vocab[index[-1]], end='')

    def LSTM(self):
        lstm = tf.contrib.rnn.BasicLSTMCell(self.hidden_size,
                                            state_is_tuple = True)
        return lstm

    def BRNN_Dynamic(self, X):
        # Default activation: Tanh
        # Default: state_is_Tuple=True
        # lstm = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, state_is_tuple = True)
        lstm = tf.nn.rnn_cell.DropoutWrapper(self.LSTM(), output_keep_prob=0.8)    # Dropout 0.8
        multi_lstm = tf.contrib.rnn.MultiRNNCell([lstm]*self.num_layers,
                                                state_is_tuple = True)
        # Bidirectional Dynamic RNN
        (fw_output, bw_output), _states = tf.nn.bidirectional_dynamic_rnn(cell_fw=multi_lstm,
                                                                        cell_bw=multi_lstm,
                                                                        inputs=X,
                                                                        dtype=tf.float32)
        outputs = tf.concat([fw_output, bw_output], axis=2)
        return outputs

    def MultiRNN_Dynamic(self, X):
        lstm = self.LSTM()
        # lstm = tf.nn.rnn_cell.DropoutWrapper(self.LSTM(), output_keep_prob=0.9)
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
            
    def Save_Model(self, saver, sess, iter, output=True):
        pathfile, ext = os.path.splitext(self.save_at)
        file = pathfile + '_' + str(iter) + ext
        save_model = saver.save(sess, file)
        if output is True:
            print("Model saved in path: %s" % save_model)
        
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
        
    def Plot_Time_Iter(self):
        plt.figure()
        plt.plot(self.elapsed)
        plt.xlabel('Iter')
        plt.ylabel('Time')
        plt.title('Iter vs Time')
        plt.show()

    def Prepare_Model(self):
        
        # load data
        self.text = self.read_dataset()
        self.vocabulary, self.vocabulary_size, self.vocab_ids, self.ids_vocab = self.Slice_Data()
        self.text = self.Data2idx()
        self.x_data, self.y_data = self.Build_Data()

        print("Text length: %s" % len(self.text))
        print("Number of characters: {}".format(self.vocabulary_size))    # length check
        print("Dataset X has {} sequences.".format(len(self.x_data)))    # dataset shape check
        print("Dataset Y has {} sequences.".format(len(self.y_data)))
        print("Dataset X has {} shape.".format(self.x_data.shape))

        # batch_size: Mini batch size
        # data_dims: How many features at once? (Characters, Words, etc.)
        # output_dim: How many features per output?
        # seq_len: How many sequences per output?
        self.batch_size = len(self.x_data)
        self.data_dims = len(self.x_data)
        self.hidden_size = self.vocabulary_size
        
        self.X = tf.placeholder(tf.int32, [None, self.seq_len])
        self.Y = tf.placeholder(tf.int32, [None, self.seq_len])

        # One-hot encoding
        # X_one_hot = tf.one_hot(self.X, self.vocabulary_size)
        # print(X_one_hot)
        
        # Embedding, with input dimension of vocab size
        self.embedding = tf.get_variable("embedding", [self.seq_len, self.vocabulary_size], dtype=tf.float32)
        embed = tf.nn.embedding_lookup(self.embedding, self.X)
        
        # LSTM Cell
        outputs = self.MultiRNN_Dynamic(embed)
        
        # Add Softmax Layer
        # input_fc = tf.reshape(outputs, [-1, self.hidden_size])    # If one-hot encoded...
        self.outputs = tf.contrib.layers.fully_connected(outputs,
                                                        self.vocabulary_size,
                                                        activation_fn=None)

        # Initialize fc weights with Ones
        # If all weights are not same, loss will explode!!
        weights = tf.ones([self.batch_size, self.seq_len])

        # Monitor loss
        loss = tf.contrib.seq2seq.sequence_loss(logits=self.outputs,
                                                targets=self.Y,
                                                weights=weights)
        self.mean_loss = tf.reduce_mean(loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = optimizer.minimize(self.mean_loss)

    def Train(self, sess):
        # Set up initializers
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        print('=' * 20, "{:^20}".format("Training Start"), '=' * 20)

        self.iter_loss = []
        self.elapsed = []
        loss_prev = 99999
        
        self.start = time.time()

        # Begin Training
        for i in range(self.epoch):
            _, l, results = sess.run(
                [self.train_op, self.mean_loss, self.outputs],
                feed_dict={self.X: self.x_data, self.Y: self.y_data})
            self.end = time.time()
            self.elapsed.append(self.end - self.start)    # Time & Iteration
            self.iter_loss.append(l)    # Iteration & Loss
            for j, result in enumerate(results):
                index = np.argmax(result, axis=1)
                if i % 100 == 0 and j == 0:
                    print("\n At step", i, ':',
                        ''.join([self.ids_vocab[t] for t in index]))
                    print('Loss:', l)
                    self.end = time.time()
                    self.Elapsed()
            if loss_prev > l:
                self.Save_Model(saver, sess, "BEST", output=False)
                loss_prev = l
            elif i % self.save_point is 0:
                self.Save_Model(saver, sess, i)

    def Embedding_Tensorboard(self, sess):
        # Tensorboard Embedding Visualization
        filepath = "C:/Users/jungw/OneDrive/??/GitHub/Text_Generation/graphs/Embedding/"
        filename = "vocabs.csv"
        file = filepath + filename
        self.Make_CSV(file)

        sess.run(self.embedding.initializer)    # Initialize Embedding Variable
        config = projector.ProjectorConfig()    # Create Projector config
        embedding = config.embeddings.add()     # Add Embedding Visualizer
        embedding.tensor_name = self.embedding.name     # Attach the name of the variable
        embedding.metadata_path = filename      # Metafile
        writer = tf.summary.FileWriter(filepath, sess.graph)    # Create summary writer
        projector.visualize_embeddings(writer, config)  # Add writer and config to Projector
        saver_embed = tf.train.Saver([self.embedding])  # Save the model
        saver_embed.save(sess, './graphs/Embedding/embedding.ckpt', 1)

if __name__ == "__main__":
    save_at = "C:/Set/Your/Directory/Save_Your_Model.ckpt"
    
    code_gen = TextGen(filename = "SAMPLE.py",
                    learning_rate = 0.1,
                    num_layers = 2,
                    seq_len = 40,
                    epoch = 5001,
                    save_point = 1000,
                    save_at = save_at,
                    encode=False,
                    chunk_word=False,
                    stride=1)
    
    code_gen.Prepare_Model()
    sess = tf.Session()
    code_gen.Train(sess)
    code_gen.Embedding_Tensorboard(sess)
    
    # Generate Text
    code_gen.Make_Text(sess)

    # Monitor Loss, and Time info
    code_gen.Plot_Loss()
    
    # Load Model of certain point
    saver = tf.train.Saver()
    saver.restore(sess, './Models/rnn_text_BEST.ckpt')
    code_gen.Make_Text(sess)