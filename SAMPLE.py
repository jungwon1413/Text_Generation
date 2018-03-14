import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import logging
import time
import numpy as np
import tensorflow as tf

logging.getLogger('tensorflow').setLevel(logging.WARNING)

class CharRNN(object):
  def __init__(self, is_training, batch_size, num_unrollings, vocab_size, 
               hidden_size, max_grad_norm, embedding_size, num_layers,
               learning_rate, model, dropout=0.0, input_dropout=0.0, use_batch=True):
    self.batch_size = batch_size
    self.num_unrollings = num_unrollings
    if not use_batch:
      self.batch_size = 1
      self.num_unrollings = 1
    self.hidden_size = hidden_size
    self.vocab_size = vocab_size
    self.max_grad_norm = max_grad_norm
    self.num_layers = num_layers
    self.embedding_size = embedding_size
    self.model = model
    self.dropout = dropout
    self.input_dropout = input_dropout
    if embedding_size <= 0:
      self.input_size = vocab_size
      self.input_dropout = 0.0
    else:
      self.input_size = embedding_size
    self.model_size = (embedding_size * vocab_size +
                       4 * hidden_size * (hidden_size + self.input_size + 1) +
                       vocab_size * (hidden_size + 1) +
                       (num_layers - 1) * 4 * hidden_size *
                       (hidden_size + hidden_size + 1))

    self.input_data = tf.placeholder(tf.int64,
                                     [self.batch_size, self.num_unrollings],
                                     name='inputs')
    self.targets = tf.placeholder(tf.int64,
                                  [self.batch_size, self.num_unrollings],
                                  name='targets')

    if self.model == 'rnn':
      cell_fn = tf.contrib.rnn.BasicRNNCell
    elif self.model == 'lstm':
      cell_fn = tf.contrib.rnn.BasicLSTMCell
    elif self.model == 'gru':
      cell_fn = tf.contrib.rnn.GRUCell

    params = {}
    if self.model == 'lstm':
      params['forget_bias'] = 0.0
      params['state_is_tuple'] = True
    cell = cell_fn(
        self.hidden_size, reuse=tf.get_variable_scope().reuse,
        **params)

    cells = [cell]
    for i in range(self.num_layers-1):
      higher_layer_cell = cell_fn(
          self.hidden_size, reuse=tf.get_variable_scope().reuse,
          **params)
      cells.append(higher_layer_cell)

    if is_training and self.dropout > 0:
      cells = [tf.contrib.rnn.DropoutWrapper(
        cell,
        output_keep_prob=1.0-self.dropout)
               for cell in cells]

    multi_cell = tf.contrib.rnn.MultiRNNCell(cells)

    with tf.name_scope('initial_state'):
      self.zero_state = multi_cell.zero_state(self.batch_size, tf.float32)

      self.initial_state = create_tuple_placeholders_with_default(
        multi_cell.zero_state(batch_size, tf.float32),
        extra_dims=(None,),
        shape=multi_cell.state_size)      
      

    with tf.name_scope('embedding_layer'):
      if embedding_size > 0:
        self.embedding = tf.get_variable(
          'embedding', [self.vocab_size, self.embedding_size])
      else:
        self.embedding = tf.constant(np.eye(self.vocab_size), dtype=tf.float32)

      inputs = tf.nn.embedding_lookup(self.embedding, self.input_data)
      if is_training and self.input_dropout > 0:
        inputs = tf.nn.dropout(inputs, 1 - self.input_dropout)

    with tf.name_scope('slice_inputs'):
      sliced_inputs = [tf.squeeze(input_, [1])
                       for input_ in tf.split(axis=1, num_or_size_splits=self.num_unrollings, value=inputs)]
      
    outputs, final_state = tf.contrib.rnn.static_rnn(
      multi_cell, sliced_inputs,
      initial_state=self.initial_state)

    self.final_state = final_state

    with tf.name_scope('flatten_ouputs'):
      flat_outputs = tf.reshape(tf.concat(axis=1, values=outputs), [-1, hidden_size])

    with tf.name_scope('flatten_targets'):
      flat_targets = tf.reshape(tf.concat(axis=1, values=self.targets), [-1])
    
    with tf.variable_scope('softmax') as sm_vs:
      softmax_w = tf.get_variable("softmax_w", [hidden_size, vocab_size])
      softmax_b = tf.get_variable("softmax_b", [vocab_size])
      self.logits = tf.matmul(flat_outputs, softmax_w) + softmax_b
      self.probs = tf.nn.softmax(self.logits)

    with tf.name_scope('loss'):
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=self.logits, labels=flat_targets)
      self.mean_loss = tf.reduce_mean(loss)

    with tf.name_scope('loss_monitor'):
      count = tf.Variable(1.0, name='count')
      sum_mean_loss = tf.Variable(1.0, name='sum_mean_loss')
      
      self.reset_loss_monitor = tf.group(sum_mean_loss.assign(0.0),
                                         count.assign(0.0),
                                         name='reset_loss_monitor')
      self.update_loss_monitor = tf.group(sum_mean_loss.assign(sum_mean_loss +
                                                               self.mean_loss),
                                          count.assign(count + 1),
                                          name='update_loss_monitor')
      with tf.control_dependencies([self.update_loss_monitor]):
        self.average_loss = sum_mean_loss / count
        self.ppl = tf.exp(self.average_loss)

      loss_summary_name = "average loss"
      ppl_summary_name = "perplexity"
  
      average_loss_summary = tf.summary.scalar(loss_summary_name, self.average_loss)
      ppl_summary = tf.summary.scalar(ppl_summary_name, self.ppl)

    self.summaries = tf.summary.merge([average_loss_summary, ppl_summary],
                                      name='loss_monitor')
    
    self.global_step = tf.get_variable('global_step', [],
                                       initializer=tf.constant_initializer(0.0))

    self.learning_rate = tf.constant(learning_rate)
    if is_training:
      tvars = tf.trainable_variables()
      grads, _ = tf.clip_by_global_norm(tf.gradients(self.mean_loss, tvars),
                                        self.max_grad_norm)
      optimizer = tf.train.AdamOptimizer(self.learning_rate)

      self.train_op = optimizer.apply_gradients(zip(grads, tvars),
                                                global_step=self.global_step)
      
  def run_epoch(self, session, data_size, batch_generator, is_training,
                verbose=0, freq=10, summary_writer=None, debug=False, divide_by_n=1):
    epoch_size = data_size // (self.batch_size * self.num_unrollings)
    if data_size % (self.batch_size * self.num_unrollings) != 0:
        epoch_size += 1

    if verbose > 0:
        logging.info('epoch_size: %d', epoch_size)
        logging.info('data_size: %d', data_size)
        logging.info('num_unrollings: %d', self.num_unrollings)
        logging.info('batch_size: %d', self.batch_size)

    if is_training:
      extra_op = self.train_op
    else:
      extra_op = tf.no_op()

    state = session.run(self.zero_state)
    self.reset_loss_monitor.run()
    start_time = time.time()
    for step in range(epoch_size // divide_by_n):
      data = batch_generator.next()
      inputs = np.array(data[:-1]).transpose()
      targets = np.array(data[1:]).transpose()

      ops = [self.average_loss, self.final_state, extra_op,
             self.summaries, self.global_step, self.learning_rate]

      feed_dict = {self.input_data: inputs, self.targets: targets,
                   self.initial_state: state}

      results = session.run(ops, feed_dict)
      average_loss, state, _, summary_str, global_step, lr = results
      
      ppl = np.exp(average_loss)
      if (verbose > 0) and ((step+1) % freq == 0):
        logging.info("%.1f%%, step:%d, perplexity: %.3f, speed: %.0f words",
                     (step + 1) * 1.0 / epoch_size * 100, step, ppl,
                     (step + 1) * self.batch_size * self.num_unrollings /
                     (time.time() - start_time))

    logging.info("Perplexity: %.3f, speed: %.0f words per sec",
                 ppl, (step + 1) * self.batch_size * self.num_unrollings /
                 (time.time() - start_time))
    return ppl, summary_str, global_step

  def sample_seq(self, session, length, start_text, vocab_index_dict,
                 index_vocab_dict, temperature=1.0, max_prob=True):

    state = session.run(self.zero_state)

    if start_text is not None and len(start_text) > 0:
      seq = list(start_text)
      for char in start_text[:-1]:
        x = np.array([[char2id(char, vocab_index_dict)]])
        state = session.run(self.final_state,
                            {self.input_data: x,
                             self.initial_state: state})
      x = np.array([[char2id(start_text[-1], vocab_index_dict)]])
    else:
      vocab_size = len(vocab_index_dict.keys())
      x = np.array([[np.random.randint(0, vocab_size)]])
      seq = []

    for i in range(length):
      state, logits = session.run([self.final_state,
                                   self.logits],
                                  {self.input_data: x,
                                   self.initial_state: state})
      unnormalized_probs = np.exp((logits - np.max(logits)) / temperature)
      probs = unnormalized_probs / np.sum(unnormalized_probs)

      if max_prob:
        sample = np.argmax(probs[0])
      else:
        sample = np.random.choice(self.vocab_size, 1, p=probs[0])[0]

      seq.append(id2char(sample, index_vocab_dict))
      x = np.array([[sample]])
    return ''.join(seq)
      
class BatchGenerator(object):
    def __init__(self, text, batch_size, n_unrollings, vocab_size,
                 vocab_index_dict, index_vocab_dict):
      self._text = text
      self._text_size = len(text)
      self._batch_size = batch_size
      self.vocab_size = vocab_size
      self._n_unrollings = n_unrollings
      self.vocab_index_dict = vocab_index_dict
      self.index_vocab_dict = index_vocab_dict
      
      segment = self._text_size // batch_size

      self._cursor = [ offset * segment for offset in range(batch_size)]
      self._last_batch = self._next_batch()
      
    def _next_batch(self):
      batch = np.zeros(shape=(self._batch_size), dtype=np.float)
      for b in range(self._batch_size):
        batch[b] = char2id(self._text[self._cursor[b]], self.vocab_index_dict)
        self._cursor[b] = (self._cursor[b] + 1) % self._text_size
      return batch

    def next(self):
      batches = [self._last_batch]
      for step in range(self._n_unrollings):
        batches.append(self._next_batch())
      self._last_batch = batches[-1]
      return batches

def batches2string(batches, index_vocab_dict):
  s = [''] * batches[0].shape[0]
  for b in batches:
    s = [''.join(x) for x in zip(s, id2char_list(b, index_vocab_dict))]
  return s

def characters(probabilities):
  return [id2char(c) for c in np.argmax(probabilities, 1)]

def char2id(char, vocab_index_dict):
  try:
    return vocab_index_dict[char]
  except KeyError:
    logging.info('Unexpected char %s', char)
    return 0

def id2char(index, index_vocab_dict):
  return index_vocab_dict[index]
    
def id2char_list(lst, index_vocab_dict):
  return [id2char(i, index_vocab_dict) for i in lst]

def create_tuple_placeholders_with_default(inputs, extra_dims, shape):
  if isinstance(shape, int):
    result = tf.placeholder_with_default(
      inputs, list(extra_dims) + [shape])
  else:
    subplaceholders = [create_tuple_placeholders_with_default(
      subinputs, extra_dims, subshape)
                       for subinputs, subshape in zip(inputs, shape)]
    t = type(shape)
    if t == tuple:
      result = t(subplaceholders)
    else:
      result = t(*subplaceholders)    
  return result
        
def create_tuple_placeholders(dtype, extra_dims, shape):
  if isinstance(shape, int):
    result = tf.placeholder(dtype, list(extra_dims) + [shape])
  else:
    subplaceholders = [create_tuple_placeholders(dtype, extra_dims, subshape)
                       for subshape in shape]
    t = type(shape)

    if t == tuple:
      result = t(subplaceholders)
    else:
      result = t(*subplaceholders)
  return result