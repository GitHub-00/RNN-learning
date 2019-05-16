
#构建计算图--LSTM模型
#   embeding
#   LSTM
#   fc
#   train_op
#训练流程代码
#数据集封装
#API next_batch(batch_size)
#词表封装 API setentce2id(text_sentence) :句子转换ID
#类别封装 API category(text_category)

import tensorflow as tf
import os
import sys
import numpy as np
import math


tf.logging.set_verbosity(tf.logging.INFO)

def get_default_param():
    return tf.contrib.training.HParams(
        num_embedding_size = 16,
        num_timesteps = 50,
        num_lstm_nodes = [32,32],
        num_lstm_layers = 2,
        num_fc_nodes = 32,
        batch_size = 100,
        clip_lstm_grads = 1.0, #控制梯度
        learning_rate = 0.001,
        num_word_threshold = 10
    )

hps = get_default_param()

train_file = os.path.abspath(os.path.join(os.getcwd(),'../cnews/cnews.train.seg.txt'))
val_file = os.path.abspath(os.path.join(os.getcwd(),'../cnews/cnews.val.seg.txt'))
test_file = os.path.abspath(os.path.join(os.getcwd(),'../cnews/cnews.test.seg.txt'))
vocab_file = os.path.abspath(os.path.join(os.getcwd(),'../cnews/cnews.vocab.txt'))
category_file = os.path.abspath(os.path.join(os.getcwd(),'../cnews/cnews.category.txt'))
output_folder = '../cnews/run_text_fun'

if not os.path.exists(output_folder):
    os.mkdir(output_folder)


class Vocab:
    def __init__(self, filename, num_word_threshold):
        self._word_to_id = {}
        self._unk = -1
        self._num_word_threshold = num_word_threshold
        self._read_dict(filename)

    def _read_dict(self, filename):
        with open(filename,'r',encoding='utf-8') as f: # for windows
        #with open(filename,'r') as f: for mac
            lines = f.readlines()
        for line in lines:
            word, frequency = line.strip('\r\n').split('\t')
            #word = word.decode('utf-8')
            frequency = int(frequency)
            #if frequency < self._num_word_threshold:
            #    continue
            idx = len(self._word_to_id)
            if word == '<UNK>':
                self._unk = idx
            self._word_to_id[word] = idx

    def word_to_id(self, word):
        return self._word_to_id.get(word,self._unk)

    @property
    def unk(self):
        return self._unk

    def size(self):
        return len(self._word_to_id)

    def sentence_to_id(self, sentence):
        word_ids = [self._word_to_id[cur_word] for cur_word in sentence.split()]
        return word_ids


class CategoryDict:
    def __init__(self, filename):
        self._category_to_id = {}
        with open(filename,'r',encoding='utf-8') as f: # for windows
        #with open(filename,'r') as f: # for mac
            lines = f.readlines()
        for line in lines:
            category = line.strip('\r\n')
            idx = len(self._category_to_id)
            self._category_to_id[category] = idx

    def category_to_id(self,category):
        if not category in self._category_to_id:
            raise Exception('%s is not in out category' % category)
        return self._category_to_id[category]

    def size(self):
        return len(self._category_to_id)

vocab =Vocab(vocab_file, hps.num_word_threshold)
vocab_size = vocab.size()
tf.logging.info('vocab_size: %d' %vocab.size())

#测试API sentence_to_id
#test_str = 'Mordin'
#tf.logging.info('label: %s, id: %s' %(test_str,vocab.sentence_to_id(test_str)))

category_vocab = CategoryDict(category_file)
num_classes = category_vocab.size()
#print(num_classes)
#测试API category_to_id
#test_str = '时尚'
#tf.logging.info('label: %s, id: %d' %(test_str,category_vocab.category_to_id(test_str)))

class TextDataSet:
    def __init__(self, filename, vocab, category_vocab, num_timesteps):
        self._vocab = vocab
        self._category_vocab = category_vocab
        self._num_timesteps = num_timesteps
        self._inputs = []
        self._outputs = []
        self._indicator = 0
        self._parse_file(filename)

    def _parse_file(self, filename):
        tf.logging.info('loading data from %s'  %filename)
        with open(filename,'r',encoding='utf-8') as f: # for windows
        # with open(filename,'r') as f: # for mac
            lines = f.readlines()
        for line in lines:
            label, content = line.strip('\r\n').split('\t')
            id_label= self._category_vocab.category_to_id(label)
            id_words = self._vocab.sentence_to_id(content)
            id_words = id_words[0: self._num_timesteps]
            padding_num = self._num_timesteps - len(id_words)
            id_words = id_words + [self._vocab.unk for i in range(padding_num)]
            self._inputs.append(id_words)
            self._outputs.append(id_label)
        self._inputs = np.asarray(self._inputs, dtype=np.int32)
        self._outputs = np.asarray(self._outputs, dtype=np.int32)
        self._random_shuffle()

    def _random_shuffle(self):
        p = np.random.permutation(len(self._inputs))
        self._inputs = self._inputs[p]
        self._outputs = self._outputs[p]

    def next_batch(self, batch_size):
        end_indicator = self._indicator + batch_size
        if end_indicator > len(self._inputs):
            self._random_shuffle()
            self._indicator = 0
            end_indicator = batch_size
        if end_indicator> len(self._inputs):
            raise Exception('batch_size : %d is too large' %batch_size)

        batch_inputs = self._inputs[self._indicator: end_indicator]
        batch_outputs = self._outputs[self._indicator: end_indicator]
        self._indicator = end_indicator
        return batch_inputs, batch_outputs

train_dataset = TextDataSet(train_file, vocab, category_vocab, hps.num_timesteps)
#val_dataset = TextDataSet(val_file, vocab, category_vocab, hps.num_timesteps)
#test_dataset = TextDataSet(test_file, vocab, category_vocab, hps.num_timesteps)
#测试 next_batch
print(train_dataset.next_batch(2))
#print(val_dataset.next_batch(2))
#print(test_dataset.next_batch(2))


def create_mode(hps, vocab_size, num_classes):
    num_timesteps = hps.num_timesteps
    batch_size = hps.batch_size

    inputs = tf.placeholder(tf.int32, (batch_size, num_timesteps))
    outputs = tf.placeholder(tf.int32, (batch_size,))
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    global_step = tf.Variable(tf.zeros([], tf.int64), name='global_step', trainable=False)





