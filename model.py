import numpy as np
import tensorflow as tf

class ModelConfig(object):
    """
    模型的配置参数
    """
    # model config
    embedding_dim = 200   # 词向量维度
    title_length = 218   # 序列长度 18 + 20 * 10 = 218
    word_length = 8    #  keyword的序列 8
    num_classes = 2  # 类别数
    vocab_size = 100000  # 词汇表大小

    num_layers = 2    #  层数
    hidden_dim = 128  # 隐藏层神经元
    rnn = 'gru'  # lstm 或 gru

    dropout_keep_prob = 0.8  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 128  # 每批训练大小
    num_epochs = 10  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard

class DSSMModel(object):
    """:param embedding， 两层中间层，得到描述向量"""
    def __init__(self, config):
        self.config = config

        # 输入title，word，label
        self.input_title = tf.placeholder(tf.int32, [None, self.config.title_length], name='input_title')
        self.input_word = tf.placeholder(tf.int32, [None, self.config.word_length], name='input_word')
        self.input_label = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_label')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.model()

    def model(self):
        """:param rnn模型"""
        def lstm_cell():    # lstm
            return tf.contrib.rnn.BasicLSTMCell(self.config.hidden_dim, state_id_tuple=True)

        def gru_cell():    # lstm
            return tf.contrib.rnn.GRUCell(self.config.hidden_dim)

        def dropout():   # rnn后加一个dropout layer
            if (self.config.rnn == 'lstm'):
                cell = lstm_cell()
            else:
                cell = gru_cell()
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        # word embedding
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputsTitle = tf.nn.embedding_lookup(embedding, self.input_title)
            embedding_inputsWord = tf.nn.embedding_lookup(embedding, self.input_word)

        with tf.variable_scope('rnn1'):
            # rnn for title
            cells_title = [dropout() for _ in range(self.config.num_layers)]
            rnn_cells_title =  tf.contrib.rnn.MultiRNNCell(cells_title, state_is_tuple=True)

            _outputs_title, _ = tf.nn.dynamic_rnn(cell=rnn_cells_title, inputs=embedding_inputsTitle, dtype=tf.float32)
            last_title = _outputs_title[:, -1, :]      # 取最后一个时序输出作为结果

        with tf.variable_scope('rnn2'):
            # rnn for words
            cells_word = [dropout() for _ in range(self.config.num_layers)]
            rnn_cells_word =  tf.contrib.rnn.MultiRNNCell(cells_word, state_is_tuple=True)

            _outputs_word, _ = tf.nn.dynamic_rnn(cell=rnn_cells_word, inputs=embedding_inputsWord, dtype=tf.float32)
            last_word = _outputs_word[:, -1, :]      # 取最后一个时序输出作为结果

        with tf.variable_scope('FC1'):
            title_fc = tf.layers.dense(last_title, self.config.hidden_dim, name='title_fc')
            title_fc = tf.contrib.layers.dropout(title_fc, self.keep_prob)
            title_fc = tf.nn.relu(title_fc)

        with tf.variable_scope('FC1'):
            word_fc = tf.layers.dense(last_word, self.config.hidden_dim, name='word_fc')
            word_fc = tf.contrib.layers.dropout(word_fc, self.keep_prob)
            word_fc = tf.nn.relu(word_fc)

        with tf.variable_scope('Cosine_Similarity'):
            title_norm = tf.sqrt(tf.reduce_sum(tf.square(title_fc), 1, True))
            word_norm = tf.sqrt(tf.reduce_sum(tf.square(word_fc), 1, True))
            prod = tf.reduce_sum(tf.multiply(title_fc, word_fc), 1, True)
            norm_prod = tf.multiply(title_norm, word_norm)

            self.cos_sim_raw = tf.truediv(prod, norm_prod)
            # cos_sim = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw),[2,self.config.batch_size])) * 20 # not know whether to use
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.cos_sim_raw),1)
        with tf.name_scope('Loss'):

            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.cos_sim_raw, labels=self.input_label)
            self.loss = tf.reduce_sum(cross_entropy)
            tf.summary.scalar('loss', self.loss)

        with tf.name_scope('Training'):
            # Optimizer
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(tf.argmax(self.input_label,1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


