import tensorflow as tf
import numpy as np

class Settings(object):

    def __init__(self):
        self.vocab_size = 114042
        self.sentence_len = 80
        self.num_classes = 53
        self.filter_sizes = None
        self.num_filter = 230
        self.batch_size = 64
        self.l2_reg_omega = 0.001
        # 0~122, every position has 5 dimensions
        self.pos_num = 123
        self.pos_size = 5

        self.pattern_num = 5
        # 0~3, every type has 5 dimensions
        self.type_num = 4
        self.type_size = 5


class CNN(object):
    """
    A CNN for generate sentence features
    Uses an embedding layer, followed by a convolutional, max-pooling layer
    """
    def __init__(self, word_embeddings, settings, l2_reg_lambda=0.0, is_training=True, is_evaluating=False, use_types=False):

        self.is_training = is_training
        self.is_evaluating = is_evaluating
        self.use_types = use_types
        self.sentence_len = sentence_len = settings.sentence_len
        self.num_classes = num_classes = settings.num_classes
        self.filter_sizes = filter_sizes = settings.filter_sizes
        self.num_filter = num_filter = settings.num_filter
        self.batch_size = batch_size = settings.batch_size
        self.pattern_num = pattern_num = settings.pattern_num
        self.l2_reg_omega = l2_reg_omega = settings.l2_reg_omega

        # Placeholders for input, output and dropout
        self.input_word = tf.placeholder(dtype=tf.int32, shape=[None, sentence_len], name='input_word')
        self.input_pos1 = tf.placeholder(dtype=tf.int32, shape=[None, sentence_len], name='input_pos1')
        self.input_pos2 = tf.placeholder(dtype=tf.int32, shape=[None, sentence_len], name='input_pos2')
        self.input_type = tf.placeholder(dtype=tf.int32, shape=[None, sentence_len], name='input_type')
        self.input_y = tf.placeholder(dtype=tf.float32, shape=[None, num_classes], name='input_y')
        self.total_shape = tf.placeholder(dtype=tf.int32, shape=[batch_size+1], name='total_shape')
        # Propertion of dropout
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, name='dropout_keep_prob')
        # Keeping track of l2 regularization loss
        self.l2_loss = tf.constant(0.0)

        """
        Embedding layer
        """
        with tf.name_scope('embedding'):
            with tf.variable_scope('vs_embedding', reuse=(self.is_training and self.is_evaluating)):
                word_embedding = tf.get_variable('word_embedding',initializer=word_embeddings)
                pos1_embedding = tf.get_variable('pos1_embedding',[settings.pos_num,settings.pos_size])
                pos2_embedding = tf.get_variable('pos2_embedding',[settings.pos_num,settings.pos_size])
                if self.use_types:
                    type_embedding = tf.get_variable('type_embedding',[settings.type_num,settings.type_size])
            self.embedding_size = word_embeddings.shape[1]

            if self.use_types:
                self.embedded_chars = tf.concat(
                    [tf.nn.embedding_lookup(word_embedding,self.input_word),
                    tf.nn.embedding_lookup(pos1_embedding,self.input_pos1),
                    tf.nn.embedding_lookup(pos2_embedding,self.input_pos2),
                    tf.nn.embedding_lookup(type_embedding,self.input_type)],
                    2)
            else:
                self.embedded_chars = tf.concat(
                    [tf.nn.embedding_lookup(word_embedding,self.input_word),
                    tf.nn.embedding_lookup(pos1_embedding,self.input_pos1),
                    tf.nn.embedding_lookup(pos2_embedding,self.input_pos2)],
                    2)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)


        """
        Create a convolution + maxpool for each filter
        """
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope('conv-maxpool-%s' %filter_size):
                # Convolution layer
                if self.use_types:
                    filter_shape = [filter_size, self.embedding_size + 2*settings.pos_size + settings.type_size, 1, self.num_filter]
                else:
                    filter_shape = [filter_size, self.embedding_size + 2*settings.pos_size, 1, self.num_filter]
                with tf.variable_scope('vs_conv-maxpool-{}'.format(filter_size), reuse=(self.is_training and self.is_evaluating)):
                    W = tf.get_variable(initializer=tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.get_variable(initializer=tf.constant(0.1, shape=[self.num_filter]), name='b')
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded, 
                    W, 
                    strides=[1,1,1,1], 
                    padding='VALID', 
                    name='conv')
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv,b), name='relu')
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.sentence_len-filter_size+1, 1, 1],
                    strides=[1,1,1,1],
                    padding='VALID',
                    name='pool')
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.num_filter * len(self.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        # Flatten the output, each sentence's output has dimension of num_filters_total
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])


        """
        Add dropout
        """
        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)


        """
        Add attention
        """
        with tf.name_scope('attention'):
            with tf.variable_scope('vs_attention', reuse=(self.is_training and self.is_evaluating)):
                self.patterns = tf.get_variable(
                    'patterns',
                    shape=[num_classes, pattern_num, num_filters_total],
                    #initializer=tf.contrib.layers.xavier_initializer())
                    initializer=tf.zeros_initializer())
                self.pattern_weights = tf.get_variable(
                    'pattern_weights',
                    shape=[num_classes, pattern_num],
                    dtype=tf.int32,
                    initializer=tf.zeros_initializer(),
                    trainable=False)

            self.l2_loss += tf.nn.l2_loss(self.patterns)

            self.att_h_drop = []

            for i in range(batch_size):
                size = self.total_shape[i+1] - self.total_shape[i]
                sen_list = self.h_drop[self.total_shape[i]:self.total_shape[i+1]]

                if not self.is_evaluating:
                    # we know the label
                    label = tf.cast(tf.argmax(self.input_y[i], axis=0), tf.int32)
                    # each sentence was expended to the number of pattern_num
                    # sentence_expand = tf.matmul(sen_list, tf.ones([num_filters_total, pattern_num], tf.float32))
                    sentence_expand = tf.reshape(tf.tile(sen_list, multiples=[1, pattern_num]), [-1, num_filters_total])
                    # each pattern of class label was expended to the number of size
                    # pattern_expand = tf.matmul(tf.ones([size, num_filters_total], tf.float32), tf.transpose(self.patterns[label]))
                    pattern_expand = tf.tile(self.patterns[label], multiples=[size, 1])
                    # distances from each sentence to each pattern
                    distances = tf.sqrt(tf.reshape(tf.reduce_sum(tf.square(sentence_expand-pattern_expand), 1), [-1, pattern_num]))

                    """
                    sequential cluster
                    """
                    i = tf.constant(0)
                    while_condition = lambda i: tf.less(i, size)
                    def body(i):
                        nearest_pattern = tf.cast(tf.argmin(distances[i], axis=0), tf.int32)
                        tf.scatter_nd_add(self.pattern_weights, [[label, nearest_pattern]], [1])
                        update = tf.cast((1/self.pattern_weights[label][nearest_pattern]), tf.float32) * (sen_list[i] - self.patterns[label][nearest_pattern])
                        tf.scatter_nd_add(self.patterns, [[label, nearest_pattern]],
                                          [update])
                        return tf.add(i,1)
                    tf.while_loop(while_condition, body, [i])

                    attention = tf.reduce_sum(tf.reciprocal(distances), 1)
                    attention = tf.reshape(tf.nn.softmax(tf.reshape(attention, [size])),[1, size])
                    att_sen = tf.reshape(tf.matmul(attention, sen_list), [1, num_filters_total])
                    self.att_h_drop.append(att_sen)
                else:
                    # we don't know the label, so each sentence may express one relation, it depends on its pattern
                    sentence_expand = tf.reshape(tf.tile(sen_list, multiples=[1, pattern_num * num_classes]), [-1, pattern_num * num_classes, num_filters_total])
                    pattern_expand = tf.reshape(tf.tile(tf.reshape(self.patterns, [pattern_num * num_classes, num_filters_total]), multiples=[size, 1]), [size, -1, num_filters_total])
                    distances = tf.sqrt(tf.reduce_sum(tf.square(sentence_expand - pattern_expand), 2))
                    attention = tf.reshape(tf.nn.softmax(tf.reduce_sum(tf.reciprocal(distances), 1)),[1, size])
                    # term1 = self.num_classes * self.pattern_num * tf.reduce_mean(tf.square(sen_list), 1)
                    # term2 = tf.reduce_mean(tf.square(self.patterns))
                    # term3 = tf.reshape(tf.matmul(sen_list, tf.reshape(tf.reduce_sum(tf.reduce_sum(self.patterns, 0), 0), [-1, 1])), [size])
                    #
                    # attention = tf.reciprocal(tf.add(tf.add(term1, tf.tile([term2], [size])), tf.multiply(tf.constant(-2.0), term3)))
                    # attention = tf.reshape(tf.nn.softmax(attention), [1, size])
                    # att_sen = tf.reshape(tf.matmul(attention, sen_list), [1, num_filters_total])
                    att_sen = tf.matmul(attention, sen_list)
                    self.att_h_drop.append(att_sen)
            self.att_h_drop = tf.reshape(self.att_h_drop, [batch_size, num_filters_total])

        """
        Add output, Final (unnormalized) scores and predictions
        """
        with tf.name_scope('output'):
            with tf.variable_scope('vs_output', reuse=(self.is_training and self.is_evaluating)):
                W = tf.get_variable(
                    "W",
                    shape=[num_filters_total, num_classes],
                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable(initializer=tf.constant(0.1, shape=[num_classes]), name="b")
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.att_h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, axis=1, name='predictions')

        """
        Calculate loss and accuracy
        """
        # CalculateMean cross-entropy loss
        with tf.name_scope('loss'):
            losses = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores, labels=self.input_y, name='losses'), name='reduce_mean_loss')
            #self.l2_regularize_loss = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l2_regularizer(0.0),weights_list=tf.trainable_variables())
            self.l2_regularize_loss = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l2_regularizer(l2_reg_omega),weights_list=[self.patterns])
            self.final_loss = losses + self.l2_regularize_loss + l2_reg_lambda * self.l2_loss

        # Accuracy
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')
