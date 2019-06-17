from __future__ import print_function
import os
import tensorflow as tf
import numpy as np
import ccnn as NETwork
import datetime
from sklearn.metrics import average_precision_score
import data_manager
import test as evaluation

"""
You must set batch_size=1 firstly
"""

# Model Hyperparameters
tf.flags.DEFINE_string('filter_sizes', '3,4,5', 'Comma-separated filter sizes default: (3,4,5)')
tf.flags.DEFINE_integer('pattern_num', 5, 'No. of patterns for each relation (default: 10)')
tf.flags.DEFINE_float('l2_reg_omega', 0.001, 'l2_regularizer weight (default: 0.0005)')

# Eval Parameters
tf.flags.DEFINE_string('checkpoint_dir', '../../runs/bag/', 'Checkpoint directory from training run')
tf.flags.DEFINE_string('model', 'model-5', "Model dir for evaluation, default: (model)")
tf.flags.DEFINE_boolean("use_types", True, "Use entity types (default: True")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


def main(_):
    """
    main function
    """
    dataset = data_manager.DataManager()
    model_dir = '../../runs/bag/{}'.format(FLAGS.model)

    time_str = datetime.datetime.now().isoformat()
    print('{}: start test'.format(time_str))

    print('reading wordembedding')
    wordembedding = np.load('../../data/bag_data/vec.npy')

    settings = NETwork.Settings()
    settings.vocab_size = len(wordembedding)
    settings.filter_sizes = list(map(int, FLAGS.filter_sizes.split(',')))
    settings.pattern_num = FLAGS.pattern_num
    settings.l2_reg_omega = FLAGS.l2_reg_omega

    #checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir + FLAGS.model + '/checkpoints/')
    checkpoint_file = FLAGS.checkpoint_dir + FLAGS.model + '/model-best'
    graph = tf.Graph()
    with graph.as_default():
        gpu_options = tf.GPUOptions(allow_growth=True)
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement,
            gpu_options=gpu_options)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            time_str = datetime.datetime.now().isoformat()
            print('{}: construct network...'.format(time_str))
            # saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_file))
            # saver.restore(sess, checkpoint_file)
            network = NETwork.CNN(word_embeddings=dataset.wordembedding, settings=settings, is_training=False, is_evaluating=True, use_types=FLAGS.use_types)
            saver = tf.train.Saver()
            time_str = datetime.datetime.now().isoformat()
            print('{}: restore checkpoint file: {}'.format(time_str, checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # test one entity relation mentions
            time_str = datetime.datetime.now().isoformat()
            print('{}: testing...'.format(time_str))


            def eval_step(test_word, test_pos1, test_pos2, test_type, test_y):
                num_batches_per_epoch = int((len(test_y)-1)/settings.batch_size) + 1
                for batch_num in range(num_batches_per_epoch):
                    start_index = batch_num * settings.batch_size
                    end_index = min((batch_num+1)*settings.batch_size, len(test_y))
                    if (end_index - start_index) != settings.batch_size:
                        start_index = end_index - settings.batch_size

                    word_batch = test_word[start_index: end_index]
                    pos1_batch = test_pos1[start_index: end_index]
                    pos2_batch = test_pos2[start_index: end_index]
                    type_batch = test_type[start_index: end_index]
                    y_batch = test_y[start_index: end_index]

                    attentions = eval_op(word_batch, pos1_batch, pos2_batch, type_batch, y_batch)
                    print(attentions)


            def eval_op(word_batch, pos1_batch, pos2_batch, type_batch, y_batch):
                """
                evaluate a batch
                """
                total_word = []
                total_pos1 = []
                total_pos2 = []
                total_type = []
                total_shape_batch = []
                total_num = 0


                for i in range(len(word_batch)):
                    total_shape_batch.append(total_num)
                    total_num += len(word_batch[i])

                    for j in range(len(word_batch[i])):
                        total_word.append(word_batch[i][j])
                        total_pos1.append(pos1_batch[i][j])
                        total_pos2.append(pos2_batch[i][j])
                        total_type.append(type_batch[i][j])

                # Here total_word and y_batch are not equal, total_word[total_shape[i]:total_shape[i+1]] is related to y_batch[i]
                total_shape_batch.append(total_num)

                total_shape_batch = np.array(total_shape_batch)
                total_word = np.array(total_word)
                total_pos1 = np.array(total_pos1)
                total_pos2 = np.array(total_pos2)
                total_type = np.array(total_type)

                feed_dict = {
                    network.input_word: total_word,
                    network.input_pos1: total_pos1,
                    network.input_pos2: total_pos2,
                    network.input_type: total_type,
                    #self.network.input_y: y_batch,
                    network.total_shape: total_shape_batch,
                    network.dropout_keep_prob: 1.0
                }

                attentions = sess.run([network.attention], feed_dict)

                return attentions

            eval_step(dataset.test_word, dataset.test_pos1, dataset.test_pos2, dataset.test_type, dataset.test_y)


if __name__ == '__main__':
    tf.app.run()
