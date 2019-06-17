from __future__ import print_function
import os
import tensorflow as tf
import numpy as np
import ccnn as NETwork
import datetime
from sklearn.metrics import average_precision_score
import data_manager
import test as evaluation


# Model Hyperparameters
tf.flags.DEFINE_string('filter_sizes', '3', 'Comma-separated filter sizes default: (3,4,5)')
tf.flags.DEFINE_integer('pattern_num', 10, 'No. of patterns for each relation (default: 10)')
tf.flags.DEFINE_float('l2_reg_omega', 0.001, 'l2_regularizer weight (default: 0.0005)')

# Eval Parameters
tf.flags.DEFINE_string('checkpoint_dir', '../../runs/bag/', 'Checkpoint directory from training run')
tf.flags.DEFINE_string('model', 'model', "Model dir for evaluation, default: (model)")
tf.flags.DEFINE_boolean("use_types", False, "Use entity types (default: True")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
"""
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")
"""


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


            # Get Evaluator for evaluation
            evaluator = evaluation.Evaluator(dataset, sess, network, model_dir, settings)
            evaluator.test()


if __name__ == '__main__':
    tf.app.run()
