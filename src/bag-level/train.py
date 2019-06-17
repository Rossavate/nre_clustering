from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import ccnn as NETwork
import data_manager
import utils
import test as evaluation

# Model Hyperparameters
tf.flags.DEFINE_string('filter_sizes', '3', 'Comma-separated filter sizes default: (3,4,5)')
tf.flags.DEFINE_float('dropout_keep_prob', 0.5, 'Dropout keep probability (default: 0.5)')
tf.flags.DEFINE_integer('pattern_num', 10, 'No. of patterns for each relation (default: 10)')
tf.flags.DEFINE_float('l2_reg_omega', 0.001, 'l2_regularizer weight (default: 0.001)')

# Training parameters
tf.flags.DEFINE_boolean("allow_evaluation", True, "Allow for evaluation (default: True)")
tf.flags.DEFINE_integer("evaluate_every", 1, "Evaluate model on test set after this many epochs (default: 1)")
tf.flags.DEFINE_integer("checkpoint_every", 1, "Save model after this many epochs (default: 1)")
tf.flags.DEFINE_integer("num_checkpoints", 1, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("num_epochs", 7, 'Number of epochs (default: 3)')
tf.flags.DEFINE_string("model", 'model', 'Model dir number')
tf.flags.DEFINE_boolean("allow_init_data", False, "Allow for initialize data (default: False)")
tf.flags.DEFINE_boolean("use_types", False, "Use entity types (default: True")

#Misc Parameters
tf.flags.DEFINE_boolean('allow_soft_placement', True, 'Allow device soft device placement')
tf.flags.DEFINE_boolean('log_device_placement', False, 'Log placement of ops on devices')

FLAGS = tf.flags.FLAGS
"""
FLAGS._parse_flags()
print('\nParameters:')
for attr, value in sorted(FLAGS.__flags.items()):
    print('{}={}'.format(attr.upper(), value))
print('')
"""


def main(_):
    """
    main function
    """

    dataset = data_manager.DataManager(init_data=FLAGS.allow_init_data)
    model_dir = '../../runs/bag/{}'.format(FLAGS.model)

    settings = NETwork.Settings()
    settings.vocab_size = len(dataset.wordembedding)
    settings.num_classes = len(dataset.train_y[0])
    settings.filter_sizes = list(map(int, FLAGS.filter_sizes.split(',')))
    settings.pattern_num = FLAGS.pattern_num
    settings.l2_reg_omega = FLAGS.l2_reg_omega


    with tf.Graph().as_default():
        #gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.4)
        gpu_options = tf.GPUOptions(allow_growth=True)
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement,
            gpu_options=gpu_options)
        sess = tf.Session(config=session_conf)
        with sess.as_default():

            # Output directory for models and summaries
            # timestamp = str(int(time.time()))
            timestamp = FLAGS.model
            out_dir = os.path.abspath(os.path.join(os.path.pardir, os.path.pardir + '/runs/bag', timestamp))

            print('Construct network for train......')
            network = NETwork.CNN(word_embeddings=dataset.wordembedding, settings=settings, is_training=True, is_evaluating=False, use_types=FLAGS.use_types)

            # Get Evaluator for evaluation
            if FLAGS.allow_evaluation:
                print('Construct network for evaluation......')
                e_network = NETwork.CNN(word_embeddings=dataset.wordembedding, settings=settings, is_training=True, is_evaluating=True, use_types=FLAGS.use_types)
                lastest_score = utils.read_pr(out_dir)
                evaluator = evaluation.Evaluator(dataset, sess, e_network, model_dir, settings, lastest_score)

            # Define training procedure
            global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.AdamOptimizer(0.001)
            grads_and_vars = optimizer.compute_gradients(network.final_loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar('loss', network.final_loss)
            acc_summary = tf.summary.scalar('accuracy', network.accuracy)
            pr_summary = tf.summary.scalar('pr_curve', evaluator.highest_score)

            # Train summaries
            train_summary_op = tf.summary.merge_all()
            train_summary_dir = os.path.join(out_dir, 'summaries', 'train')
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Checkpoint derectory, tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
            checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # train_step
            def train_step(word_batch, pos1_batch, pos2_batch, type_batch, y_batch, mask_batch):
                """
                A single training step
                """
                total_word = []
                total_pos1 = []
                total_pos2 = []
                total_type = []
                total_shape = []
                total_mask = []
                total_num = 0

                for i in range(len(word_batch)):
                    total_shape.append(total_num)
                    total_num += len(word_batch[i])

                    for j in range(len(word_batch[i])):
                        total_word.append(word_batch[i][j])
                        total_pos1.append(pos1_batch[i][j])
                        total_pos2.append(pos2_batch[i][j])
                        total_type.append(type_batch[i][j])
                        total_mask.append(mask_batch[i][j])

                # Here total_word and y_batch are not equal, total_word[total_shape[i]:total_shape[i+1]] is related to y_batch[i]
                total_shape.append(total_num)

                total_shape = np.array(total_shape)
                total_word = np.array(total_word)
                total_pos1 = np.array(total_pos1)
                total_pos2 = np.array(total_pos2)
                total_type = np.array(total_type)
                total_mask = np.array(total_mask)

                feed_dict = {
                    network.input_word: total_word,
                    network.input_pos1: total_pos1,
                    network.input_pos2: total_pos2,
                    network.input_type: total_type,
                    network.input_y: y_batch,
                    network.total_shape: total_shape,
                    network.dropout_keep_prob: FLAGS.dropout_keep_prob,
                    # network.input_mask: total_mask
                }

                _, step, summaries, loss, accuracy= sess.run(
                    [train_op, global_step, train_summary_op, network.final_loss, network.accuracy], feed_dict)
                train_summary_writer.add_summary(summaries, step)
                """
                if step % 100 == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print('{}: step {}, loss {:g}, acc {:g}'.format(time_str, step, loss, accuracy))
                """


            """
            Train epochs
            """
            print('Start training......')
            for epoch in range(FLAGS.num_epochs):
                # Randomly shuffle data
                shuffle_indices = np.random.permutation(np.arange(len(dataset.train_y)))
                num_batches_per_epoch = int((len(dataset.train_y)-1)/settings.batch_size) + 1
                #num_batches_per_epoch = int(len(shuffle_indices)/float(settings.batch_size))

                epoch_last_step = 0
                for batch_num in range(num_batches_per_epoch):
                    start_index = batch_num * settings.batch_size
                    end_index = min((batch_num+1)*settings.batch_size, len(dataset.train_y))
                    if (end_index - start_index) != settings.batch_size:
                        start_index = end_index - settings.batch_size
                    batch_index = shuffle_indices[start_index:end_index]

                    word_batch = dataset.train_word[batch_index]
                    pos1_batch = dataset.train_pos1[batch_index]
                    pos2_batch = dataset.train_pos2[batch_index]
                    type_batch = dataset.train_type[batch_index]
                    mask_batch = dataset.train_mask[batch_index]
                    y_batch = dataset.train_y[batch_index]

                    train_step(word_batch, pos1_batch, pos2_batch, type_batch, y_batch, mask_batch)

                if epoch % FLAGS.checkpoint_every == 0:
                    epoch_last_step = tf.train.global_step(sess, global_step)
                    path = saver.save(sess, checkpoint_prefix, global_step=epoch_last_step)
                    print('Epoch {} batch {} Saved model checkpoint to {}, pattern_num {}'.format(epoch, batch_num, path, FLAGS.pattern_num))

                if FLAGS.allow_evaluation and epoch % FLAGS.evaluate_every == 0:
                    new_highest = evaluator.test()
                    print('Best precision recall area now is {}, progress: {}\n'.format(evaluator.highest_score, utils.calculate_progress(epoch, FLAGS.pattern_num)))
                    if new_highest:
                        utils.copy_model(out_dir, epoch_last_step)
                        utils.store_pr(out_dir, evaluator.highest_score)


            print('final best precision recall: {} pattern_num: {}\n'.format(evaluator.highest_score, FLAGS.pattern_num))



if __name__ == '__main__':
    tf.app.run()
