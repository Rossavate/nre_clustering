import numpy as np
from sklearn.metrics import average_precision_score

class Evaluator(object):
    def __init__(self, data_manager, session, network, model, settings, highest_score=0):
        self.data_manager = data_manager
        self.sess = session
        self.network = network
        self.model = model
        self.settings = settings
        self.highest_score = highest_score


    def test(self):
        
        new_highest = False
        """
        Start evaluation
        """

        with open(self.model+'/result.txt', 'a') as f:
            print('Evaluating P@N for one')
            top100, top200, top300, pr = self.__eval_step(self.data_manager.pone_test_word, 
                self.data_manager.pone_test_pos1, 
                self.data_manager.pone_test_pos2, 
                self.data_manager.pone_test_type, 
                self.data_manager.pone_test_y,
                self.data_manager.pone_test_mask,
            )
            f.write('{:g}\t{:g}\t{:g}\t'.format(top100, top200, top300))

            print('Evaluating P@N for two')
            top100, top200, top300, pr = self.__eval_step(self.data_manager.ptwo_test_word, 
                self.data_manager.ptwo_test_pos1, 
                self.data_manager.ptwo_test_pos2, 
                self.data_manager.ptwo_test_type, 
                self.data_manager.ptwo_test_y,
                self.data_manager.ptwo_test_mask,
            )
            f.write('{:g}\t{:g}\t{:g}\t'.format(top100, top200, top300))

            print('Evaluating P@N for all')
            top100, top200, top300, pr = self.__eval_step(self.data_manager.pall_test_word, 
                self.data_manager.pall_test_pos1, 
                self.data_manager.pall_test_pos2, 
                self.data_manager.pall_test_type, 
                self.data_manager.pall_test_y,
                self.data_manager.pall_test_mask,
            )
            f.write('{:g}\t{:g}\t{:g}\t'.format(top100, top200, top300))

            print('Evaluating all test data and save data for PR curve')
            top100, top200, top300, pr = self.__eval_step(self.data_manager.test_word, 
                self.data_manager.test_pos1, 
                self.data_manager.test_pos2, 
                self.data_manager.test_type, 
                self.data_manager.test_y,
                self.data_manager.test_mask,
                save=True)
            f.write('{:g}\n'.format(pr))

            if pr > self.highest_score:
                new_highest = True
                self.highest_score = pr

        return new_highest


    def __eval_op(self, word_batch, pos1_batch, pos2_batch, type_batch, y_batch, mask_batch):
        """
        evaluate a batch
        """
        total_word = []
        total_pos1 = []
        total_pos2 = []
        total_type = []
        total_mask = []
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
                total_mask.append(mask_batch[i][j])

        # Here total_word and y_batch are not equal, total_word[total_shape[i]:total_shape[i+1]] is related to y_batch[i]
        total_shape_batch.append(total_num)

        total_shape_batch = np.array(total_shape_batch)
        total_word = np.array(total_word)
        total_pos1 = np.array(total_pos1)
        total_pos2 = np.array(total_pos2)
        total_type = np.array(total_type)
        total_mask = np.array(total_mask)

        feed_dict = {
            self.network.input_word: total_word,
            self.network.input_pos1: total_pos1,
            self.network.input_pos2: total_pos2,
            self.network.input_type: total_type,
            #self.network.input_y: y_batch,
            self.network.total_shape: total_shape_batch,
            # self.network.input_mask: total_mask,
            self.network.dropout_keep_prob: 1.0
        }

        scores, predictions = self.sess.run([self.network.scores, self.network.predictions], feed_dict)

        return scores, predictions

    
    def __eval_step(self, test_word, test_pos1, test_pos2, test_type, test_y, test_mask, save=False):
        """
        evaluate P@N
        """
        # Metrics that we want
        average_precision = -1
        top100_precision = 0
        top200_precision = 0
        top300_precision = 0

        # Test and get all probability
        allprob = []
        allans = []

        num_batches_per_epoch = int((len(test_y)-1)/self.settings.batch_size) + 1
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * self.settings.batch_size
            end_index = min((batch_num+1)*self.settings.batch_size, len(test_y))
            if (end_index - start_index) != self.settings.batch_size:
                start_index = end_index - self.settings.batch_size

            word_batch = test_word[start_index: end_index]
            pos1_batch = test_pos1[start_index: end_index]
            pos2_batch = test_pos2[start_index: end_index]
            type_batch = test_type[start_index: end_index]
            mask_batch = test_mask[start_index: end_index]
            y_batch = test_y[start_index: end_index]

            scores, predictions = self.__eval_op(word_batch, pos1_batch, pos2_batch, type_batch, y_batch, mask_batch)

            for score in scores:
                allprob.append(score[1:])
            for entity_y in y_batch:
                allans.append(entity_y[1:])
        allprob = np.reshape(np.array(allprob), (-1))
        allans = np.reshape(np.array(allans), (-1))

        # Caculate the pr curve area
        if save == True:
            average_precision = average_precision_score(allans, allprob)
            print('PR curve area: {:g}'.format(average_precision))

            # if this evaluation has higher precision, save it
            if average_precision > self.highest_score:
                np.save('{}/allans_bag.npy'.format(self.model), allans)
                np.save('{}/allprob_bag.npy'.format(self.model), allprob)

        # P@N test
        order = np.argsort(-allprob)
        # P@100
        top100 = order[:100]
        correct_num_100 = 0.0
        for i in top100:
            if allans[i] == 1:
                correct_num_100 += 1.0
        top100_precision = correct_num_100/100
        print('P@100 result: {:g}'.format(top100_precision))
        # P@200
        top200 = order[:200]
        correct_num_200 = 0.0
        for i in top200:
            if allans[i] == 1:
                correct_num_200 += 1.0
        top200_precision = correct_num_200/200
        print('P@200 result: {:g}'.format(top200_precision))
        # P@300
        top300 = order[:300]
        correct_num_300 = 0.0
        for i in top300:
            if allans[i] == 1:
                correct_num_300 += 1.0
        top300_precision = correct_num_300/300
        print('P@300 result: {:g}'.format(top300_precision))

        # Mean
        print('Mean result: {:g}'.format((top100_precision + top200_precision + top300_precision)/3))

        return top100_precision, top200_precision, top300_precision, average_precision