import numpy as np
import init_data

class DataManager(object):
    def __init__(self, init_data=False, sentence_length=80, position_length=60):
        self.sentence_length = sentence_length
        self.position_length = position_length

        if init_data == True:
            initializer = init_data.DataInitializer()
            initializer.get_Binary_Data()

        # Load data
        self.__load_wordembedding()
        self.__load_train_data()
        self.__load_test_data()

    
    def __load_wordembedding(self):
        """
        Load wordembedding
        """
        self.wordembedding = np.load('../../data/bag_data/vec.npy')

    
    def __load_train_data(self):
        """
        Load train data
        """
        self.train_word = np.load('../../data/bag_data/train_word.npy')
        self.train_pos1 = np.load('../../data/bag_data/train_pos1.npy')
        self.train_pos2 = np.load('../../data/bag_data/train_pos2.npy')
        self.train_type = np.load('../../data/bag_data/train_type.npy')
        self.train_mask = np.load('../../data/bag_data/train_mask.npy')
        self.train_y = np.load('../../data/bag_data/train_y.npy')


    def __load_test_data(self):
        """
        Load test data
        """
        self.pone_test_word = np.load('../../data/bag_data/pone_test_word.npy')
        self.pone_test_pos1 = np.load('../../data/bag_data/pone_test_pos1.npy')
        self.pone_test_pos2 = np.load('../../data/bag_data/pone_test_pos2.npy')
        self.pone_test_type = np.load('../../data/bag_data/pone_test_type.npy')
        self.pone_test_mask = np.load('../../data/bag_data/pone_test_mask.npy')
        self.pone_test_y = np.load('../../data/bag_data/pone_test_y.npy')

        self.ptwo_test_word = np.load('../../data/bag_data/ptwo_test_word.npy')
        self.ptwo_test_pos1 = np.load('../../data/bag_data/ptwo_test_pos1.npy')
        self.ptwo_test_pos2 = np.load('../../data/bag_data/ptwo_test_pos2.npy')
        self.ptwo_test_type = np.load('../../data/bag_data/ptwo_test_type.npy')
        self.ptwo_test_mask = np.load('../../data/bag_data/ptwo_test_mask.npy')
        self.ptwo_test_y = np.load('../../data/bag_data/ptwo_test_y.npy')

        self.pall_test_word = np.load('../../data/bag_data/pall_test_word.npy')
        self.pall_test_pos1 = np.load('../../data/bag_data/pall_test_pos1.npy')
        self.pall_test_pos2 = np.load('../../data/bag_data/pall_test_pos2.npy')
        self.pall_test_type = np.load('../../data/bag_data/pall_test_type.npy')
        self.pall_test_mask = np.load('../../data/bag_data/pall_test_mask.npy')
        self.pall_test_y = np.load('../../data/bag_data/pall_test_y.npy')

        self.test_word = np.load('../../data/bag_data/test_word.npy')
        self.test_pos1 = np.load('../../data/bag_data/test_pos1.npy')
        self.test_pos2 = np.load('../../data/bag_data/test_pos2.npy')
        self.test_type = np.load('../../data/bag_data/test_type.npy')
        self.test_y = np.load('../../data/bag_data/test_y.npy')
        self.test_mask = np.load('../../data/bag_data/test_mask.npy')
