import numpy as np

class DataInitializer(object):
    def __init__(self, sentence_length=80, position_length=60):
        self.sentence_length = sentence_length
        self.position_length = position_length


    def get_Binary_Data(self):
        """
        Transform dataset from txt to binary data
        """

        print('reading word embedding data...')
        # a map, key is word, value is its id
        word2id = {}
        vec = []

        with open('../../data/origin_data/vec.txt', 'r') as f:
            f.readline()

            while True:
                content = f.readline()
                if content == '':
                    break
                content = content.strip().split()
                word2id[content[0]] = len(word2id)
                content = content[1:]
                content = [(float)(i) for i in content]
                vec.append(content)

        dim = 50

        word2id['UNK'] = len(word2id)
        word2id['BLANK'] = len(word2id)
        vec.append(np.random.normal(size=dim,loc=0,scale=0.05))
        vec.append(np.random.normal(size=dim,loc=0,scale=0.05))
        vec = np.array(vec,dtype=np.float32)
        np.save('../../data/bag_data/vec.npy',vec)

        """
        Read relation to id
        """

        print('reading relation to id')
        relation2id = {}

        with open('../../data/origin_data/relation2id.txt','r') as f:
            while True:
                content = f.readline()
                if content == '':
                    break
                content = content.strip().split()
                relation2id[content[0]] = int(content[1])


        """
        Read type to id
        """

        print("reading type to id")
        type2id = {}

        with open('../../data/origin_data/type2id.txt', 'r') as f:
            while True:
                content = f.readline()
                if content == '':
                    break
                print(content)
                content = content.strip().split()
                type2id[content[0]] = int(content[1])


        """
        reading train and test data
        """

        # length of sentence is 80
        fixlen = self.sentence_length
        # max length of position embedding is self.position_length (-60~+60)
        maxlen = self.position_length

        print('reading train data ...')
        train_sen = {} # {entity pair:[[[label1-sentence 1],[label1-sentence 2]...],[[label2-sentence 1],[label2-sentence 2]...]]}
        train_ans = {} # {entity pair:[label1,label2,...]} the label is one-hot vector
        with open('../../data/origin_data/train.txt', 'r') as f:
            while True:
                content = f.readline()
                if content == '':
                    break
                
                content = content.strip().split()
                # get entity name
                en1 = content[2] 
                en2 = content[3]
                relation = 0
                if content[6] not in relation2id:
                    relation = relation2id['NA']
                else:
                    relation = relation2id[content[6]]
                # put the same entity pair sentences into a dict
                tup = (en1,en2)
                # entity pair's label index
                label_tag = 0
                if tup not in train_sen:
                    train_sen[tup]=[]
                    train_sen[tup].append([])
                    y_id = relation
                    label_tag = 0
                    label = [0 for i in range(len(relation2id))]
                    label[y_id] = 1
                    train_ans[tup] = []
                    train_ans[tup].append(label)
                else:
                    y_id = relation
                    label_tag = 0
                    label = [0 for i in range(len(relation2id))]
                    label[y_id] = 1
                    
                    temp = self.find_index(label,train_ans[tup])
                    # if relation 'label' did not occur in entity pair 'tup' before
                    if temp == -1:
                        train_ans[tup].append(label)
                        label_tag = len(train_ans[tup])-1
                        train_sen[tup].append([])
                    else:
                        label_tag = temp

                # so the sentence should be put in entity 'tup' and index 'label_tag'
                sentence = content[7:-1]
                # entits' positions
                en1pos = 0
                en2pos = 0
                
                for i in range(len(sentence)):
                    if sentence[i] == en1:
                        en1pos = i
                    if sentence[i] == en2:
                        en2pos = i
                output = []
                for i in range(fixlen):
                    word = word2id['BLANK']
                    rel_e1 = self.pos_embed(i - en1pos)
                    rel_e2 = self.pos_embed(i - en2pos)
                    entity_type = type2id['NA']
                    if i >= len(sentence):
                        mask = 0
                    elif i - en1pos <= 0:
                        mask = 1
                    elif i - en2pos <= 0:
                        mask = 2
                    else:
                        mask = 3
                    output.append([word,rel_e1,rel_e2,entity_type,mask])

                for i in range(min(fixlen,len(sentence))):
                    word = 0
                    if sentence[i] not in word2id:
                        word = word2id['UNK']
                    else:
                        word = word2id[sentence[i]]
                    
                    output[i][0] = word
                # entity type
                if en1pos < fixlen:
                    output[en1pos][3] = type2id[content[4]]
                if en2pos < fixlen:
                    output[en2pos][3] = type2id[content[5]]
                # add a sentence to entity 'tup' and its label index is 'label_tag'
                train_sen[tup][label_tag].append(output)


        print('reading test data ...')
        test_sen = {} # {entity pair:[[sentence 1],[sentence 2]...]} don't care about label
        test_ans = {} # {entity pair:[labels,...]} the labels is N-hot vector (N is the number of multi-label), so may be multi-ones
        test_mask = []

        with open('../../data/origin_data/test.txt', 'r') as f:
            while True:
                content = f.readline()
                if content == '':
                    break
                
                content = content.strip().split()
                en1 = content[2]
                en2 = content[3]
                relation = 0
                if content[6] not in relation2id:
                    relation = relation2id['NA']
                else:
                    relation = relation2id[content[6]]        
                tup = (en1,en2)
                
                if tup not in test_sen:
                    test_sen[tup]=[]
                    y_id = relation
                    label = [0 for i in range(len(relation2id))]
                    label[y_id] = 1
                    test_ans[tup] = label
                else:
                    y_id = relation
                    test_ans[tup][y_id] = 1
                    
                sentence = content[7:-1]

                en1pos = 0
                en2pos = 0
                
                for i in range(len(sentence)):
                    if sentence[i] == en1:
                        en1pos = i
                    if sentence[i] == en2:
                        en2pos = i
                output = []

                tmp = []
                for i in range(fixlen):
                    word = word2id['BLANK']
                    rel_e1 = self.pos_embed(i - en1pos)
                    rel_e2 = self.pos_embed(i - en2pos)
                    entity_type = type2id['NA']
                    if i >= len(sentence):
                        mask = 0
                    elif i - en1pos <= 0:
                        mask = 1
                    elif i - en2pos <= 0:
                        mask = 2
                    else:
                        mask = 3
                    output.append([word,rel_e1,rel_e2,entity_type,mask])

                for i in range(min(fixlen,len(sentence))):
                    word = 0
                    if sentence[i] not in word2id:
                        word = word2id['UNK']
                    else:
                        word = word2id[sentence[i]]

                    output[i][0] = word
                # entity type
                if en1pos < fixlen:
                    output[en1pos][3] = type2id[content[4]]
                if en2pos < fixlen:
                    output[en2pos][3] = type2id[content[5]]

                test_sen[tup].append(output)


        """
        organizing train and test data
        """
        print('organizing train data...')
        train_x = []
        train_y = []

        for i in train_sen:
            if len(train_ans[i]) != len(train_sen[i]):
                print('length of label of entity i not equal')
            lenth = len(train_ans[i])
            for j in range(lenth):
                train_x.append(train_sen[i][j])  # train_sen[i][j] is a list of sentences with entity i and label j
                train_y.append(train_ans[i][j])  # train_ans[i][j] is one-hot label j of entity i 

        print('organizing test data...')
        test_x = []
        test_y = []

        for i in test_sen:        
            test_x.append(test_sen[i]) # test_sen[i] is a list of sentences about entity i
            test_y.append(test_ans[i]) # test_ans[i] is N-hot lable of label i


        train_x = np.array(train_x)
        train_y = np.array(train_y)
        test_x = np.array(test_x)
        test_y = np.array(test_y)

        """
        get test data for P@N evaluation, in which only entity pairs with more than 1 sentence exist
        """
        print('get test data for p@n test')
        pone_test_x = []
        pone_test_y = []

        ptwo_test_x = []
        ptwo_test_y = []

        pall_test_x = []
        pall_test_y = []

        for i in range(len(test_x)):
            if len(test_x[i]) > 1:

                pall_test_x.append(test_x[i])
                pall_test_y.append(test_y[i])

                onetest = []
                temp = np.random.randint(len(test_x[i]))
                onetest.append(test_x[i][temp])
                pone_test_x.append(onetest)
                pone_test_y.append(test_y[i])

                twotest = []
                temp1 = np.random.randint(len(test_x[i]))
                temp2 = np.random.randint(len(test_x[i]))
                while temp1 == temp2:
                    temp2 = np.random.randint(len(test_x[i]))
                twotest.append(test_x[i][temp1])
                twotest.append(test_x[i][temp2])
                ptwo_test_x.append(twotest)
                ptwo_test_y.append(test_y[i])

        pone_test_x = np.array(pone_test_x)
        pone_test_y = np.array(pone_test_y)
        ptwo_test_x = np.array(ptwo_test_x)
        ptwo_test_y = np.array(ptwo_test_y)    
        pall_test_x = np.array(pall_test_x)
        pall_test_y = np.array(pall_test_y)

        """
        Seperate word and positions, type
        """
        print('seperating training data')

        x_train = train_x

        train_word = []
        train_pos1 = []
        train_pos2 = []
        train_type = []
        train_mask = []

        print('seprating train data...')
        for i in range(len(x_train)):
            word = []
            pos1 = []
            pos2 = []
            types = []
            mask = []
            for j in x_train[i]:   # j is a sentence
                temp_word = []
                temp_pos1 = []
                temp_pos2 = []
                temp_type = []
                temp_mask = []
                for k in j:
                    temp_word.append(k[0])
                    temp_pos1.append(k[1])
                    temp_pos2.append(k[2])
                    temp_type.append(k[3])
                    temp_mask.append(k[4])
                word.append(temp_word)
                pos1.append(temp_pos1)
                pos2.append(temp_pos2)
                types.append(temp_type)
                mask.append(temp_mask)
            train_word.append(word)
            train_pos1.append(pos1)
            train_pos2.append(pos2)
            train_type.append(types)
            train_mask.append(mask)

        train_word = np.array(train_word)
        train_pos1 = np.array(train_pos1)
        train_pos2 = np.array(train_pos2)
        train_type = np.array(train_type)
        train_mask = np.array(train_mask)

        """
        slice some big batch in train data into small batches in case of running out of memory
        """
        
        print('smalling training data...')

        word = train_word
        pos1 = train_pos1
        pos2 = train_pos2
        types = train_type
        y = train_y
        mask = train_mask

        new_word = []
        new_pos1 = []
        new_pos2 = []
        new_type = []
        new_y = []
        new_mask = []

        print('get small training data')

        # word[i] is a list whose elements are a list of sentences related to entity pair i with label train_y[i]
        # but entity pair i may contains too many sentences, so slice them
        for i in range(len(word)):
            # think about entity pair i
            length = len(word[i])
            slice_size = 1000
            num_sum = int((length-1)/slice_size) + 1
            for num in range(num_sum):
                start_index = num*slice_size
                end_index = min((num+1)*slice_size, length)

                new_word.append(word[i][start_index:end_index])
                new_pos1.append(pos1[i][start_index:end_index])
                new_pos2.append(pos2[i][start_index:end_index])
                new_type.append(types[i][start_index:end_index])
                new_mask.append(mask[i][start_index:end_index])
                new_y.append(y[i])


        new_word = np.array(new_word)
        new_pos1 = np.array(new_pos1)
        new_pos2 = np.array(new_pos2)
        new_type = np.array(new_type)
        new_mask = np.array(new_mask)
        new_y = np.array(new_y)

        np.save('../../data/bag_data/train_word.npy',new_word)
        np.save('../../data/bag_data/train_pos1.npy',new_pos1)
        np.save('../../data/bag_data/train_pos2.npy',new_pos2)
        np.save('../../data/bag_data/train_type.npy',new_type)
        np.save('../../data/bag_data/train_mask.npy',new_mask)
        np.save('../../data/bag_data/train_y.npy',new_y)


        print('seperating test data...')

        print('seperating p-one test data')
        self.seperate_test_data(pone_test_x, pone_test_y, 'pone_test')

        print('seperating p-two test data')
        self.seperate_test_data(ptwo_test_x, ptwo_test_y, 'ptwo_test')

        print('seperating p-all test data')
        self.seperate_test_data(pall_test_x, pall_test_y, 'pall_test')

        print('seperating test all data')
        self.seperate_test_data(test_x, test_y, 'test')


    def pos_embed(self, x):
        """
        get position embedding of x
        """

        if x < -(self.position_length):
            return 0
        elif x >= -(self.position_length) and x <= self.position_length:
            return x+self.position_length+1
        elif x > self.position_length:
            return 2*(self.position_length+1)


    def find_index(self, x,y):
        """
        find the index of x in y, if x not in y, return -1
        """
        flag = -1
        for i in range(len(y)):
            if x != y[i]:
                continue
            else:
                return i
        return flag


    def seperate_test_data(self, x_test, y_test, name):
        """
        seperating word, position, type for test data
        """
        test_word = []
        test_pos1 = []
        test_pos2 = []
        test_type = []
        test_mask = []

        for i in range(len(x_test)):
            word = []
            pos1 = []
            pos2 = []
            types = []
            mask = []
            for j in x_test[i]:
                temp_word = []
                temp_pos1 = []
                temp_pos2 = []
                temp_type = []
                temp_mask = []
                for k in j:
                    temp_word.append(k[0])
                    temp_pos1.append(k[1])
                    temp_pos2.append(k[2])
                    temp_type.append(k[3])
                    temp_mask.append(k[4])
                word.append(temp_word)
                pos1.append(temp_pos1)
                pos2.append(temp_pos2)
                types.append(temp_type)
                mask.append(temp_mask)
            test_word.append(word)
            test_pos1.append(pos1)
            test_pos2.append(pos2)
            test_type.append(types)
            test_mask.append(mask)



        test_word = np.array(test_word)
        test_pos1 = np.array(test_pos1)
        test_pos2 = np.array(test_pos2)
        test_type = np.array(test_type)
        test_mask = np.array(test_mask)

        np.save('../../data/bag_data/{}_word.npy'.format(name),test_word)
        np.save('../../data/bag_data/{}_pos1.npy'.format(name),test_pos1)
        np.save('../../data/bag_data/{}_pos2.npy'.format(name),test_pos2)
        np.save('../../data/bag_data/{}_type.npy'.format(name),test_type)
        np.save('../../data/bag_data/{}_mask.npy'.format(name),test_mask)
        np.save('../../data/bag_data/{}_y.npy'.format(name),y_test)


if __name__ == '__main__':
    initializer = DataInitializer()
    initializer.get_Binary_Data()
