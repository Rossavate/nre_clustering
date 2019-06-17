print('reading')

dic = {}
# Read dataset with entity types
with open('train1.txt', 'r', encoding='UTF-8') as f:
    while True:
        content = f.readline()
        if content == '':
            break
        content = content.strip().split('\t')
        # pair = entity1-entity2-relation-sentence
        pair = '{}-{}-{}-{}'.format(content[2], content[3], content[6], content[7])
        types = '{}-{}'.format(content[4], content[5])
        dic[pair] = types


f_write = open('train2.txt', 'w', encoding='UTF-8')
# Read correct dataset
with open('train.txt', 'r', encoding='UTF-8') as f:
    while True:
        content = f.readline()
        if content == '':
            break
        content = content.strip().split()
        sentence = content[5:-1]
        sentence = ' '.join(sentence)
        s = sentence.lower()

        # pari= entity1-entity2-relation-sentence
        pair = '{}-{}-{}-{}'.format(content[2].lower(), content[3].lower(), content[4], s)
        
        if pair not in dic:
            print(pair)
        else:
            types = dic[pair].split('-')
            row = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{} ###END###\n'.format(content[0], content[1], content[2], content[3], types[0], types[1], content[4], sentence)
            f_write.write(row)

f_write.close()