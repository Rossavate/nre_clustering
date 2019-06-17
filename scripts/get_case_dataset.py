print('reading')

def get_data():
    bags = set()
    bags.add('asia-russia') #15
    bags.add('california-berkeley')#13
    bags.add('mel_karmazin-sirius_satellite_radio')#5

    f_zore = open('test_zore.txt', 'w')
    f_nre = open('test_nre.txt', 'w')

    with open('test1.txt', 'r') as f:
        while True:
            content = f.readline()
            if content == '':
                break
            contents = content.strip().split('\t')

            bag = '{}-{}'.format(contents[2], contents[3])

            if bag in bags:
                f_zore.write(content)
                f_nre.write('{}\t{}\t{}\t{}\t{}\t{}\t###END###\n'.format(contents[0], contents[1], contents[2], contents[3], contents[6], contents[7]))

def count():
    dic = {}
    with open('test1.txt', 'r') as f:
        while True:
            content = f.readline()
            if content == '':
                break
            content = content.strip().split('\t')


            bag = '{}-{}'.format(content[2], content[3])
            
            if bag not in dic:
                dic[bag] = 0
            dic[bag] += 1

    for key in dic:
        if dic[key] >= 10 and dic[key] <= 15:
            print('%s:  %d' %(key, dic[key]))


if __name__ == '__main__':
    #count()
    get_data()