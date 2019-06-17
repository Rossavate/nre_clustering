import os
import shutil
import numpy as np

def copy_model(out_dir, step):
    """
    Copy model file to outer dir
    """
    src_file1 = out_dir + '/checkpoints/model-' + str(step) + '.data-00000-of-00001'
    src_file2 = out_dir + '/checkpoints/model-' + str(step) + '.index'
    src_file3 = out_dir + '/checkpoints/model-' + str(step) + '.meta'

    dest_file1 = out_dir + '/model-best.data-00000-of-00001'
    dest_file2 = out_dir + '/model-best.index'
    dest_file3 = out_dir + '/model-best.meta'

    copy_file(src_file1, dest_file1)
    copy_file(src_file2, dest_file2)
    copy_file(src_file3, dest_file3)

def copy_file(src, dest):
    if not os.path.isfile(src):
        print('{} not exists!'.format(src))
    else:
        # Split path and filename
        fpath,fname=os.path.split(dest)
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        # Copy file
        shutil.copyfile(src,dest)


def store_pr(directory, value):
    file = directory + '/pr.npy'
    if not os.path.isfile(file):
        pr_list = []
    else:
        pr_list = list(np.load(file))
    pr_list.append(value)
    np.save(file, np.array(pr_list))

def read_pr(directory):
    file = directory + '/pr.npy'
    if not os.path.isfile(file):
        return 0.0
    else:
        pr_list = np.load(file)
        return pr_list[-1]

def calculate_progress(epoch, pattern_num):
    total = 6 * 10
    current = ((pattern_num//5)-1)*10 + epoch + 1
    return '{:.2f}'.format(current/total)