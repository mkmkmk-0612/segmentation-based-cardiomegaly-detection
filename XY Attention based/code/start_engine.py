import os
import time
import cv2
from inferences import resize
from test import start

dir_prefix = ''

def make_txt():
    path = dir_prefix + 'data/test/'
    print(path)
    file_list = os.listdir(path)
    f = open(dir_prefix + 'data/list.txt', 'a')
 
    for i in range(0, len(file_list)):
        print(str(i+1), file_list[i])
        f.write('test/' + file_list[i] + '\n')

def preprocess():
    path = dir_prefix + 'data/test/'

    for k, filename in enumerate(os.listdir(path)):
        img = cv2.imread(path+filename)
        cv2.imwrite(path+filename[:-4]+'.png', img)
    
if __name__ == '__main__':
    #start_time = time.time()
    preprocess()
    resize()
    make_txt() 
    start()
    #print('time : ', time.time() - start_time)

