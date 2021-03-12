from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from skimage import morphology, color, io, exposure
from find_code.find_center_bottom import find_center_middle
from find_code.find_close import close
from find_code.find_R_L import find_bottom
from find_code.find_R_L import find_max_index 
from find_code.find_angle import angle
from find_code.find_angle import angle_
from find_code.find_mrd import mrd
import os
import math
import torch
import torch.nn.functional as F    
import numpy as np
import cv2 as cv

os.environ["CUDA_VISIBLE_DEVICES"]="0"
dir_prefix = ''

def resize():
    path = dir_prefix + 'data/test/'
    
    if not (os.path.exists(dir_prefix + 'resize_input_images/')):
        os.mkdir(dir_prefix + 'resize_input_images/')
        
    for i, filename in enumerate(os.listdir(path)):
        extension = os.path.splitext(filename)[1]
        if extension == '.png':
            img = io.imread(path+filename[:-4]+'.png')
            img = cv.resize(img, dsize=(512,512), interpolation=cv.INTER_AREA)
            cv.imwrite(dir_prefix + 'resize_input_images/'+filename[:-4]+'.png',img)
        else:
            print('EXTENSION ERROR')

"""Find lowest point, leftmost, rightmost in Lung image"""
def lung_seg(img):
    point_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thr = cv.threshold(point_gray, 127, 255, 0)
    _, contours, _ = cv.findContours(thr, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    array = [] 

    for i in range(len(contours)):
       array.append([i, len(contours[i])])
   
    array.sort(key=lambda x: x[1])

    if len(contours) >= 2:
        cnt_R = contours[array[-1:][0][0]]
        cnt_L = contours[array[-2:-1][0][0]]
       
        x = cnt_L[0][0][0]
        y = cnt_R[0][0][0]
 
        if x > y:
            swap = []
            swap = cnt_L
            cnt_L = cnt_R
            cnt_R = swap

        rightmost_i, leftmost_i = find_max_index(cnt_L, cnt_R)
        
        rightmost_p = cnt_L[rightmost_i][0]
        leftmost_p = cnt_R[leftmost_i][0]

        bottom_L, bottom_R, bottom_L_i, bottom_R_i = find_bottom(cnt_L, cnt_R)
        
        center_neck = [0, 0]
        center_neck[0] = int((rightmost_p[0]+ leftmost_p[0])/2)
        center_neck[1] = int((rightmost_p[1]+ leftmost_p[1])/2)

        return bottom_L, bottom_R, bottom_L_i, bottom_R_i, center_neck, cnt_L, cnt_R

    else:
        print("Lung contour error")

"""Predict middle, right_max, left_min point in center data"""
def center_seg(img, name):
    
    point_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thr = cv.threshold(point_gray, 127, 255, 0)
    _, contours, _ = cv.findContours(thr, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    array = []
   
    for i in range(len(contours)):
       array.append([i, len(contours[i])])

    array.sort(key=lambda x: x[1])
    cnt = contours[array[-1:][0][0]]
    
    right_max = cnt[0][0][0] 
    left_min = cnt[0][0][0]   

    right_max_i = 0
    left_min_i = 0
    
    for x in range(len(cnt)):
        if right_max <= cnt[x][0][0]:
            right_max = cnt[x][0][0]
            right_max_i = x 

        elif left_min > cnt[x][0][0]:
            left_min = cnt[x][0][0]
            left_min_i = x 
    
    rightmost = cnt[right_max_i][0]
    leftmost = cnt[left_min_i][0] 
    middle_x = math.trunc((rightmost[0] + leftmost[0]) / 2)
    middlemost = find_center_middle(cnt, middle_x)

    a = angle(cnt, middlemost, 4) # R
    a_ = angle_(cnt, middlemost, 4) # L
    i_R = 0 
    i_L = 0 

    while(True):
         if a == None:
             if i_R > 3:
                 break
             i_R += 1
             #print('R_angle_refind : ', 4+i_R)
             a = angle(cnt, middlemost, 4+i_R)
         elif a_ == None:
             if i_L > 3:
                 break
             i_L += 1
             #print('L_angle_refind : ', 4+i_L)
             a_ = angle_(cnt, middlemost, 4+i_L)
         else:
             break 

    img = cv.circle(img, tuple(cnt[right_max_i][0]), 1, (0, 0, 255), -1)
    img = cv.circle(img, tuple(cnt[left_min_i][0]), 1, (0, 0, 255), -1)

    cv.imwrite(dir_prefix + 'center_point/{}_predict.png'.format(name), img)

    o_image = cv.imread(dir_prefix + 'result_lung/' + name + '.png')
    neck_image = cv.imread(dir_prefix + 'resize_input_images/' + name + '.png')
    original_center = cv.imread(dir_prefix + 'original_result_center/' + name + '.png')  

    n_mrd, n_mld = mrd(original_center, name)
    bottom_L, bottom_R, bottom_L_i, bottom_R_i, center_neck, cnt_L, cnt_R = lung_seg(o_image)
    f_n_l, f_n_r, id_n_l, id_n_r, neck_ratio = close(bottom_L_i, bottom_R_i, center_neck, cnt[right_max_i][0], cnt[left_min_i][0], cnt, middlemost, i_R, i_L, o_image, name)

    neck_image = cv.line(neck_image, tuple(id_n_l), tuple(id_n_r), (0, 215, 255), 2) #ID
    neck_image = cv.line(neck_image, (center_neck[0], 0), (center_neck[0], 512), (255, 144, 30), 2) #Center
    neck_image = cv.line(neck_image, tuple(f_n_l), (center_neck[0], f_n_l[1]), (60, 0, 220), 2) #MLD
    neck_image = cv.line(neck_image, tuple(f_n_r), (center_neck[0], f_n_r[1]), (60, 0, 220), 2) #MRD
    
    cv.imwrite(dir_prefix + 'result_neck/{}_neck.png'.format(name), neck_image)
