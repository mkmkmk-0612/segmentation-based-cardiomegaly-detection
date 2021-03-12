import numpy as np

def find_bottom(cnt_L, cnt_R):
    bottom_L = cnt_L[0][0][1]
    bottom_R = cnt_R[0][0][1]
    bottom_L_i = 0
    bottom_R_i = 0
    
    for x in range(1, len(cnt_R)):
        if cnt_R[x][0][1] >= bottom_R:
            bottom_R = cnt_R[x][0][1]
            bottom_R_i = x

    for y in range(1, len(cnt_L)):
        if cnt_L[y][0][1] >= bottom_L:
            bottom_L = cnt_L[y][0][1]
            bottom_L_i = y

    return bottom_L, bottom_R, bottom_L_i, bottom_R_i

def find_top(cnt):
    top = cnt[0][0][1] 
    index = 0
   
    for w in range(1, len(cnt)):
        if cnt[w][0][1] <= top:
            top = cnt[w][0][1]
            index = w 

    return index

def find_max_index(cnt_L, cnt_R):
    right_max = cnt_L[0][0][0]
    left_max = cnt_R[0][0][0]
  
    index_r_m = 0
    index_l_m = 0

    for x in range(len(cnt_L)):
        if right_max <= cnt_L[x][0][0]:
            right_max = cnt_L[x][0][0]
            index_r_m = x 

    for y in range(len(cnt_R)):
        if left_max > cnt_R[y][0][0]:
            left_max = cnt_R[y][0][0]
            index_l_m = y 
 
    return index_r_m, index_l_m
