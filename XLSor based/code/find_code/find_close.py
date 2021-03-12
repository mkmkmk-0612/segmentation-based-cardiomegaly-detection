from find_code.center_fun import c_f_L
from find_code.center_fun import c_f_R
from find_code.find_id import find_neck_id 
from find_code.lung_lrm import find_max_index
from find_code.find_id import find_range
from find_code.find_angle import angle
from find_code.find_angle import angle_
import cv2 as cv
import math

dir_prefix = ''

"""Find close point in center"""
def center_close(cnt_L, cnt_R, b_index_L, b_index_R, c_point_L, c_point_R, center_neck):
        
        min_r = (c_point_R[0] - cnt_R[0][0][0])**2 + (c_point_R[1] - cnt_R[0][0][1])**2
        index_r = 0
            
        for l in range(0, b_index_R):
            next_d_R = (c_point_R[0] - cnt_R[l][0][0])**2 + (c_point_R[1] - cnt_R[l][0][1])**2
            if min_r >= next_d_R:
                min_r = next_d_R
                index_r = l

        min_l = (c_point_L[0] - cnt_L[0][0][0])**2 + (c_point_L[1] - cnt_L[0][0][1])**2
        index_l = 0

        for v in range(b_index_L, len(cnt_L)):
            next_d_L = (c_point_L[0] - cnt_L[v][0][0])**2 + (c_point_L[1] - cnt_L[v][0][1])**2
            if min_l > next_d_L:
                min_l = next_d_L
                index_l = v

        print('close point : ', cnt_L[index_l][0], cnt_R[index_r][0])

        return index_l, index_r

def close(b_index_L, b_index_R, center_neck, c_point_R, c_point_L, cnt_C, middlemost, i_R, i_L, img, name):
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

            rightmost_index, leftmost_index = find_max_index(cnt_L, cnt_R)
 
            index_l, index_r = center_close(cnt_L, cnt_R, b_index_L, b_index_R, c_point_L, c_point_R, center_neck)
            id_min_L, id_min_R, bottom = find_range(cnt_L, cnt_R, b_index_L, b_index_R, index_l, index_r, rightmost_index, leftmost_index)

            id_middle_range = math.trunc((id_min_L[1] + bottom[1]) * 0.5)
            print('range : ', id_middle_range)
            t_L, MLD_N = c_f_L(center_neck, cnt_L, index_l, rightmost_index, id_middle_range)
            t_R, MRD_N = c_f_R(center_neck, cnt_R, leftmost_index, index_r, id_middle_range)
            print('----------------------------------')

            new_image = cv.imread(dir_prefix + 'center_point/'+ name + '_predict.png')
            while(True):
                if cnt_L[t_L][0][1] < id_middle_range:
                    print('refind_L : ', cnt_L[t_L][0][1])
                    if i_L > 3:
                        print('L_over refind 3 times')
                        break
                    i_L += 1
                    new_c_point_L = angle_(cnt_C, middlemost, 4+i_L)
                    index_l, index_r = center_close(cnt_L, cnt_R, b_index_L, b_index_R, cnt_C[middlemost-new_c_point_L][0], c_point_R, center_neck)
                    id_min_L, id_min_R, bottom = find_range(cnt_L, cnt_R, b_index_L, b_index_R, index_l, index_r, rightmost_index, leftmost_index)
                    id_middle_range = math.trunc((id_min_L[1] + bottom[1]) * 0.5)
                    t_L, MLD_N = c_f_L(center_neck, cnt_L, index_l, rightmost_index, id_middle_range)
                    new_image = cv.circle(new_image, tuple(cnt_C[middlemost-new_c_point_L][0]), 1, (255, 0, 0), -1)
                    print('----------------------------------')

                elif cnt_R[t_R][0][1] < id_middle_range:
                    print('refind_R : ', cnt_R[t_R][0][1])
                    if i_R > 3:
                        print('R_over refind 3 times')
                        break
                    i_R += 1
                    new_c_point_R = angle(cnt_C, middlemost, 4+i_R)
                    index_l, index_r = center_close(cnt_L, cnt_R, b_index_L, b_index_R, c_point_L, cnt_C[middlemost+new_c_point_R][0], center_neck)
                    id_min_L, id_min_R, bottom = find_range(cnt_L, cnt_R, b_index_L, b_index_R, index_l, index_r, rightmost_index, leftmost_index)
                    id_middle_range = math.trunc((id_min_L[1] + bottom[1]) * 0.5)
                    t_R, MRD_N = c_f_R(center_neck, cnt_R, leftmost_index, index_r, id_middle_range)
                    new_image = cv.circle(new_image, tuple(cnt_C[middlemost+new_c_point_R][0]), 1, (255, 0, 0), -1)
                    print('----------------------------------')
                else:
                    break
            
            cv.imwrite(dir_prefix + 'center_point/{}_predict.png'.format(name), new_image)
            neck_id, id_n_l, id_n_r = find_neck_id(cnt_L, cnt_R, b_index_L, b_index_R, id_middle_range)
            file = open(dir_prefix + 'result_txt/' + name+'.txt', 'a')   
            file.write(str(MRD_N) + '\n') # MRD_N
            file.write(str(MLD_N) + '\n') # MLD_N
            file.write(str(neck_id) + '\n') # ID_N
            neck_ratio = (MRD_N + MLD_N)/neck_id
            file.write(str(neck_ratio)) # RATIO_N
            file.close() 
            
            return cnt_L[t_L][0], cnt_R[t_R][0], cnt_L[id_n_l][0], cnt_R[id_n_r][0], neck_ratio 
