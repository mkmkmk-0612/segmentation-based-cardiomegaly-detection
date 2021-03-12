
def find_same_point(start, end, cnt, value):
    value_index = start
    for i in range(start, end+1):
        if value==cnt[i][0][1]:
            value_index = i
    
    return value_index

def find_range(cnt_L, cnt_R, b_index_L, b_index_R, index_l, index_r, rightmost_index, leftmost_index):

    if cnt_L[index_l][0][1] > cnt_R[index_r][0][1]:
        bottom = cnt_L[index_l][0]
    else:
        bottom = cnt_R[index_r][0]
        
    min_line = 513

    if cnt_L[rightmost_index][0][1] >= cnt_R[leftmost_index][0][1]:
        line_range_max = cnt_R[leftmost_index][0][1]
    else:
        line_range_max = cnt_L[rightmost_index][0][1]

    if cnt_L[b_index_L][0][1] >= cnt_R[b_index_R][0][1]:
        # print('L > R')
        end_index = find_same_point(0, b_index_R, cnt_R, line_range_max)

        for i in reversed(range(end_index, b_index_R+1)):
            id_cnt_L_index = find_same_point(b_index_L, len(cnt_L)-1, cnt_L, cnt_R[i][0][1])
            line = abs(cnt_R[i][0][0] - cnt_L[id_cnt_L_index][0][0])

            if line <= min_line:
                min_line = line
                min_index_L = id_cnt_L_index
                min_index_R = i

        return cnt_L[min_index_L][0], cnt_R[min_index_R][0], bottom

    else:
        # print('L < R')
        end_index = find_same_point(b_index_L, len(cnt_L)-1, cnt_L, line_range_max) 

        for j in range(b_index_L, end_index+1):
            id_cnt_R_index = find_same_point(0, b_index_R, cnt_R, cnt_L[j][0][1])
            line = abs(cnt_R[id_cnt_R_index][0][0] - cnt_L[j][0][0])

            if line <= min_line:
                min_line = line
                min_index_L = j
                min_index_R = id_cnt_R_index

        return cnt_L[min_index_L][0], cnt_R[min_index_R][0], bottom
   
def find_neck_id(cnt_L, cnt_R, b_index_L, b_index_R, middle):
    max_line = 0 
    
    if cnt_L[b_index_L][0][1] > cnt_R[b_index_R][0][1]:
        for i in range(b_index_R, len(cnt_R)):
            id_cnt_L_index = find_same_point(0, b_index_L, cnt_L, cnt_R[i][0][1])
            line = abs(cnt_R[i][0][0]- cnt_L[id_cnt_L_index][0][0])
            if line > max_line:
                if middle > cnt_R[i][0][1]: # end_range = 55%
                    if i == b_index_R:
                        max_line = line
                        n_id_index_L = id_cnt_L_index
                        n_id_index_R = b_index_R
                    #break
                # print('max_line : ', max_line)
                max_line = line
                n_id_index_L = id_cnt_L_index
                n_id_index_R = i
        print('ID : ', max_line)
        return max_line, n_id_index_L, n_id_index_R

    else:  
        for j in reversed(range(0, b_index_L+1)):
            id_cnt_R_index = find_same_point(b_index_R, len(cnt_R)-1, cnt_R, cnt_L[j][0][1])
            line = abs(cnt_R[id_cnt_R_index][0][0] - cnt_L[j][0][0])
            if line > max_line:
                if middle > cnt_L[j][0][1]: # end_range = 55%
                    if j == b_index_L:
                        max_line = line
                        n_id_index_L_ = b_index_L
                        n_id_index_R_ = id_cnt_R_index 
                    #break
                #print(cnt_L[n_id_index_L_][0])
                max_line = line
                n_id_index_L_ = j 
                n_id_index_R_= id_cnt_R_index
                
        print('ID : ', max_line)
        return max_line, n_id_index_L_, n_id_index_R_
