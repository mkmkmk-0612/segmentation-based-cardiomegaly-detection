# MLD
def c_f_L(center_neck, cnt, start, end, id_middle_range):
    print('L_origin start : ', cnt[start][0])

    d_l = abs(center_neck[0] - cnt[start][0][0])
    max = d_l
    index_n_l = start
    count = 0
    for i in range(start+1, end+1):
        d_l = abs(center_neck[0] - cnt[i][0][0])   

        if max <= d_l:
            if cnt[i][0][1] < id_middle_range:
                break
            if max == d_l:
                count += 1 
            else:
                count = 0 
            max = d_l
            index_n_l = i - int(count/2)
    return index_n_l, max

# MRD
def c_f_R(center_neck, cnt, start, end, id_middle_range):
    print('R_origin start : ', cnt[end][0])
    new_start = column_pixel(center_neck, cnt, start, end)

    if new_start != None:
        end = new_start

    d_r = abs(center_neck[0] - cnt[end][0][0])
    max = d_r
    index_n_r = end
    count = 0
    for j in reversed(range(start+1, end+1)):
        d_r = abs(center_neck[0] - cnt[j][0][0])   

        if max <= d_r:
            if cnt[j][0][1] < id_middle_range:
                break
            if max == d_r:
                count += 1
            else:
                count = 0
            max = d_r
            index_n_r = j + int(count/2)
    return index_n_r, max

def reculsion_R(cnt, i, s):
    if i == cnt[s-1][0][0]:
         count = 1 + reculsion_R(cnt, i, s-1) 
         return count
    else:
        return 1 

def reculsion_L(cnt, i, s):
    if i == cnt[s+1][0][0]:
        count = 1 + reculsion_L(cnt, i, s+1)
        return count
    else:
        return 1

def column_pixel(center_neck, cnt, start, end): 
    if cnt[start][0][1] > cnt[end][0][1]:
        f_count_index = start

        for i in range(cnt[start][0][0], cnt[start][0][0]+21):
            s = start 
            e = end 
            f_count = 0

            while(True):
                if s > e:
                    break
                jump = 1
                if cnt[s][0][0] == i:
                    if f_count == 0:
                        f_count_index = s
                    s_point = reculsion_L(cnt, i, s)
                     
                    if s_point < 25:
                        f_count += 1
                        jump = s_point

                    if s_point > 25:
                        f_count += 1
                        jump = s_point - 25
                        
                s += jump

            if f_count == 2:
                return f_count_index

    else:
        f_count_index = end

        for j in reversed(range(cnt[end][0][0]-20, cnt[end][0][0]+1)):
            s = end 
            e = start
            f_count = 0
            while(True):
                if e > s:
                    break
                jump = 1
                if cnt[s][0][0] == j:
                    if f_count == 0:
                        f_count_index = s
                    s_point = reculsion_R(cnt, j, s)

                    if s_point < 25:
                        f_count += 1
                        jump = s_point

                    if s_point > 25:
                        f_count += 1
                        jump = s_point - 25

                s -=  jump

            if f_count == 2:
                return f_count_index
