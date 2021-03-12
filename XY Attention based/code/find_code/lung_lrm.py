
def find_max_index(cnt_L, cnt_R):
    right_max = cnt_L[0][0][0]
    left_max = cnt_R[0][0][0]

    for u in range(len(cnt_L)):
        if right_max <= cnt_L[u][0][0]:
            right_max = cnt_L[u][0][0]
            index_r_m = u
    for l in range(len(cnt_R)):
        if left_max > cnt_R[l][0][0]:
            left_max = cnt_R[l][0][0]
            index_l_m = l

    return index_r_m, index_l_m
