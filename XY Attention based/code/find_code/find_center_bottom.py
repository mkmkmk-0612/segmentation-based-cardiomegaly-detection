import numpy as np

"""Find at least 2 same x point. If find 2 points, Y point is the bottommost point""" 
def find_center_middle(cnt, middle_x):
    middle_x_array = []
    x = int(middle_x)
    for i in range(0, len(cnt)):
        if cnt[i][0][0] == x:
            middle_x_array.append(i)

    max = 0
    index = 0

    for j in range(len(middle_x_array)):
        if cnt[middle_x_array[j]][0][1] > max:
            max = cnt[middle_x_array[j]][0][1]
            index = j

    return middle_x_array[index]
