import numpy as np
import math

"""Find angle point right"""
def angle(cnt, middlemost, r):
    arc_array = []

    for i in range(middlemost, len(cnt)-2*r, 2):

        x = cnt[i][0]
        x_m = cnt[i + r][0]
        x_p = cnt[i + 2*r][0]

        dx_32 = x_p[0] - x_m[0]
        dy_32 = x_p[1] - x_m[1]

        dx_21 = x_m[0] - x[0]
        dy_21 = x_m[1] - x[1]

        dx_31 = x_p[0] - x[0]
        dy_31 = x_p[1] - x[1]

        CB = round(math.sqrt((dx_32 * dx_32) + (dy_32 * dy_32)), 4)
        BA = round(math.sqrt((dx_21 * dx_21) + (dy_21 * dy_21)), 4)
        CA = round(math.sqrt((dx_31 * dx_31) + (dy_31 * dy_31)), 4)
        B = (round(BA * BA) + round(CB * CB) - round(CA * CA)) / (2 * round(BA * CB) + 0.001)
        B_ = math.ceil(math.degrees(math.acos(B)))

        arc_array.append([i-middlemost, B_])

    for j in range(len(arc_array) - 1):
        if arc_array[j][1] != 180:
            if arc_array[j+1][1] != 180:
            
                return arc_array[j+1][0]

"""Find angle point left"""
def angle_(cnt, middlemost, r):
    arc_array_ = []
    i = 0 
    for k in reversed(range(2*r,  middlemost+1)):
        x = cnt[k][0]
        x_m = cnt[k - r][0]
        x_p = cnt[k - 2*r][0]

        dx_32 = x_p[0] - x_m[0]
        dy_32 = x_p[1] - x_m[1]

        dx_21 = x_m[0] - x[0]
        dy_21 = x_m[1] - x[1]

        dx_31 = x_p[0] - x[0]
        dy_31 = x_p[1] - x[1]

        CB = round(math.sqrt((dx_32 * dx_32) + (dy_32 * dy_32)), 4)
        BA = round(math.sqrt((dx_21 * dx_21) + (dy_21 * dy_21)), 4)
        CA = round(math.sqrt((dx_31 * dx_31) + (dy_31 * dy_31)), 4)
        B = (round(BA * BA) + round(CB * CB) - round(CA * CA)) / (2 * round(BA * CB) + 0.001)
        B_ = math.ceil(math.degrees(math.acos(B)))
        
        arc_array_.append([i, B_])
        i += 1

    for p in range(len(arc_array_) - 1):
        if arc_array_[p][1] != 180:
            if arc_array_[p+1][1] != 180:
                return arc_array_[p+1][0]
