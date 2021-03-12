import cv2 as cv

def mrd(img, name):
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
    
    for x in range(len(cnt)):
        if right_max <= cnt[x][0][0]:
            right_max = cnt[x][0][0]
            right_max_i = x 

        elif left_min > cnt[x][0][0]:
            left_min = cnt[x][0][0]
            left_min_i = x 
    
    rightmost = cnt[right_max_i][0]
    leftmost = cnt[left_min_i][0] 

    return rightmost, leftmost
