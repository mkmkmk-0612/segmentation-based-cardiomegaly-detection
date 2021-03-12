import cv2 as cv
import os 
import shutil

dir_prefix = ''

def abst(lung, center, name):
    for i in range(lung.shape[0]):
        for j in range(lung.shape[1]):
            if center[i][j][0] == 255:
                if center[i][j][0] == lung[i][j][0]:
                    center[i][j][0] = 0 
                    center[i][j][1] = 0 
                    center[i][j][2] = 0 

    cv.imwrite(dir_prefix + 'result_center/' + name + '.png', center)

def differences(name):
    path = dir_prefix +  'result_lung/' + name + '.png'
    path_ = dir_prefix + 'result_center/' + name + '.png'

    for i, filename in enumerate(os.listdir(dir_prefix + 'result_center/')):
        shutil.copy2(path_, dir_prefix + 'original_result_center/' + filename[:-4] + '.png')

    im = cv.imread(path)
    im_ = cv.imread(path_) 
    abst(im, im_, name) 

