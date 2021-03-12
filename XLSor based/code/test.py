import os
import argparse
import numpy as np
import torch.nn as nn
import cv2
import torch
from find_code.find_differences import differences
from inferences import lung_seg
from inferences import center_seg
from torch.utils import data
from networks.xlsor import XLSor
from dataset.datasets import XRAYDataTestSet
from PIL import Image as PILImage

dir_prefix = ''
IMG_MEAN = np.array((104.00698793,11.668767626,122.67891434), dtype=np.float32)
NUM_CLASSES = 2

DATA_DIRECTORY = dir_prefix + 'data/'
DATA_LIST_PATH = dir_prefix + 'data/list.txt'
RESTORE_FROM = dir_prefix + 'models/'

def get_arguments():
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=str, default='0',
                        help="choose gpu device.")
    parser.add_argument("--recurrence", type=int, default=2,
                        help="choose the number of recurrence.")
    return parser.parse_args() 

def main():
    args = get_arguments()

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    model = XLSor(num_classes=args.num_classes)
    
    saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict)

    model.eval()
    model.cuda()
    testloader = data.DataLoader(XRAYDataTestSet(args.data_dir, args.data_list, crop_size=(512, 512), scale=False, mirror=False), batch_size=1, shuffle=False, pin_memory=True)
    interp = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=True)

    if not os.path.exists(dir_prefix + 'result_lung'):
        os.makedirs(dir_prefix + 'result_lung')

    if not os.path.exists(dir_prefix + 'result_center'):
        os.makedirs(dir_prefix + 'result_center')

    for index, batch in enumerate(testloader):
        image, size, name = batch

        with torch.no_grad():
            prediction = model(image.cuda(), args.recurrence)

            if isinstance(prediction, list):
                prediction = prediction[0]
            prediction = interp(prediction).cpu().data[0].numpy().transpose(1, 2, 0)

        print(index+1, os.path.basename(name[0][:-4])) 

        prediction[:,:,0] = np.where(prediction[:,:,0]<=0.5,0,prediction[:,:,0])
        prediction[:,:,0] = np.where(prediction[:,:,0]>1,1,prediction[:,:,0])
        prediction[:,:,0] = np.where(prediction[:,:,0]>0.5,1,prediction[:,:,0])
        output_im = PILImage.fromarray((prediction[:,:,0]* 255).astype(np.uint8))
        output_im.save(dir_prefix + 'result_lung/' + os.path.basename(name[0][:-4])+ '.png')

        prediction[:,:,1] = np.where(prediction[:,:,1]<=0.5,0,prediction[:,:,1])
        prediction[:,:,1] = np.where(prediction[:,:,1]>1,1,prediction[:,:,1])
        prediction[:,:,1] = np.where(prediction[:,:,1]>0.5,1,prediction[:,:,1])
        output_im = PILImage.fromarray((prediction[:,:,1]* 255).astype(np.uint8))
        output_im.save(dir_prefix + 'result_center/' + os.path.basename(name[0][:-4]) +'.png')

        differences(os.path.basename(name[0][:-4]))
        point = cv2.imread(dir_prefix + 'result_center/' + os.path.basename(name[0]))

        try:
            center_seg(point, os.path.basename(name[0][:-4]))
        except TypeError:
            continue
        except IndexError:
            print('center_seg_error')
            continue
        
def start():
    main()
