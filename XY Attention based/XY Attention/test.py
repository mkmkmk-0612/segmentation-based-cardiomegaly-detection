import argparse
import numpy as np
import torch
from torch.utils import data
from networks.xlsor import XLSor
from networks.unet import unet
from dataset.datasets import XRAYDataTestSet
from backboned_unet import Unet
import os
from crf import dense_crf
from PIL import Image as PILImage
import torch.nn as nn
from PIL import Image
from torchvision.utils import save_image

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
NUM_CLASSES = 1

DATA_DIRECTORY = './data/'
DATA_LIST_PATH = './data/test.txt'
RESTORE_FROM = './'

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
    parser.add_argument("--gpu", type=str, default='1',
                        help="choose gpu device.")
    parser.add_argument("--recurrence", type=int, default=2,
                        help="choose the number of recurrence.")
    return parser.parse_args()


def main():
    args = get_arguments()

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    model = unet()
    c = 0
    
    saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict)

    model.eval()
    model.cuda()
    testloader = data.DataLoader(XRAYDataTestSet(args.data_dir, args.data_list, crop_size=(512, 512), mean=IMG_MEAN, scale=False, mirror=False), batch_size=1, shuffle=False, pin_memory=True)
    interp = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=True)

    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    for index, batch in enumerate(testloader):
        image, size, name = batch

        with torch.no_grad():
            prediction = model(image.cuda())

            if isinstance(prediction, list):
                prediction = prediction[0]

            prediction = interp(prediction).cpu().data[0].numpy().transpose(1, 2, 0)

        prediction[:,:,0] = np.where(prediction[:,:,0]<0.5, 0, prediction[:,:,0])
        prediction[:,:,0] = np.where(prediction[:,:,0]>1, 1, prediction[:,:,0])
        prediction[:,:,0] = np.where(prediction[:,:,0]>=0.5, 1, prediction[:,:,0])
        output_im = PILImage.fromarray((prediction[:,:,0]*255).astype(np.uint8))
        output_im.save('./outputs/' + os.path.basename(name[0]).replace('.png', '_result.png'), 'png')


        print(str(c) + ' ... saved ')
        c = c+1

if __name__ == '__main__':
    main()
