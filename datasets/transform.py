#-----------------------------------------------------------------------------
#                               Base
#-----------------------------------------------------------------------------
from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance

import random
import numpy as np
import torch
from scipy.ndimage.interpolation import shift


class RandomFlipOrRotate(object):
    def __call__(self, sample):
        img1, img2, mask1, mask2, mask_bin = \
            sample['img1'], sample['img2'], sample['mask1'], sample['mask2'], sample['mask_bin']

        rand = random.random()
        if rand < 1 / 6:
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
            mask1 = mask1.transpose(Image.FLIP_LEFT_RIGHT)
            img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
            mask2 = mask2.transpose(Image.FLIP_LEFT_RIGHT)
            mask_bin = mask_bin.transpose(Image.FLIP_LEFT_RIGHT)

        elif rand < 2 / 6:
            img1 = img1.transpose(Image.FLIP_TOP_BOTTOM)
            mask1 = mask1.transpose(Image.FLIP_TOP_BOTTOM)
            img2 = img2.transpose(Image.FLIP_TOP_BOTTOM)
            mask2 = mask2.transpose(Image.FLIP_TOP_BOTTOM)
            mask_bin = mask_bin.transpose(Image.FLIP_TOP_BOTTOM)

        elif rand < 3 / 6:
            img1 = img1.transpose(Image.ROTATE_90)
            mask1 = mask1.transpose(Image.ROTATE_90)
            img2 = img2.transpose(Image.ROTATE_90)
            mask2 = mask2.transpose(Image.ROTATE_90)
            mask_bin = mask_bin.transpose(Image.ROTATE_90)

        elif rand < 4 / 6:
            img1 = img1.transpose(Image.ROTATE_180)
            mask1 = mask1.transpose(Image.ROTATE_180)
            img2 = img2.transpose(Image.ROTATE_180)
            mask2 = mask2.transpose(Image.ROTATE_180)
            mask_bin = mask_bin.transpose(Image.ROTATE_180)

        elif rand < 5 / 6:
            img1 = img1.transpose(Image.ROTATE_270)
            mask1 = mask1.transpose(Image.ROTATE_270)
            img2 = img2.transpose(Image.ROTATE_270)
            mask2 = mask2.transpose(Image.ROTATE_270)
            mask_bin = mask_bin.transpose(Image.ROTATE_270)


        return {'img1': img1, 'img2': img2, 'mask1': mask1, 'mask2': mask2, 'mask_bin': mask_bin}



class RandomImageFilter(object):
    def __call__(self, sample):
        img1, img2 = sample['img1'], sample['img2']

        rand = random.random()
        if rand < 1 / 6:
            img1 = img1.filter(ImageFilter.BLUR)
            img2 = img2.filter(ImageFilter.BLUR)

        elif rand < 2 / 6:
            img1 = img1.filter(ImageFilter.DETAIL)
            img2 = img2.filter(ImageFilter.DETAIL)

        elif rand < 3 / 6:
            img1 = img1.filter(ImageFilter.EDGE_ENHANCE)
            img2 = img2.filter(ImageFilter.EDGE_ENHANCE)

        elif rand < 4 / 6:
            img1 = img1.filter(ImageFilter.SMOOTH)
            img2 = img2.filter(ImageFilter.SMOOTH)

        elif rand < 5 / 6:
            img1 = img1.filter(ImageFilter.SHARPEN)
            img2 = img2.filter(ImageFilter.SHARPEN)
            
        return {'img1': img1, 'img2': img2}





class RandomImageEnhance(object):
    def __call__(self, sample):
        img1, img2 = sample['img1'], sample['img2']

        rand = random.random()
        if rand < 1 / 7:
            img1 = ImageEnhance.Brightness(img1).enhance(factor=1.5)
            img2 = ImageEnhance.Brightness(img2).enhance(factor=1.5)

        elif rand < 2 / 7:
            img1 = ImageEnhance.Brightness(img1).enhance(factor=0.5)
            img2 = ImageEnhance.Brightness(img2).enhance(factor=0.5)

        elif rand < 3 / 7:
            img1 = ImageEnhance.Color(img1).enhance(factor=0.5)
            img2 = ImageEnhance.Color(img2).enhance(factor=0.5)

        elif rand < 4 / 7:
            img1 = ImageEnhance.Color(img1).enhance(factor=1.5)
            img2 = ImageEnhance.Color(img2).enhance(factor=1.5)

        elif rand < 5 / 7:
            img1 = ImageEnhance.Contrast(img1).enhance(factor=0.5)
            img2 = ImageEnhance.Contrast(img2).enhance(factor=0.5)

        elif rand < 6 / 7:
            img1 = ImageEnhance.Contrast(img1).enhance(factor=1.5)
            img2 = ImageEnhance.Contrast(img2).enhance(factor=1.5)



        return {'img1': img1, 'img2': img2}

