import datasets.transform as tr
import numpy as np
import os
from PIL import Image
import random
import torch
import cv2
from torch.utils.data import Dataset
from torchvision import transforms


class ChangeDetection(Dataset):
    CLASSES = ['未变化区域', '水体', '地面', '低矮植被', '树木', '建筑物', '运动场']

    def __init__(self, root, mode):
        super(ChangeDetection, self).__init__()
        self.root = root
        self.mode = mode

        if mode == 'train':
            self.root = os.path.join(self.root, 'train-+')
            self.ids = os.listdir(os.path.join(self.root, "im1"))
        elif mode == 'val':
            self.root = os.path.join(self.root, 'val-+')
            self.ids = os.listdir(os.path.join(self.root, "im1"))
        self.ids.sort()

        self.transform = transforms.Compose([
            tr.RandomFlipOrRotate()
        ])

        self.transform_imagefilter = transforms.Compose([
            tr.RandomImageFilter()
        ])

        self.transform_imageenhance = transforms.Compose([
            tr.RandomImageEnhance()
        ])


        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])


    def __getitem__(self, index):
        id = self.ids[index]

        img1 = Image.open(os.path.join(self.root, 'im1', id))
        img2 = Image.open(os.path.join(self.root, 'im2', id))
            

        mask1 = Image.open(os.path.join(self.root, 'label1', id))
        mask2 = Image.open(os.path.join(self.root, 'label2', id))


        gt_mask1 = np.array(Image.open(os.path.join(self.root, 'label1', id)))
        mask_bin = np.zeros_like(gt_mask1)
        mask_bin[gt_mask1 != 0] = 1
        mask_bin = Image.fromarray(mask_bin)
        
        if self.mode == 'train':
        
            #------------Weak data augmentation----------------------------
            sample = self.transform({'img1': img1, 'img2': img2, 'mask1': mask1, 'mask2': mask2,
                                        'mask_bin': mask_bin})
            img1, img2, mask1, mask2, mask_bin = sample['img1'], sample['img2'], sample['mask1'], \
                                                    sample['mask2'], sample['mask_bin']

            #------------Strong data augmentation----------------------------
            sample = self.transform_imagefilter({'img1': img1, 'img2': img2})
            img1, img2 = sample['img1'], sample['img2']

            sample = self.transform_imageenhance({'img1': img1, 'img2': img2})
            img1, img2 = sample['img1'], sample['img2']
   
    
        img1 = self.normalize(img1)
        img2 = self.normalize(img2)
        mask1 = torch.from_numpy(np.array(mask1)).long()
        mask2 = torch.from_numpy(np.array(mask2)).long()
        mask_bin = torch.from_numpy(np.array(mask_bin)).float()
        if self.mode == 'train':
            return img1, img2, mask1, mask2, mask_bin

        return img1, img2, mask1, mask2, mask_bin, id

    def __len__(self):
        return len(self.ids)


#-----------------------------------------------------------------------------
#                               Land-sat dataset
#-----------------------------------------------------------------------------
# import datasets.transform as tr
# import numpy as np
# import os
# from PIL import Image
# import random
# import torch
# import cv2
# from torch.utils.data import Dataset
# from torchvision import transforms


# class ChangeDetection(Dataset):
#     CLASSES = ['未变化区域', '农田', '沙漠', '建筑', '水']

#     def __init__(self, root, mode):
#         super(ChangeDetection, self).__init__()
#         self.root = root
#         self.mode = mode

#         if mode == 'train':
#             self.root = os.path.join(self.root, 'Landsat-SCD_dataset_train')
#             self.ids = os.listdir(os.path.join(self.root, "A"))
#         elif mode == 'val':
#             self.root = os.path.join(self.root, 'Landsat-SCD_dataset_val')
#             self.ids = os.listdir(os.path.join(self.root, "A"))
#         self.ids.sort()

#         self.transform = transforms.Compose([
#             tr.RandomFlipOrRotate()
#         ])

#         self.transform_imagefilter = transforms.Compose([
#             tr.RandomImageFilter()
#         ])

#         self.transform_imageenhance = transforms.Compose([
#             tr.RandomImageEnhance()
#         ])


#         self.normalize = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
#         ])


#     def __getitem__(self, index):
#         id = self.ids[index]

#         img1 = Image.open(os.path.join(self.root, 'A', id))
#         img2 = Image.open(os.path.join(self.root, 'B', id))
            

#         mask1 = Image.open(os.path.join(self.root, 'A_l', id))
#         mask2 = Image.open(os.path.join(self.root, 'B_l', id))


#         gt_mask1 = np.array(Image.open(os.path.join(self.root, 'label', id)))
#         mask_bin = np.zeros_like(gt_mask1)
#         mask_bin[gt_mask1 != 0] = 1
#         mask_bin = Image.fromarray(mask_bin)
        
#         if self.mode == 'train':
        
#             #------------Weak data augmentation----------------------------
#             sample = self.transform({'img1': img1, 'img2': img2, 'mask1': mask1, 'mask2': mask2,
#                                         'mask_bin': mask_bin})
#             img1, img2, mask1, mask2, mask_bin = sample['img1'], sample['img2'], sample['mask1'], \
#                                                     sample['mask2'], sample['mask_bin']

#             #------------Strong data augmentation----------------------------
#             # sample = self.transform_imagefilter({'img1': img1, 'img2': img2})
#             # img1, img2 = sample['img1'], sample['img2']

#             # sample = self.transform_imageenhance({'img1': img1, 'img2': img2})
#             # img1, img2 = sample['img1'], sample['img2']
   
    
#         img1 = self.normalize(img1)
#         img2 = self.normalize(img2)
#         mask1 = torch.from_numpy(np.array(mask1)).long()
#         mask2 = torch.from_numpy(np.array(mask2)).long()
#         mask_bin = torch.from_numpy(np.array(mask_bin)).float()
#         if self.mode == 'train':
#             return img1, img2, mask1, mask2, mask_bin

#         return img1, img2, mask1, mask2, mask_bin, id

#     def __len__(self):
#         return len(self.ids)

