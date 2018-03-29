import numpy as np
import torch
from torch.utils.data import Dataset
import os
import time
import collections
from scipy.ndimage.interpolation import rotate

class DataBowl3Detector(Dataset):
    def __init__(self, data_dir, split_path, config, phase = 'train',split_comber=None):
        assert(phase == 'train' or phase == 'val' or phase == 'test')
        self.phase = phase
        self.isScale = config['aug_scale']
        self.r_rand = config['r_rand_crop']

        self.augtype = config['augtype']
        self.pad_value = config['pad_value']
        self.idcs = np.load(split_path)

        self.filenames = [os.path.join(data_dir, '%s_clean.npy' % idx) for idx in self.idcs]

        labels = []
        candidates = []
        # labels
        # name, pos_z, pos_y, pos_x, size, malignacy, sphericiy, margin, spiculation, texture, calcification, internal_structure, lobulation, subtlety, hit_count
        for idx in self.idcs:
            l = np.load(os.path.join(data_dir, '%s_attribute.npy' %idx))
            if np.all(l==0):
                l=np.array([])
            labels.append(l)

        if self.phase != 'test':
            self.bboxes = []
            for i, l in enumerate(labels):
                if len(l) > 0 :
                    for label in l:
                        self.bboxes.append([np.concatenate([[i], label[1:]])])

            self.bboxes = np.concatenate(self.bboxes,axis = 0)
            np.random.shuffle(self.bboxes)
        #self.crop = Crop(config)
        self.crop = padding_crop(config)

    def __getitem__(self, idx,split=None):
        #print ('get item', idx)
        t = time.time()

        bbox = self.bboxes[idx]
        filename = self.filenames[int(bbox[0])]
        imgs = np.load(filename)
        isScale = self.augtype['scale'] and (self.phase=='train')

        if self.phase=='train':
            sample, target = self.crop(imgs, bbox[1:], True)
            sample, target = augment(sample, target,
                                    ifflip=self.augtype['flip'], ifrotate=self.augtype['rotate'],
                                    ifswap=self.augtype['swap'])
        else:
            sample, target = self.crop(imgs, bbox[1:], False)

        label = np.array(bbox[5:14])
        sample = (sample.astype(np.float32)-128)/128

        #if filename in self.kagglenames and self.phase=='train':
        #    label[label==-1]=0
        return torch.from_numpy(sample), torch.from_numpy(label)

    def __len__(self):
        return len(self.bboxes)
        
        
def augment(sample, target, ifflip = True, ifrotate=True, ifswap = True):
    #                     angle1 = np.random.rand()*180

    if ifrotate:
        validrot = False
        counter = 0
        while not validrot:
            newtarget = np.copy(target)
            angle1 = np.random.rand()*180
            size = np.array(sample.shape[2:4]).astype('float')
            rotmat = np.array([[np.cos(angle1/180*np.pi),-np.sin(angle1/180*np.pi)],[np.sin(angle1/180*np.pi),np.cos(angle1/180*np.pi)]])
            newtarget[1:3] = np.dot(rotmat,target[1:3]-size/2)+size/2
            if np.all(newtarget[:3]>target[3]) and np.all(newtarget[:3]< np.array(sample.shape[1:4])-newtarget[3]):
                validrot = True
                target = newtarget
                sample = rotate(sample,angle1,axes=(2,3),reshape=False)
            else:
                counter += 1
                if counter ==3:
                    break
    if ifswap:
        if sample.shape[1]==sample.shape[2] and sample.shape[1]==sample.shape[3]:
            axisorder = np.random.permutation(3)
            sample = np.transpose(sample,np.concatenate([[0],axisorder+1]))
            target[:3] = target[:3][axisorder]

    if ifflip:
#         flipid = np.array([np.random.randint(2),np.random.randint(2),np.random.randint(2)])*2-1
        flipid = np.array([1,np.random.randint(2),np.random.randint(2)])*2-1
        sample = np.ascontiguousarray(sample[:,::flipid[0],::flipid[1],::flipid[2]])
        for ax in range(3):
            if flipid[ax]==-1:
                target[ax] = np.array(sample.shape[ax+1])-target[ax]

    return sample, target

class Crop(object):
    def __init__(self, config):
        self.img_size = config['img_size']
        self.bound_size = config['bound_size']
        #self.stride = config['stride']
        self.pad_value = config['pad_value']

    def __call__(self, imgs, target):

        crop_size = self.img_size
        bound_size = self.bound_size

        target = np.copy(target)
        
        start = []
        for i in range(3):
            s = np.floor(target[i])+ 1 - bound_size
            e = np.ceil (target[i])+ 1 + bound_size - crop_size[i]


            if s>e:
                start.append(np.random.randint(e,s))#!
            else:
                start.append(int(target[i])-crop_size[i]/2+np.random.randint(-bound_size/2,bound_size/2))

        pad = []
        pad.append([0,0])
        for i in range(3):
            leftpad = max(0,-start[i])
            rightpad = max(0,start[i]+crop_size[i]-imgs.shape[i+1])
            pad.append([leftpad,rightpad])
        crop = imgs[:,
            max(start[0],0):min(start[0] + crop_size[0],imgs.shape[1]),
            max(start[1],0):min(start[1] + crop_size[1],imgs.shape[2]),
            max(start[2],0):min(start[2] + crop_size[2],imgs.shape[3])]

        crop = np.pad(crop,pad,'constant',constant_values =self.pad_value)

        for i in range(3):
            target[i] = target[i] - start[i]

        return crop, target


class padding_crop(object):
    def __init__(self, config):
        self.img_size = config['img_size']
        self.bound_size = config['bound_size']
        # self.stride = config['stride']
        self.pad_value = config['pad_value']

    def crop(self, imgs, target, train=True):
        crop_size = self.img_size
        bound_size = self.bound_size
        target = np.copy(target)

        start = []
        for i in range(3):
            start.append(int(target[i]) - int(crop_size[i] / 2))

        pad = []
        pad.append([0, 0])
        for i in range(3):
            leftpad = max(0, -start[i])
            rightpad = max(0, start[i] + crop_size[i] - imgs.shape[i + 1])
            pad.append([leftpad, rightpad])
        crop = imgs[:,
               max(start[0], 0):min(start[0] + crop_size[0], imgs.shape[1]),
               max(start[1], 0):min(start[1] + crop_size[1], imgs.shape[2]),
               max(start[2], 0):min(start[2] + crop_size[2], imgs.shape[3])]

        crop = np.pad(crop, pad, 'constant', constant_values=self.pad_value)

        for i in range(3):
            target[i] = target[i] - start[i]

        return crop, target

    def __call__(self, imgs, target, train=True):

        crop_img, target = self.crop(imgs, target, train)
        imgs = np.squeeze(crop_img, axis=0)

        z = int(target[0])
        y = int(target[1])
        x = int(target[2])
        #z = 24
        #y = 24
        #x = 24

        nodule_size = int(target[3])
        margin = max(7, nodule_size * 0.4)
        radius = int((nodule_size + margin) / 2)

        s_z_pad = 0
        e_z_pad = 0
        s_y_pad = 0
        e_y_pad = 0
        s_x_pad = 0
        e_x_pad = 0

        s_z = max(0, z - radius)
        if (s_z == 0):
            s_z_pad = -(z - radius)

        e_z = min(np.shape(imgs)[0], z + radius)
        if (e_z == np.shape(imgs)[0]):
            e_z_pad = (z + radius) - np.shape(imgs)[0]

        s_y = max(0, y - radius)
        if (s_y == 0):
            s_y_pad = -(y - radius)

        e_y = min(np.shape(imgs)[1], y + radius)
        if (e_y == np.shape(imgs)[1]):
            e_y_pad = (y + radius) - np.shape(imgs)[1]

        s_x = max(0, x - radius)
        if (s_x == 0):
            s_x_pad = -(x - radius)

        e_x = min(np.shape(imgs)[2], x + radius)
        if (e_x == np.shape(imgs)[2]):
            e_x_pad = (x + radius) - np.shape(imgs)[2]

        # print (s_x, e_x, s_y, e_y, s_z, e_z)
        # print (np.shape(img_arr[s_z:e_z, s_y:e_y, s_x:e_x]))
        nodule_img = imgs[s_z:e_z, s_y:e_y, s_x:e_x]
        nodule_img = np.pad(nodule_img, [[s_z_pad, e_z_pad], [s_y_pad, e_y_pad], [s_x_pad, e_x_pad]], 'constant',
                            constant_values=0)

        imgpad_size = [self.img_size[0] - np.shape(nodule_img)[0],
                       self.img_size[1] - np.shape(nodule_img)[1],
                       self.img_size[2] - np.shape(nodule_img)[2]]
        imgpad = []
        imgpad_left = [int(imgpad_size[0] / 2),
                       int(imgpad_size[1] / 2),
                       int(imgpad_size[2] / 2)]
        imgpad_right = [int(imgpad_size[0] / 2),
                        int(imgpad_size[1] / 2),
                        int(imgpad_size[2] / 2)]

        for i in range(3):
            if (imgpad_size[i] % 2 != 0):

                rand = np.random.randint(2)
                if rand == 0:
                    imgpad.append([imgpad_left[i], imgpad_right[i] + 1])
                else:
                    imgpad.append([imgpad_left[i] + 1, imgpad_right[i]])
            else:
                imgpad.append([imgpad_left[i], imgpad_right[i]])

        padding_crop = np.pad(nodule_img, imgpad, 'constant', constant_values=self.pad_value)

        padding_crop = np.expand_dims(padding_crop, axis=0)

        crop = np.concatenate((padding_crop, crop_img))

        return crop, target

def collate(batch):
    if torch.is_tensor(batch[0]):
        return [b.unsqueeze(0) for b in batch]
    elif isinstance(batch[0], np.ndarray):
        return batch
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], collections.Iterable):
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]

