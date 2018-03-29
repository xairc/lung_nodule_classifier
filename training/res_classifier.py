import torch
from torch import nn

from layers import *

config = {}
#config['anchors'] = [ 5.0, 10.0, 30.0, 60.0]
config['chanel'] = 1
config['img_size'] = [48, 48, 48]

config['num_neg'] = 800
config['bound_size'] = 3
config['reso'] = 1
config['aug_scale'] = True
config['r_rand_crop'] = 0.3
config['pad_value'] = 0
#config['pad_value'] = 0
config['augtype'] = {'flip':True,'swap':False,'scale':False,'rotate':False}

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # The first few layers consumes the most memory, so use simple convolution to save memory.
        # Call these layers preBlock, i.e., before the residual blocks of later layers.
        self.preBlock = nn.Sequential(
            nn.Conv3d(2, 48, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(48),
            nn.ReLU(inplace = True),
            nn.Conv3d(48, 48, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(48),
            nn.ReLU(inplace = True))
        
        # 3 poolings, each pooling downsamples the feature map by a factor 2.
        # 3 groups of blocks. The first block of each group has one pooling.
        num_blocks_forw = [3,3,4]
        self.featureNum_forw = [48,64,128,256]
        for i in range(len(num_blocks_forw)):
            blocks = []
            for j in range(num_blocks_forw[i]):
                if j == 0:
                    blocks.append(PostRes(self.featureNum_forw[i], self.featureNum_forw[i+1]))
                else:
                    blocks.append(PostRes(self.featureNum_forw[i+1], self.featureNum_forw[i+1]))
            setattr(self, 'forw' + str(i + 1), nn.Sequential(*blocks))

        self.maxpool1 = nn.MaxPool3d(kernel_size=2,stride=2,return_indices =True)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2,stride=2,return_indices =True)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2,stride=2,return_indices =True)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)

        self.drop = nn.Dropout3d(p = 0.5, inplace = False)
        self.malignacy_conv = nn.Sequential(nn.Conv3d(256, 128, kernel_size = 2),
                                    nn.BatchNorm3d(128),
                                    nn.ReLU(),
                                    nn.Conv3d(128, 128, kernel_size=1),
                                    nn.ReLU(),
                                    nn.AvgPool3d((2, 2, 2), stride=(2, 2, 2)))
        self.malignacy_fc = nn.Sequential(nn.Linear(128, 1))

        self.sphericiy_conv = nn.Sequential(nn.Conv3d(256, 128, kernel_size = 2),
                                    nn.BatchNorm3d(128),
                                    nn.ReLU(),
                                    nn.Conv3d(128, 128, kernel_size=1),
                                    nn.ReLU(),
                                    nn.AvgPool3d((2, 2, 2), stride=(2, 2, 2)))
        self.sphericiy_fc = nn.Sequential(nn.Linear(128, 1))

        self.margin_conv = nn.Sequential(nn.Conv3d(256, 128, kernel_size = 2),
                                    nn.BatchNorm3d(128),
                                    nn.ReLU(),
                                    nn.Conv3d(128, 128, kernel_size=1),
                                    nn.ReLU(),
                                    nn.AvgPool3d((2, 2, 2), stride=(2, 2, 2)))
        self.margin_fc = nn.Sequential(nn.Linear(128, 1))

        self.spiculation_conv = nn.Sequential(nn.Conv3d(256, 128, kernel_size = 2),
                                    nn.BatchNorm3d(128),
                                    nn.ReLU(),
                                    nn.Conv3d(128, 128, kernel_size=1),
                                    nn.ReLU(),
                                    nn.AvgPool3d((2, 2, 2), stride=(2, 2, 2)))
        self.spiculation_fc = nn.Sequential(nn.Linear(128, 1))

        self.texture_conv = nn.Sequential(nn.Conv3d(256, 128, kernel_size = 2),
                                    nn.BatchNorm3d(128),
                                    nn.ReLU(),
                                    nn.Conv3d(128, 128, kernel_size=1),
                                    nn.ReLU(),
                                    nn.AvgPool3d((2, 2, 2), stride=(2, 2, 2)))
        self.texture_fc = nn.Sequential(nn.Linear(128, 1))

        self.calcification_conv = nn.Sequential(nn.Conv3d(256, 128, kernel_size = 2),
                                    nn.BatchNorm3d(128),
                                    nn.ReLU(),
                                    nn.Conv3d(128, 128, kernel_size=1),
                                    nn.ReLU(),
                                    nn.AvgPool3d((2, 2, 2), stride=(2, 2, 2)))
        self.calcification_fc = nn.Sequential(nn.Linear(128, 1))

        self.internal_structure_conv = nn.Sequential(nn.Conv3d(256, 128, kernel_size = 2),
                                    nn.BatchNorm3d(128),
                                    nn.ReLU(),
                                    nn.Conv3d(128, 128, kernel_size=1),
                                    nn.ReLU(),
                                    nn.AvgPool3d((2, 2, 2), stride=(2, 2, 2)))
        self.internal_structure_fc = nn.Sequential(nn.Linear(128, 1))

        self.lobulation_conv = nn.Sequential(nn.Conv3d(256, 128, kernel_size = 2),
                                    nn.BatchNorm3d(128),
                                    nn.ReLU(),
                                    nn.Conv3d(128, 128, kernel_size=1),
                                    nn.ReLU(),
                                    nn.AvgPool3d((2, 2, 2), stride=(2, 2, 2)))
        self.lobulation_fc = nn.Sequential(nn.Linear(128, 1))

        self.subtlety_conv = nn.Sequential(nn.Conv3d(256, 128, kernel_size = 2),
                                    nn.BatchNorm3d(128),
                                    nn.ReLU(),
                                    nn.Conv3d(128, 128, kernel_size=1),
                                    nn.ReLU(),
                                    nn.AvgPool3d((2, 2, 2), stride=(2, 2, 2)))
        self.subtlety_fc = nn.Sequential(nn.Linear(128, 1))


    def forward(self, x, dropout=False):
        out = self.preBlock(x)#16
        out_pool,indices0 = self.maxpool1(out)
        if dropout:
            out_pool = self.drop(out_pool)
        out1 = self.forw1(out_pool)#32
        out1_pool,indices1 = self.maxpool2(out1)
        if dropout:
            out1_pool = self.drop(out1_pool)
        out2 = self.forw2(out1_pool)#64
        out2_pool,indices2 = self.maxpool3(out2)
        if dropout:
            out2_pool = self.drop(out2_pool)
        out3 = self.forw3(out2_pool)#96
        out3_pool, indices3 = self.maxpool4(out3)
        if dropout:
            out3_pool = self.drop(out3_pool)

        malignacy = self.malignacy_conv(out3_pool)
        malignacy = self.drop(malignacy)
        malignacy = malignacy.view(malignacy.size(0), -1)
        malignacy = self.malignacy_fc(malignacy)

        sphericiy = self.sphericiy_conv(out3_pool)
        sphericiy = self.drop(sphericiy)
        sphericiy = sphericiy.view(sphericiy.size(0), -1)
        sphericiy = self.sphericiy_fc(sphericiy)

        margin = self.margin_conv(out3_pool)
        margin = self.drop(margin)
        margin = margin.view(margin.size(0), -1)
        margin = self.margin_fc(margin)

        spiculation = self.spiculation_conv(out3_pool)
        spiculation = self.drop(spiculation)
        spiculation = spiculation.view(spiculation.size(0), -1)
        spiculation = self.spiculation_fc(spiculation)

        texture = self.texture_conv(out3_pool)
        texture = self.drop(texture)
        texture = texture.view(texture.size(0), -1)
        texture = self.texture_fc(texture)

        calcification = self.calcification_conv(out3_pool)
        calcification = self.drop(calcification)
        calcification = calcification.view(calcification.size(0), -1)
        calcification = self.calcification_fc(calcification)

        internal_structure = self.internal_structure_conv(out3_pool)
        internal_structure = self.drop(internal_structure)
        internal_structure = internal_structure.view(internal_structure.size(0), -1)
        internal_structure = self.internal_structure_fc(internal_structure)

        lobulation = self.lobulation_conv(out3_pool)
        lobulation = self.drop(lobulation)
        lobulation = lobulation.view(lobulation.size(0), -1)
        lobulation = self.lobulation_fc(lobulation)

        subtlety = self.subtlety_conv(out3_pool)
        subtlety = self.drop(subtlety)
        subtlety = subtlety.view(subtlety.size(0), -1)
        subtlety = self.subtlety_fc(subtlety)

        out_all = torch.cat((malignacy, sphericiy, margin, spiculation,
                             texture, calcification, internal_structure, lobulation, subtlety), 1)

        return out_all

    
def get_model():
    net = Net()
    loss = MAE_Loss()
    return config, net, loss
