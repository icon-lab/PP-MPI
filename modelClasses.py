# import libraries
import torch
from torch import nn
from reconUtils import *
import os


class dense_block(nn.Module):
    def __init__(self, in_channels, addition_channels, bias = True):
        super(dense_block, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=addition_channels, kernel_size=3,stride=1,padding=1, bias = bias)
        self.relu = nn.ReLU()
    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))],dim=1)
    
class rdb(nn.Module):
    def __init__(self, in_channels, C, growth_at_each_dense, bias = True):
        super(rdb, self).__init__()
        denses = nn.ModuleList()
        for i in range(0,C):
            denses.append(dense_block(in_channels+i*growth_at_each_dense,growth_at_each_dense, bias = bias))
        self.local_res_block = nn.Sequential(*denses)
        self.last_conv = nn.Conv2d(in_channels=in_channels+C*growth_at_each_dense,out_channels=in_channels,kernel_size=1,stride=1,padding=0, bias = bias)
    def forward(self,x):
        return x + self.last_conv(self.local_res_block(x))


class rdnDenoiserResRelu(nn.Module):
    def __init__(self,input_channels, nb_of_features, nb_of_blocks, layer_in_each_block, growth_rate, out_channel, bias = True):
        super(rdnDenoiserResRelu,self).__init__()
        
        self.conv0 = nn.Conv2d(input_channels,nb_of_features, kernel_size = 3, stride = 1, padding=1, bias = bias)
        self.conv1 = nn.Conv2d(nb_of_features, nb_of_features, kernel_size = 3, stride = 1, padding=1, bias = bias)
        self.rdbs = nn.ModuleList()
        for i in range(0,nb_of_blocks):
            self.rdbs.append(rdb(nb_of_features, layer_in_each_block, growth_rate, bias = bias))
        self.conv2 = nn.Conv2d(in_channels=nb_of_blocks*nb_of_features, out_channels= nb_of_features,kernel_size=1,stride=1,padding=0, bias = bias)
        self.conv3 = nn.Conv2d(in_channels=nb_of_features, out_channels= nb_of_features,kernel_size=3,stride=1,padding=1, bias = bias)
        self.conv4 = nn.Conv2d(in_channels=nb_of_features, out_channels= out_channel, kernel_size=3,stride=1,padding=1, bias = bias)
        self.lastReLU = nn.ReLU(inplace=False)
    def forward(self, x):
        x_init = x
        x = self.conv0(x)
        residual0 = x
        x = self.conv1(x)
        rdb_outs = list()
        for layer in self.rdbs:
            x = layer(x)
            rdb_outs.append(x)
        x = torch.cat(rdb_outs, dim=1)
        x = self.conv2(x)
        x = self.conv3(x) +residual0
        return self.lastReLU(self.conv4(x) + x_init)


class rdnDenoiserResRelu3d(nn.Module):
    def __init__(self,input_channels, nb_of_features, nb_of_blocks, layer_in_each_block, growth_rate, out_channel, bias = True):
        super(rdnDenoiserResRelu3d,self).__init__()
        
        self.conv0 = nn.Conv3d(input_channels,nb_of_features, kernel_size = 3, stride = 1, padding=1, bias = bias)
        self.conv1 = nn.Conv3d(nb_of_features, nb_of_features, kernel_size = 3, stride = 1, padding=1, bias = bias)
        self.rdbs = nn.ModuleList()
        for i in range(0,nb_of_blocks):
            self.rdbs.append(rdb3d(nb_of_features, layer_in_each_block, growth_rate, bias = bias))
        self.conv2 = nn.Conv3d(in_channels=nb_of_blocks*nb_of_features, out_channels= nb_of_features,kernel_size=1,stride=1,padding=0, bias = bias)
        self.conv3 = nn.Conv3d(in_channels=nb_of_features, out_channels= nb_of_features,kernel_size=3,stride=1,padding=1, bias = bias)
        self.conv4 = nn.Conv3d(in_channels=nb_of_features, out_channels= out_channel, kernel_size=3,stride=1,padding=1, bias = bias)
        self.lastReLU = nn.ReLU(inplace=False)
    def forward(self, x):
        x_init = x
        x = self.conv0(x)
        residual0 = x
        x = self.conv1(x)
        rdb_outs = list()
        for layer in self.rdbs:
            x = layer(x)
            rdb_outs.append(x)
        x = torch.cat(rdb_outs, dim=1)
        x = self.conv2(x)
        x = self.conv3(x) +residual0
        return self.lastReLU(self.conv4(x) + x_init)
    
class dense_block3d(nn.Module):
    def __init__(self, in_channels, addition_channels, bias = True):
        super(dense_block3d, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=addition_channels, kernel_size=3,stride=1,padding=1, bias = bias)
        self.relu = nn.ReLU()
    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))],dim=1)
    
class rdb3d(nn.Module):
    def __init__(self, in_channels, C, growth_at_each_dense, bias = True):
        super(rdb3d, self).__init__()
        denses = nn.ModuleList()
        for i in range(0,C):
            denses.append(dense_block3d(in_channels+i*growth_at_each_dense,growth_at_each_dense, bias = bias))
        self.local_res_block = nn.Sequential(*denses)
        self.last_conv = nn.Conv3d(in_channels=in_channels+C*growth_at_each_dense,out_channels=in_channels,kernel_size=1,stride=1,padding=0, bias = bias)
    def forward(self,x):
        return x + self.last_conv(self.local_res_block(x))

def getModel(descriptor):
    descriptor = descriptor[:-11] if "\n_scheduled" in descriptor else descriptor
    descriptor = descriptor[:-3] if "+tv" in descriptor else descriptor
    descriptor = descriptor[:-3] if "+l1" in descriptor else descriptor
    folderPath = "training/denoiser/"+descriptor

    #     print(folderPath)
    fileName = [i for i in os.listdir(folderPath) if "END" in i][0]
    filePath = folderPath + "/" +fileName
    
    descriptor = descriptor[6:]

    nb_of_features = int(descriptor.split("_")[13][2:])
    nb_of_blocks = int(descriptor.split("_")[14][2:])
    layer_in_each_block = int(descriptor.split("_")[15][4:])
    growth_rate = int(descriptor.split("_")[16][2:])
    biasFlag = True


    model = rdnDenoiserResRelu(input_channels=1,
                nb_of_features=nb_of_features,
                nb_of_blocks=nb_of_blocks,
                layer_in_each_block=layer_in_each_block, 
                growth_rate=growth_rate,
                out_channel=1,
                bias = biasFlag)
#         print(filePath)

    model.load_state_dict(torch.load(filePath, map_location='cpu'))
    model.cuda()

    for param in model.parameters():
        param.requires_grad = False
    model.eval()
#     print(sum(pVal.numel() for pVal in model.parameters() if pVal.requires_grad))
    print("num params of model: ", sum(pVal.numel() for pVal in model.parameters() if not pVal.requires_grad))
    return model
