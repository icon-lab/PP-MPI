import numpy as np
from torch import nn
import torch.nn.functional as F
import time

from torch.utils.data import DataLoader
import wandb
import torch

def admmInputGenerator(genData, U_, S_, V_, imgSize):

    compData = F.linear(genData, U_.T) # compressed data

    lsqrInp = F.linear(compData / (S_ + 1e-4), V_).reshape(imgSize)

    return compData, lsqrInp


def transformDataset(data, imgSizes, rescaleVals, randVals = None):

    dims = len(imgSizes)

    reScaleMin, reScaleMax = rescaleVals
    imgSize = data.shape

    if dims == 2:
        n1, n2 = imgSizes

        data -= data.reshape(imgSize[0], imgSize[1], -1).min(dim = 2).values[:,:,None,None]
        data /= data.reshape(imgSize[0], imgSize[1], -1).max(dim = 2).values[:,:,None,None]

        if (n1 < data.shape[2]) or (n2 < data.shape[3]):
            if randVals is None:
                rand1 = torch.randint(low = 0, high = data.shape[2] - n1, size = (1,))
                rand2 = torch.randint(low = 0, high = data.shape[3] - n2, size = (1,))
            else:
                rand1 = randVals[0]
                rand2 = randVals[1]
        else:
            rand1 = 0
            rand2 = 0
        data = data[:, :, rand1:rand1 + n1, rand2:rand2 + n2]
    else:
        n1, n2, n3 = imgSizes # n1 is 16

        data -= data.reshape(imgSize[0], imgSize[1], -1).min(dim = 2).values[:,:,None,None,None]
        data /= data.reshape(imgSize[0], imgSize[1], -1).max(dim = 2).values[:,:,None,None,None]

        diffS = np.array(imgSizes) - np.array(data.shape[2:])
        diffS *= (diffS > 0)
        data = F.pad(data, (diffS[2], diffS[2], diffS[1], diffS[1], diffS[0], diffS[0])) # new data size is 

        if (n1 < data.shape[2]) or (n2 < data.shape[3]) or (n3 < data.shape[3]):
            if randVals is None:
                rand1 = torch.randint(low = 0, high = data.shape[2] - n1, size = (1,))
                rand2 = torch.randint(low = 0, high = data.shape[3] - n2, size = (1,))
                rand3 = torch.randint(low = 0, high = data.shape[4] - n3, size = (1,))
            else:
                rand1 = randVals[0]
                rand2 = randVals[1]
                rand3 = randVals[2]
        else:
            rand1 = 0
            rand2 = 0
            rand3 = 0

        data = data[:, :, rand1:rand1 + n1, rand2:rand2 + n2, rand3:rand3 + n3]
    
    if reScaleMax > 0:
        randScale = torch.rand((imgSize[0], 1, 1, 1), device = data.device) * reScaleMax + reScaleMin
        if dims == 3:
            randScale = randScale.reshape(imgSize[0], 1, 1, 1, 1)
    else:
        randScale = 1

    data *= randScale
    return data

def returnNsTerm(fixedNoiseStdFlag = False,shape=0, maxNoiseStd=0.05, minNoiseStd=0, useDev = torch.device("cpu")):
    if fixedNoiseStdFlag:
        noiseStd = maxNoiseStd
        return torch.randn(shape, device = useDev) * noiseStd
    else:
        noiseStd = (torch.rand(shape, device = useDev)*(maxNoiseStd-minNoiseStd)+minNoiseStd)
        return torch.randn(shape, device = useDev) * noiseStd




def trainDenoiser(model, epoch_nb, loss, optimizer, scheduler, trainDataset, valDataset, batch_size_train, batch_size_val, rescaleVals = [1, 1], saveModelEpoch=0, valEpoch=0, saveDirectory='', maxNoiseStd = 0.1, optionalMessage="", 
    wandbFlag=False, fixedNoiseStdFlag = False, minNoiseStd =0, dims = 2):

    trainLosses = torch.zeros(epoch_nb)
    trainNrmses = torch.zeros(epoch_nb)
    trainPsnrs = torch.zeros(epoch_nb)
    valLosses = list()
    valNrmses = list()
    valPsnrs = list()
    trainLoader = DataLoader(trainDataset, batch_size_train, shuffle=True)
    valLoader = DataLoader(valDataset, valDataset.__len__(), shuffle=False)

    for epoch in range(1,1+int(epoch_nb)):
        tempLosses = list()
        model.train()
        tempNrmseNumeratorSquare = 0
        tempNrmseDenumeratorSquare = 0
        tempNumel = 0
        tempTime = time.time()

        if saveModelEpoch > 0:
            if (epoch % saveModelEpoch == 0):
                torch.save(model.state_dict(), saveDirectory+r"/"+ optionalMessage +"epoch"+ str(epoch)+ ".pth")

        for idx, data in enumerate(trainLoader, 0):
            data = data.float().cuda()

            data = transformDataset(data, [*data.shape[2:]], rescaleVals)

            noiseTerm = returnNsTerm(fixedNoiseStdFlag = fixedNoiseStdFlag, shape=data.shape, \
                                     maxNoiseStd=maxNoiseStd, minNoiseStd=minNoiseStd, useDev = data.device)
            noisyInp = data + noiseTerm
            modelOut = model(noisyInp)
            model.zero_grad()
            model_loss = loss(modelOut, data)
            model_loss.backward()
            optimizer.step()
            if dims == 3:
                percent = 10
                if idx%int(trainDataset.__len__()/batch_size_train/percent)==0:
                    print('Epoch {0:d} | {1:d}% | batch nrmse: {2:.5f}'.format(epoch,percent*idx//int(trainDataset.__len__()/batch_size_train/percent),(float(torch.norm(modelOut-data)))/(float(torch.norm(data))))) # myflag

            with torch.no_grad():
                tempLosses.append(float(model_loss))
                tempNrmseNumeratorSquare += (float(torch.norm(modelOut-data)))**2
                tempNrmseDenumeratorSquare += (float(torch.norm(data)))**2
                tempNumel += modelOut.numel()
        # back to epoch
        model.eval()
        scheduler.step()
            
        trainLosses[epoch-1] = sum(tempLosses)/len(tempLosses)
        trainNrmses[epoch-1] = (tempNrmseNumeratorSquare/tempNrmseDenumeratorSquare)**(1/2)
        trainPsnrs[epoch-1] = 20 * \
                    torch.log10(1 / (tempNrmseDenumeratorSquare**(1/2) * #Should we correct 1 -> valGround.max()
                                trainNrmses[epoch-1] / (tempNumel) ** (1/2)))
        epochTime = time.time() - tempTime
        if wandbFlag:
            wandb.log({"train_loss": trainLosses[epoch-1], "train_nrmse": trainNrmses[epoch-1], "train_psnr": trainPsnrs[epoch-1]})
        
        print("Epoch: {0}, Train Loss = {1:.6f}, Train nRMSE = {2:.6f}, Train pSNR = {3:.6f}, time elapsed = {4:.6f}".format(epoch, 
                                                trainLosses[epoch-1], trainNrmses[epoch-1], trainPsnrs[epoch-1], epochTime))        
        
        if valEpoch>0:
            if epoch % valEpoch == 0:
                with torch.no_grad():
                    model.eval()
                    valInp = next(iter(valLoader))

                    valGround = valInp.clone()
                    valGround = transformDataset(valGround, [*valInp.shape[2:]], rescaleVals)
                    noiseTerm = returnNsTerm(fixedNoiseStdFlag = fixedNoiseStdFlag,shape=valInp.shape,\
                                     maxNoiseStd=maxNoiseStd, minNoiseStd=minNoiseStd, useDev = valGround.device)
                    valInpVal = valGround + noiseTerm

                    valOut = torch.zeros_like(valGround)
                    deviceVal = valGround.device #'cuda' if valOut.is_cuda else 'cpu'
                    
                    iii = 0
                    while(iii < valInpVal.shape[0]-(valInpVal.shape[0] % batch_size_val)):
                        valInpC = valInpVal[iii:iii+batch_size_val].float().cuda()
                        valOut[iii:iii+batch_size_val] = model(valInpC).to(deviceVal)
                        iii += batch_size_val
                    
                    valInpC = valInpVal[iii:].float().cuda()
                    valOut[iii:] = model(valInpC).to(deviceVal)
                
                    valLoss = float(nn.L1Loss()(valGround, valOut))
                    valNrmse = float(torch.norm(valGround-valOut)/torch.norm(valGround))
                    valPSNR= float(20 * \
                    torch.log10(1 / (torch.norm(valGround) * #Should we correct 1 -> valGround.max()
                                valNrmse / (valOut.numel()) ** (1/2))))
                    valPSNR_avg = (20 *
                                    torch.log10(1 / (torch.norm(valGround.reshape(valGround.shape[0],-1)-valOut.reshape(valGround.shape[0],-1), dim = (1)).squeeze() / (valOut[0,0].numel()) ** (1/2))))

                    valLosses.append(valLoss)
                    valNrmses.append(valNrmse)
                    valPsnrs.append(valPSNR)
                if wandbFlag:
                    wandb.log({"valid_nRMSE": valNrmse,
                               "ref_nRMSE": torch.norm(valInp-valGround)/torch.norm(valGround),
                              'valid_pSNR': valPSNR,
                              'valid_pSNRavg': valPSNR_avg.mean(0),
                              'valid_pSNRstd': valPSNR_avg.std(0),
                               'valid_loss': valLoss})
                print("---Epoch: {0}, Val Loss = {1:.6f}, Val nRMSE = {2:.6f}, Val pSNR = {3:.6f}".format(epoch, valLoss, valNrmse, valPSNR))

    if wandbFlag:
        wandb.log({"valid_nRMSE": valNrmse,
                  'valid_pSNR': valPSNR,
                  'valid_pSNRavg': valPSNR_avg.mean(0),
                  'valid_pSNRstd': valPSNR_avg.std(0),
                   'valid_loss': valLoss})
    print("---Epoch: {0}, Val Loss = {1:.6f}, Val nRMSE = {2:.6f}, Val pSNR = {3:.6f}".format(epoch, valLoss, valNrmse, valPSNR))        
    torch.save(model.state_dict(), saveDirectory+r"/"+ optionalMessage +"epoch"+ str(epoch)+ "END.pth")
    return model, [trainLosses.numpy(), trainNrmses.numpy(), trainPsnrs.numpy()], [np.array(valLosses), np.array(valNrmses), np.array(valPsnrs)]
