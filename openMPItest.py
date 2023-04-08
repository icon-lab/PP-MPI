import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import torch.nn.functional as F

from data import *
from reconAlgos import *
from modelClasses import *
from trainerClasses import *

import gc
import time
import copy

# Settings for inference:

gpuNo = 0 # use GPU ID

initializationTo = 1 # initialize ADMM to least squares input
# 0: zeros
# 1: least squares input
# 2: Regularized least squares input

# run inference for these techniques:
# L1, TV, L1_TV are the conventional hand-crafted regularizers.
# other ones include folder names under "training/denoiser/" 
# after "+" one can include l1, tv for a linear combination of plug-and-play and l1 and/or tv. Finally, "(-1,0,1,2)" includes the proposed averaging in all three axes.

descriptorsHere = [\
"L1",\
"TV",\
"L1_TV",\
"ppmpi_lr_0.001_wd_0_bs_64_mxNs_0.1_fixNs_1_data_mnNs_0_nF12_nB4_lieb4_gr12_rMn1.0_0.0+  (2)",\
"ppmpi_lr_0.001_wd_0_bs_64_mxNs_0.1_fixNs_1_data_mnNs_0_nF12_nB4_lieb4_gr12_rMn1.0_0.0+  (-1,0,1,2)" \
                     ]

nbOfSingulars = 2200 # used number of singular values for truncated image reconstruction.

torch.cuda.set_device(gpuNo)
print(torch.cuda.get_device_name(gpuNo))

# Load Data

n1 = n2 = n3 = 19 # image dimensions

loadMtxOpenMPI = loadmat("OpenMPI/mtxAndPhantomOpenMpi.mat")
sysMtx = torch.from_numpy(loadMtxOpenMPI['Aconcat']).float().cuda()
U, S, Vh = torch.linalg.svd(\
    sysMtx.reshape(-1, n1 * n2 * n3), full_matrices=False)

if nbOfSingulars is None:
    nbOfSingulars = min(sysMtx.shape)

nbSvd = nbOfSingulars
U_ = U[:, :nbSvd]
S_ = S[:nbSvd]
Vh_ = Vh[:nbSvd, :]
V_ = Vh_.T
theSys = U_.T @ sysMtx 

bconcat = loadMtxOpenMPI['bconcat']
underlyingEpsilon = None
imgSize = [-1, 1, n1, n2,n3]
myDataGen = torch.from_numpy(bconcat).float().cuda().reshape(1, -1)

datatC, lsqrInp = admmInputGenerator(myDataGen, U_, S_, V_, imgSize)
underlyingImage = torch.linalg.lstsq(sysMtx, myDataGen.T).solution.T.reshape(-1, n1, n2,n3)

Nbatch = datatC.shape[0]
fwdBatch = None # None for batchless process

imgSize = [n1,n2,n3]
theWholeSize = ((-1, 1, *imgSize))

simMtx2 = theSys

AtC = simMtx2.reshape(-1, n1*n2*n3)
descriptors = copy.deepcopy(descriptorsHere)


if underlyingEpsilon is not None:
    epsilonVal = underlyingEpsilon
else:
    epsilonVal = torch.norm(datatC, dim=1) * 10**(-23/20)*1.5
    if epsilonVal.numel() == 1:
        epsilonVal = float(epsilonVal)

epsilon = epsilonVal
refVals = underlyingImage.reshape(Nbatch, -1)

datatC, lsqrInp = admmInputGenerator(myDataGen, U_, S_, V_, theWholeSize)
datatC = datatC.reshape(Nbatch, -1)

if initializationTo == 0:
    x_in = torch.zeros_like(underlyingImage)
elif initializationTo == 1:
    x_in = lsqrInp.reshape(-1, n1, n2)
else:
    x_in = F.linear(F.linear(datatC, AtC.T), torch.inverse(
        1 * torch.eye(n1 * n2).type_as(AtC.T) + AtC.T @ AtC)).reshape(-1, n1, n2)

outImgs = list()
outDiags = list()
outMaxs = list()
outNetworkNorms = list()
outNetworkNrmses = list()
outPsnrs = list()
outHfens = list()
outL1Obj = list()
outTVobj = list()
outCritobj = list()

outX1 = list()
outX2 = list()

inPsnrs = list()

inPsnrs.append(psnr(refVals, x_in.reshape(Nbatch, -1)))

mu1 = 1
mu2 = 10
mu3 = 50
MaxIter = 1200
verboseIn = 300
theADMMclass = ADMMfncs(AtC, MaxIter, verboseIn, imgSize)

Madmm = theADMMclass.MtC

for i, descriptor in enumerate(descriptors):
    theADMMclass.MaxIter = 350 if 'L1' in descriptor or 'TV' in descriptor else 100

    print("Run Reconstruction for: ", descriptor,end=' ')
    print("")

    startTime = time.time()
    if '(' in descriptor:
        theSliceIndices = eval(descriptor[descriptor.find('('):descriptor.find(')')+1])
        descriptors[i] = descriptor = descriptor[:descriptor.find('(')]
        print('theSliceIndices:',theSliceIndices)

    if not ("L1" in descriptor or "TV" in descriptor):

        model = getModel(descriptor[:-3])
        muScaleIter = 1
        mu2Scale = 1
        fnc1 = ppFnc2dnmBatch3d(1/mu1, model, shape3d = imgSize, batchAxis=theSliceIndices,fwdBatch=fwdBatch)

        fnc2 = softTV(1/30, imgSize, 10)
        fnc3 = softT(1/20)

        fncUse = None
        synCaller = False

        if 'tv' in descriptors[i]:
            fncUse = fnc2
        elif 'l1' in descriptors[i]:
            fncUse = fnc3
        elif 'lS' in descriptors[i]:
            fncUse = softTpos(1/20)
            synCaller = True

        if synCaller:
            x_rec, outDiag, outPsnr, outHfen, inpOutNorms, inpOutNrmses, outMax, l1Obj, TVobj, critObj, x1, x2 = theADMMclass.ADMMreconDualSynthesis(
                datatC, fnc1, fncUse, epsilon, refVals, mu2Scale=mu2Scale, muScaleIter=muScaleIter, x_in=x_in)
            outX1.append(x1)
            outX2.append(x2)
        else:
            if fncUse is None:
                x_rec, outDiag, outPsnr, outHfen, inpOutNorms, inpOutNrmses, outMax, l1Obj, TVobj, critObj = theADMMclass.afterFnc(
                    datatC, fnc1, fnc1, 0, epsilon, refVals, mu2Scale=1, muScaleIter=muScaleIter, x_in=x_in)
            else:
                x_rec, outDiag, outPsnr, outHfen, inpOutNorms, inpOutNrmses, outMax, l1Obj, TVobj, critObj = theADMMclass.ADMMreconDual(
                    datatC, fnc1, fncUse, epsilon, refVals, mu2Scale=mu2Scale, muScaleIter=muScaleIter, x_in=x_in)
    else:
        l1c = 100
        tvc = 100

        if not "TV" in descriptor:
            l1c = 1e3
            theADMMclass.MaxIter = 1000
        elif not "L1" in descriptor:
            tvc = 1e3
            theADMMclass.MaxIter = 1000
        else:
            l1c = 1e5
            tvc = 1e4
            theADMMclass.MaxIter = 2200

        muScaleIter = 70
        if not "TV" in descriptor:
            fnc3 = softT(1/l1c)
            x_rec, outDiag, outPsnr, outHfen, inpOutNorms, inpOutNrmses, outMax, l1Obj, TVobj, critObj = theADMMclass.afterFnc(
                datatC, fnc3, fnc3, 0, epsilon, refVals, mu2Scale=1, muScaleIter=muScaleIter, x_in=x_in)

        elif not "L1" in descriptor:
            fnc2 = softTV(1/tvc, imgSize, 10)
            x_rec, outDiag, outPsnr, outHfen, inpOutNorms, inpOutNrmses, outMax, l1Obj, TVobj, critObj = theADMMclass.afterFnc(
                datatC, fnc2, fnc2, 0, epsilon, refVals, mu2Scale=1, muScaleIter=muScaleIter, x_in=x_in)
        else:
            fnc3 = softT(1/l1c)
            fnc2 = softTV(1/tvc, imgSize, 10)
            x_rec, outDiag, outPsnr, outHfen, inpOutNorms, inpOutNrmses, outMax, l1Obj, TVobj, critObj = theADMMclass.ADMMreconDual(
                datatC, fnc2, fnc3, epsilon, refVals, mu2Scale=1, muScaleIter=muScaleIter, x_in=x_in)

    outImgs.append(x_rec)
    outDiags.append(outDiag)
    outMaxs.append(outMax)
    outPsnrs.append(outPsnr)
    outHfens.append(outHfen)
    outL1Obj.append(l1Obj)
    outTVobj.append(TVobj)
    outCritobj.append(critObj)
    outNetworkNorms.append(inpOutNorms)
    outNetworkNrmses.append(inpOutNrmses)
    print('Elapsed time for this method is {0:.2f} s.'.format(time.time()-startTime))
    gc.collect()
    torch.cuda.empty_cache()

startTime = time.time()
print("Run Reconstruction for: ART")
xART = ART(AtC, datatC, 5, 1e-2)
print('Elapsed time for this method is {0:.2f} s.'.format(time.time()-startTime))

### Plot images
### 
### 

def showImg(x, selAxes = 0):
    theRes = (n1, n2, n3)
    if x.numel() < n1 * n2 * n3:
        theRes = (19, 19, 19)
    if selAxes == 1:
        return x.cpu().reshape(theRes)[:,:,theRes[0] // 2].squeeze()
    elif selAxes == 2:
        return x.cpu().reshape(theRes)[:,theRes[0] // 2 - 1,:].squeeze()
    else:
        return x.cpu().reshape(theRes)[theRes[0] // 2,:,:].squeeze()

for selAxes in range(3):
    fontsize = 25
    plt.figure(figsize=(20,12))
    plt.subplot(len(descriptors)//4+1,4,1)
    plt.imshow(showImg(xART, selAxes = selAxes),cmap='gray')
    plt.clim((0,0.2))
    plt.axis('off')
    plt.title('ART', fontdict = {'fontsize' : fontsize})
    for idx, desc in enumerate(descriptorsHere):
        plt.subplot(len(descriptors)//4+1,4,idx+2)
        plt.imshow(showImg(outImgs[idx], selAxes = selAxes),cmap='gray')
        plt.clim((0,0.2))
        # plt.colorbar()
        plt.axis('off')
        if 'ppmpi' in desc and '(-1'  not in desc:
            title = 'PP-MPI (Single Slice)'
        elif 'ppmpi' in desc and '(-1'  in desc:
            title = 'PP-MPI (All Slices)'
        elif 'E2E' in desc:
            title = 'E2E'
        elif 'L1_TV' in desc:
            title = 'ADMM (ℓ1&TV)'
        elif 'L1' in desc:
            title = 'ADMM (ℓ1)'
        elif 'TV' in desc:
            title = 'ADMM (TV)'
        plt.title(title, fontdict = {'fontsize' : fontsize})
    plt.subplots_adjust(wspace=0.1, hspace=0.05)
    plt.show()
    plt.savefig("inferenceOut_ax{}.png".format(selAxes))

### Plot objective values
### 
### 

def printFnc(x):
    return x[0].reshape(n1,n2).cpu().T

def selFnc(x):
    return x[0, 1:]

def selFnc(x):
    try:
        if len(x.shape) > 1:
            return x[:, 1:].mean(0)
        elif len(x.shape) == 1:
            return x.mean(0).repeat(MaxIter - 1)
        else:
            return x.repeat(MaxIter - 1)
    except:
        return np.zeros(1).repeat(MaxIter-1)

namesOrdered = copy.deepcopy(descriptorsHere)
    
for d, e, epsVls in zip(outPsnrs, namesOrdered, outCritobj):
    dSel = selFnc(d)
    dFin = dSel[-1]
    dMax = dSel.max()
    epsVl = round(selFnc(epsVls)[-1], 2)
    print('Final pSNR: ', round(dFin, 2), ' max pSNR: ', round(dMax, 2), ', crit: ', epsVl ,', method: {0:60}'.format(e))

fig_size = (25,20)
plt.figure(figsize=fig_size)

plt.subplot(3,2,1)
for d in outPsnrs:
    plt.plot(selFnc(d))
plt.title('PSNR')

plt.subplot(3,2,2)
for d in outHfens:
    plt.plot(selFnc(d))
plt.ylim([0, 2])
plt.title('HFEN')

plt.subplot(3,2,3)
for d in outNetworkNrmses:
    plt.plot(selFnc(d))
plt.title('1st objective(output - input)')
plt.legend(namesOrdered)

plt.subplot(3,2,4)
for d in outL1Obj:
    plt.plot(selFnc(d))
plt.title('l1Obj')

plt.subplot(3,2,5)
for d in outTVobj:
    plt.plot(selFnc(d))
plt.title('TVObj')
plt.legend(namesOrdered)

plt.subplot(3,2,6)
for d in outCritobj:
    plt.semilogy(selFnc(d))
plt.title('Criterion')

plt.show()
plt.savefig("inferencePlots.png")
