import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import matplotlib.image as mpimg
import warnings
warnings.filterwarnings('ignore')
import argparse

import csv
import progressbar
from PIL import Image, ImageTk
import skimage.transform

import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim
import torch_optimizer as optimNew

import torchvision
from torchvision import datasets,transforms 
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import models

from sklearn.metrics import roc_auc_score
import sklearn.metrics as metrics
import random

from CheXpertDataSet import *

parser = argparse.ArgumentParser(description='PyTorch Benchmarking')
parser.add_argument('--doTrain',   '-t',           default=False, required=False, help="Run Training",             action='store_true')
parser.add_argument('--doTest',    '-v',           default=False, required=False, help="Run Validation Test",      action='store_true')
parser.add_argument('--batchSize', '-b', type=int, default=4,     required=False, help='Batch Size')
parser.add_argument('--batchLimit','-l', type=int, default=0,     required=False, help='Batch Limit')
parser.add_argument('--epochs',    '-e', type=int, default=10,    required=False, help='Num of Epochs')
parser.add_argument('--clean',     '-c',           default=False, required=False, help='Start with clean Network', action='store_true')
args = parser.parse_args()
trBatchSize = args.batchSize
trMaxEpoch = args.epochs
trBatchLimit = args.batchLimit # 0 means to the full scale of the training set
doTrain = args.doTrain
doTest = args.doTest
doClean = args.clean # not yet in use


use_gpu = torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

#pathFileTrain = '../CheXpert-v1.0-small/train.csv'
#pathFileValid = '../CheXpert-v1.0-small/valid.csv'
pathFileTrain = 'Data_Train.csv'
pathFileValid = 'Data_Valid.csv'


# Parameters related to image transforms: size of the down-scaled image, cropped image
#imgtransResize = (320, 320)
#imgtransCrop = 224
imgtransResize = (512, 512)
imgtransCrop = 512
#imgtransCrop = 320

# Class names new classes = 2
#class_names = ['No Finding','Atelectasis','Cardiomegaly','Consolidation','Edema','Effusion','Emphysema','Fibrosis','Hernia','Infiltration','Mass','Nodule','Pleural_Thickening','Pneumonia','Pneumothorax']
class_names = ['No Finding','Covid']

# Neural network parameters:
nnIsTrained = True                 #pre-trained using ImageNet
nnClassCount = len(class_names)     #dimension of the output
               

#TRANSFORM DATA

normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transformList = []
#transformList.append(transforms.Resize(imgtransCrop))
transformList.append(transforms.RandomResizedCrop(imgtransCrop))
transformList.append(transforms.RandomHorizontalFlip())
transformList.append(transforms.ToTensor())
transformList.append(normalize)      
transformSequence=transforms.Compose(transformList)

#LOAD DATASET
print('Defining DataSets')

dataset = CheXpertDataSet(pathFileTrain ,transformSequence, policy="ones")
#datasetTest, datasetTrain = random_split(dataset, [5000, len(dataset) - 5000])
datasetTest, datasetTrain = random_split(dataset, [50, len(dataset) - 50])
datasetValid = CheXpertDataSet(pathFileValid, transformSequence)            

print('Defining DataLoader')
dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True)
dataLoaderVal = DataLoader(dataset=datasetValid, batch_size=trBatchSize, shuffle=False)
dataLoaderTest = DataLoader(dataset=datasetTest, num_workers=0)

img_path, images, classes = next(iter(dataLoaderTrain))
#print(images.size())
#print(classes.size())

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225] 
def gridshow(inp, title):

    inp = inp.cpu().numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    
    plt.figure (figsize = (12, 6))

    plt.imshow(inp)
    plt.title(title)
#    plt.pause(5)  
    plt.show()  
    
#out = torchvision.utils.make_grid(images)
#gridshow(out, title="sample")
#print(classes[1])

#exit()
# check first if data is readable, then remove exit call

checkpt = 'ResNextTrained.pth'

#model = models.alexnet(pretrained=True)
#model = models.resnext101_32x8d (pretrained=True)
model = models.resnext50_32x4d (pretrained=nnIsTrained)
#print(model)
# alexNet 
# num_ftrs = model.classifier[6].in_features
#Redefining the last layer to classify inputs into the 14+1 classes we need as opposed to the original 1000 it was trained for.
# model.classifier[6] = nn.Linear(num_ftrs, len(class_names))
#resnext
num_ftrs = model.fc.in_features
#Redefining the last layer to classify inputs into the 14+1 classes we need as opposed to the original 1000 it was trained for.
model.fc = nn.Linear(num_ftrs, len(class_names))

#criterion   = nn.CrossEntropyLoss()
criterion = nn.BCELoss(size_average = True)
# SETTINGS: OPTIMIZER & SCHEDULER
#optimizer = optim.Adam (model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
#optimizer = optimNew.RAdam (model.parameters(), lr= 1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
#optimizer = optimNew.AdaBound(model.parameters(), lr= 1e-3, betas= (0.9, 0.999), gamma=1e-3, eps= 1e-8, weight_decay=0, amsbound=False)
#optimizer = optim.AdaMod(model.parameters(), lr= 1e-3, betas=(0.9, 0.999), beta3=0.999, eps=1e-8, weight_decay=0)
#optimizer = optim.DiffGrad(model.parameters(), lr= 1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
optimizer   = torch.optim.SGD(model.parameters(), lr=0.001, momentum = 0.9)

def train_model(model, checkpoint, criterion, optimizer, num_epochs=25, maxBatch=50, maxEval=10):

    lossTrain = []
    lossEval = []
    dataStep = 1
    if maxBatch>100:
        dataStep = round(maxBatch/100)
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime

    f_csv = open('LossOutput'+timestampLaunch+'.csv','a', newline='')
    fw_csv = csv.writer(f_csv)

    if checkpoint != None and use_gpu:
        if os.path.isfile(checkpoint):
            model.load_state_dict(torch.load(checkpoint))
            print ("Loading checkpoint ", checkpoint)
        else:
            print ("No checkpoint - using fresh net")

    model = model.to(device)

    lossMIN = 100000
    
    for epoch in range(num_epochs):
        print('Epoch = ', epoch, " Start Training")

        # train model for this epoch
        model.train()
        
        maxStep = maxBatch
        bar = progressbar.ProgressBar(max_value=maxStep)
        nBatch=0
        lossVal = 0
        for img_path, images, labels  in (dataLoaderTrain):
            bar.update(nBatch)
            #print(".",end='')
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            outputs = outputs.to(device)
            loss = criterion(torch.sigmoid(outputs),labels)            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lv = loss.item()
            lossVal += lv
            if nBatch%dataStep==0:
                fw_csv.writerow((epoch, nBatch, lv))
            nBatch += 1
            if nBatch > maxStep:
                break
               
        bar.finish()
        
        avgLossT = lossVal/nBatch
        lossTrain.append(avgLossT)
        print('Average loss: %0.5f '%(avgLossT),end='')
        #evaluate model improvements for this epoch
        model.eval()
        nBatch=0
        lossVal = 0
        print(" Evaluating")
        
        maxStep=maxEval
        bar = progressbar.ProgressBar(max_value=maxStep)

        with torch.no_grad():
            for img_path, images, labels  in (dataLoaderVal):
                bar.update(nBatch)
                #print(".",end='')
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                outputs = outputs.to(device)
                loss = criterion(torch.sigmoid(outputs),labels)
                lv = loss.item()
                lossVal += lv
                nBatch += 1
                if nBatch > maxStep:
                    break
        
        bar.finish()
        
        avgLossE = lossVal/nBatch
        lossEval.append(avgLossE)
        fw_csv.writerow((epoch, avgLossT, avgLossE))
        print('Eval avg. loss - %0.5f '%(avgLossE))
        print("Saving model")
        torch.save(model.state_dict(), checkpoint)

    f_csv.close()

    return model, lossTrain, lossEval
    
def test_model(model, checkpoint, class_names, maxBatch=50):
    cudnn.benchmark = True
    
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime

    f_csv = open('TestOutput'+timestampLaunch+'.csv','w', newline='')
    fw_csv = csv.writer(f_csv)

    if checkpoint != None and use_gpu:
        model.load_state_dict(torch.load(checkpoint))
        model = model.to(device)

    model.eval()
    nBatch=0
    lossVal = 0
    accVal = [0 for i in range(len(class_names))]
    failVal = [0 for i in range(len(class_names))]
    confVal = [0 for i in range(len(class_names))]
    weakVal = [0 for i in range(len(class_names))]
    accTotal = 0
    failTotal = 0
    confTotal = 0
    weakTotal = 0
    TPVal = [0 for i in range(len(class_names))]
    TNVal = [0 for i in range(len(class_names))]
    FPVal = [0 for i in range(len(class_names))]
    FNVal = [0 for i in range(len(class_names))]
    outTruth = []
    outPred  = []
    print(" Testing")
    maxStep = maxBatch
    bar = progressbar.ProgressBar(max_value=maxStep)
    with torch.no_grad():
        for img_path, images, labels  in (dataLoaderTest):
            bar.update(nBatch)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            outputs = outputs.to(device)

            sglTruth = np.array(labels[0].cpu(), dtype='int')
            sglPred  = np.array(torch.sigmoid(outputs[0]).cpu(), dtype='float')
#            fw_csv.writerow(sglPred)
            for i in range(len(class_names)):
                if sglTruth[i]==0:
                    if sglPred[i]<=0.5:
                        # True Negative
                        TNVal[i]  += 1
                        accVal[i]  += 1
                        accTotal   += 1
                        confVal[i] += 1-sglPred[i]
                        confTotal  += 1-sglPred[i]
                    else:
                        # False Positive
                        FPVal[i]  += 1
                        failVal[i] += 1
                        failTotal  += 1
                        weakVal[i] += 1-sglPred[i]
                        weakTotal  += 1-sglPred[i]
                if sglTruth[i]==1:
                    if sglPred[i]>=0.5:
                        # True Positive
                        TPVal[i]  += 1
                        accVal[i]  += 1
                        accTotal   += 1
                        confVal[i] += sglPred[i]
                        confTotal  += sglPred[i]
                    else:
                        # Flase Negative
                        FNVal[i]  += 1
                        failVal[i] += 1
                        failTotal  += 1
                        weakVal[i] += sglPred[i]
                        weakTotal  += sglPred[i]
                        
            if nBatch==0:
                outTruth = sglTruth
                outPred = sglPred
            else:
                outTruth = np.vstack((outTruth,sglTruth))
                outPred  = np.vstack((outPred, sglPred))
            roc_auc = 0 # tbd
            nBatch += 1
            if nBatch > maxBatch:
                nBatch -= 1
                break

    bar.finish()
    
    print("\n==============================================================")
    if nBatch > 0:
        accTotal = accTotal/(nBatch*len(class_names))
        failTotal = failTotal/(nBatch*len(class_names))
        confTotal = confTotal/(nBatch*len(class_names))
        weakTotal = weakTotal/(nBatch*len(class_names))
        fw_csv.writerow(("Accuracy", accTotal, failTotal, confTotal, weakTotal))
        print("Accuracy/Fail/Conf/Weak : ",accTotal, failTotal, confTotal, weakTotal, nBatch)
        for i in range(len(class_names)):
            if accVal[i] > 0:
                confVal[i] = confVal[i]/accVal[i]
            else:
                confVal[i] = 0
            if failVal[i] > 0:
                weakVal[i] = weakVal[i]/failVal[i]
            else:
                weakVal[i] = 0
            accVal[i] = accVal[i]/nBatch
            failVal[i] = failVal[i]/nBatch
            fw_csv.writerow((class_names[i], accVal[i], failVal[i], confVal[i], weakVal[i], TPVal[i], TNVal[i], FPVal[i], FNVal[i] ))
            print(class_names[i],accVal[i],failVal[i], confVal[i], weakVal[i])

    f_csv.close()

    return outTruth, outPred
    

    
trMaxBatch = len(datasetTrain)//trBatchSize
trMaxEval = len(datasetValid)//trBatchSize
#debug limit batch runs
if trBatchLimit>0: trMaxBatch = trBatchLimit
print("batch size      :", trBatchSize)
print("Max# of epochs  :", trMaxEpoch)
print("Max# of batches :", trMaxBatch)
print("Max# of evals   :", trMaxEval)

if doTrain:
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime

    model, lTrain, lEval = train_model(model, checkpt, criterion, optimizer, num_epochs=trMaxEpoch, maxBatch=trMaxBatch, maxEval=trMaxEval)
    batch = [i for i in range(len(lTrain))]
    plt.plot(batch, lTrain, label = "train")
    plt.plot(batch, lEval, label = "eval")
    plt.xlabel("Nb of batches")
    plt.ylabel("BCE loss")
    plt.title("BCE loss evolution")
    plt.legend()
    plt.savefig('BCEloss'+timestampLaunch+'.png', dpi=600)
    #plt.pause(5)  
    #plt.show()

model.eval()

trMaxBatch = len(datasetTest)
if doTest:
    truth, predict = test_model(model, checkpt, class_names, maxBatch=trMaxBatch)

    aucAvg = 0 
    aucAvgW = 0
    aucAvgN = 0
    fig= plt.figure(figsize=(19,9.5))

    for i in range(nnClassCount):
        fpr, tpr, thresholds = metrics.roc_curve(truth[:,i], predict[:,i], pos_label=1)
        timestampTime = time.strftime("%H%M%S")
        timestampDate = time.strftime("%d%m%Y")
        timestampLaunch = timestampDate + '-' + timestampTime
        f_csv = open('ROC'+class_names[i]+timestampLaunch+'.csv','w', newline='')
        fw_csv = csv.writer(f_csv)
        fw_csv.writerow(fpr)
        fw_csv.writerow(tpr)
        f_csv.close()
        roc_auc = metrics.auc(fpr, tpr)
        aucAvg += roc_auc
        aucAvgN += len(thresholds)
        aucAvgW += roc_auc*len(thresholds)
        print(class_names[i], roc_auc, len(thresholds))
        f = plt.subplot(2, 8, i+1)
        plt.title(class_names[i])

        plt.plot(fpr, tpr, label = 'AUC = %0.2f' % roc_auc)

        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')

    aucAvg = aucAvg/nnClassCount
    aucAvgW = aucAvgW/aucAvgN
    print("Average AUROC : ", aucAvg)
    print("Weighted Average AUROC : ", aucAvgW)
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime
    
    plt.savefig('auroc'+timestampLaunch+'.png', dpi=600)
    #plt.pause(5)  
    plt.show()

# heatmap
class SaveFeatures(): 
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)  # attach the hook to the specified layer
    def hook_fn(self, module, hinput, houtput): self.features = ((houtput.cpu()).data).numpy() # copy the activation features as an instance variable
    def remove(self): self.hook.remove()
    

print ("Loading checkpoint ", checkpt)
model.load_state_dict(torch.load(checkpt))
model = model.to(device)

final_layer = model._modules.get('layer4') # Grab the final layer of the model
activated_features = SaveFeatures(final_layer) # attach the call back hook to the final layer of the model

print("=========================================")
print(" hit q to exit")
print("=========================================")

imgSample = iter(dataLoaderTest)
while True:
    print("reading next image")
    img_path, images, classes = next(imgSample)
    ld_path = img_path[0]
    print(img_path)
    print(ld_path)
    images = images.to(device)

    output = model(images)
    sglPred  = torch.sigmoid(output[0]).cpu()
    sglTruth = classes[0].cpu()

    truthLabel = "Truth:"
    prediLabel = "Preticd:"
    maxPred = 0
    idxPred = 0
    lblCount = 0
    for i in range(nnClassCount):
        if sglPred[i]>maxPred:
            maxPred = sglPred[i]
            idxPred = i
        if sglPred[i]>0.4:
            prediLabel = prediLabel + " " + class_names[i]
            lblCount += 1
        if sglTruth[i]==1:
            truthLabel = truthLabel + " " + class_names[i]

    if lblCount==0:
        prediLabel = prediLabel + " " + class_names[idxPred] + " (WEAK)"
    print(truthLabel)
    print(prediLabel)

    output = torch.sigmoid(output.squeeze())
    #print("Features shape:",activated_features.features.shape)

    #print("Prediction: ", output)
    pred_probabilities = F.softmax(output).data.squeeze() # Pass the predictions through a softmax layer to convert into probabilities for each class
    #print("Truth     Class: ", classes)
    #print("Predicted Class: ", pred_probabilities)

    def getCAM(feature_conv, weight_fc, class_idx):
        _, nc, h, w = feature_conv.shape
    #    print(feature_conv.shape)
        cam = weight_fc.dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        return [cam_img]
        
    weight_softmax_params = list(model._modules.get('fc').parameters()) # This gives a list of weights for the fully connected layers 
    weight_softmax = weight_softmax_params[0].cpu().data.numpy() # what does this do ??

    #cam_img = getCAM(activated_features.features, weight_softmax[0], pred_probabilities )
    cam_img = getCAM(activated_features.features, weight_softmax[0], sglPred )

    rawimg = mpimg.imread(ld_path)
    fig, ax = plt.subplots(1,3, figsize=(15,5))

    ax[0].imshow(rawimg, cmap='gray')
    ax[0].set_title(truthLabel)
    ax[1].set_title(prediLabel)
    ax[1].imshow(rawimg, cmap='gray')
    ax[1].imshow(skimage.transform.resize(cam_img[0], (rawimg.shape[0],rawimg.shape[1] )), alpha=0.25, cmap='jet')
    y_pos = np.arange(nnClassCount)
    bar_w = 0.35
    ax[2].barh(y_pos+bar_w/2,sglPred.detach().numpy(), bar_w, label='Prediction')
    ax[2].barh(y_pos-bar_w/2,sglTruth.detach().numpy(), bar_w, label='Truth')
    ax[2].set_yticks(y_pos)
    ax[2].set_yticklabels(class_names)
    ax[2].legend()
    fig.tight_layout()
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime
    plt.savefig('heatmap'+timestampLaunch+'.png', dpi=600)
    plt.show()
    
    keyb = input()
    if keyb=="q":
        break
        