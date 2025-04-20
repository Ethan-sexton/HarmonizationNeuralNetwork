import DataProcessing
from model.HarmonizationNeuralNetwork import HarmonizationNeuralNetwork
import random as rd
import torch    
import torch.nn as F
from torch.utils.data import DataLoader
import torch.optim as optim
import Constants

device = torch.device("cpu") #Could allow GPU utilization, but my laptop cries enough as is
LEARNING_RATE = Constants.LEARNING_RATE
BATCH_SIZE = Constants.BATCH_SIZE
SEQUENCE_LENGTH = Constants.SEQUENCE_LENGTH
CLASS_NUMBER = Constants.CLASS_NUMBER

class Trainer:
    def __init__(self, model: HarmonizationNeuralNetwork, dataLoader: DataLoader):
        self.model = model
        self.dataLoader = dataLoader
        self.lossFunction = F.NLLLoss() 
        
        self.optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    def calculateLoss(self, y_prediction, y_target):
        return self.lossFunction(y_prediction, y_target)
    
    def sumCorrectPredictions(self, y_prediction, y_target):
        #Returns the total number of correctly predicted notes
        totalCorrect = 0
        modelT = y_prediction[0].transpose(0,1)
        for i in range(SEQUENCE_LENGTH):
            modelPred = torch.argmax(modelT[i])
            targetPred = y_target[0][i]

            if modelPred == targetPred:
                totalCorrect += 1
            
        return totalCorrect

    def train(self, epoch):

        altoCorrectCollec = []
        tenorCorrectCollec = []
        bassCorrectCollec = []
        altoLossCollec = []
        tenorLossCollec = []
        bassLossCollec = []
        totalLossCollec = []
        totalCorrectCollec = []

        self.model.train()
        for i, (x_soprano, y_alto, y_tenor, y_bass) in enumerate(self.dataLoader):
            if len(x_soprano) < BATCH_SIZE:
                continue
            x_soprano = x_soprano.to(device)
            y_alto = y_alto.to(device)
            y_tenor = y_tenor.to(device)
            y_bass = y_bass.to(device)

            self.optimizer.zero_grad()
            predAlto, predTenor, predBass = self.model(x_soprano)

            predAlto = torch.reshape(predAlto, (BATCH_SIZE, CLASS_NUMBER, SEQUENCE_LENGTH))
            predTenor = torch.reshape(predTenor, (BATCH_SIZE, CLASS_NUMBER, SEQUENCE_LENGTH))
            predBass = torch.reshape(predBass, (BATCH_SIZE, CLASS_NUMBER, SEQUENCE_LENGTH))           

            lossAlto = self.calculateLoss(predAlto, y_alto)
            lossTenor = self.calculateLoss(predTenor, y_tenor)
            lossBass = self.calculateLoss(predBass, y_bass)

            lossTotal = lossAlto + lossTenor + lossBass

            altoLossCollec.append([lossAlto, epoch])
            tenorLossCollec.append([lossTenor, epoch])
            bassLossCollec.append([lossBass, epoch])
            totalLossCollec.append([lossTotal, epoch])

            lossTotal.backward()
            self.optimizer.step()

            if i % 100 == 0:
                altoCorrect = self.sumCorrectPredictions(predAlto, y_alto)
                tenorCorrect = self.sumCorrectPredictions(predTenor, y_tenor)
                bassCorrect = self.sumCorrectPredictions(predBass, y_bass)
                totalCorrect = altoCorrect + tenorCorrect + bassCorrect

                altoCorrectCollec.append([altoCorrect, epoch])
                tenorCorrectCollec.append([tenorCorrect, epoch])
                bassCorrectCollec.append([bassCorrect, epoch])
                totalCorrectCollec.append([totalCorrect, epoch])

            currentItem = i * len(x_soprano)
            print(f"Epoch #{epoch} \n Loss: {lossTotal}\n Current Item: {currentItem} \nTotal Correct Predictions: {totalCorrect}\n")
        return totalLossCollec, altoLossCollec, tenorLossCollec, bassLossCollec, altoCorrectCollec, tenorCorrectCollec, bassCorrectCollec, totalCorrectCollec
        
