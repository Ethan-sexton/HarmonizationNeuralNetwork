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
        modelT = y_prediction.transpose(0,1)
        for i in range(SEQUENCE_LENGTH):
            modelPred = torch.argmax(modelT[i])
            targetPred = y_target[0][i]

            if modelPred == targetPred:
                totalCorrect += 1
            
        return totalCorrect

    def train(self, epoch):
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
            
            print(f"Shape of predictions: Alto: {predAlto.shape}, Tenor: {predTenor.shape}, Bass: {predBass.shape}")
            print(f"Shape of targets: Alto: {y_alto.shape}, Tenor: {y_tenor.shape}, Bass: {y_bass.shape}")
            print(f"Predictions: Alto: {predAlto} \nTenor: {predTenor} \nBass: {predBass}")
            print(f"Targets: Alto: {y_alto} \nTenor: {y_tenor} \nBass: {y_bass}")
            

            lossAlto = self.calculateLoss(predAlto, y_alto)
            lossTenor = self.calculateLoss(predTenor, y_tenor)
            lossBass = self.calculateLoss(predBass, y_bass)

            lossTotal = lossAlto + lossTenor + lossBass

            lossTotal.backward()
            self.optimizer.step()

            if i % 100 == 0:
                altoCorrect = self.sumCorrectPredictions(predAlto, y_alto)
                tenorCorrect = self.sumCorrectPredictions(predTenor, y_tenor)
                bassCorrect = self.sumCorrectPredictions(predBass, y_bass)

                totalCorrect = altoCorrect + tenorCorrect + bassCorrect

            print(f"Epock #{epoch} \n Loss: \n Total Correct Predictions: {totalCorrect}\n")

        
