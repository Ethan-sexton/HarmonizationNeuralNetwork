from model.FHarmonizationNeuralNetwork import FHarmonizationNeuralNetwork
from model.RHarmonizationNeuralNetwork import RHarmonizationNeuralNetwork
from classes.Voices import Voices
from classes.Voice import Voice
import torch    
import torch.nn as F
from torch.utils.data import DataLoader
import torch.optim as optim
import Constants as Constants
import random as rand

device = torch.device("cpu") #Could allow GPU utilization, but my laptop cries enough as is

LEARNING_RATE = Constants.LEARNING_RATE
BATCH_SIZE = Constants.BATCH_SIZE
SEQUENCE_LENGTH = Constants.SEQUENCE_LENGTH
SOPRANO_CLASS_NUMBER = Constants.SOPRANO_CLASS_NUMBER
ALTO_CLASS_NUMBER = Constants.ALTO_CLASS_NUMBER
TENOR_CLASS_NUMBER = Constants.TENOR_CLASS_NUMBER
BASS_CLASS_NUMBER = Constants.BASS_CLASS_NUMBER

class Trainer:
    def __init__(self, model, dataLoader: DataLoader):
        self.model = model
        self.dataLoader = dataLoader
        if isinstance(model, FHarmonizationNeuralNetwork):
            self.lossFunction = F.NLLLoss() 
        elif isinstance(model, RHarmonizationNeuralNetwork):
            self.lossFunction = F.CrossEntropyLoss()
        else:
            raise ValueError("Model must be an instance of FHarmonizationNeuralNetwork or RHarmonizationNeuralNetwork")
        
        self.optimizer = optim.Adam(model.parameters(), lr=Constants.LEARNING_RATE)

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
    
    def createBaseline(self, voice: Voice):
        baseline = []
        for i in range(SEQUENCE_LENGTH):
            baseline.append(rand.randint(voice.range.lowestNote, voice.range.highestNote - 1))
        return baseline

    def train(self, epoch):

        altoCorrectCollec = []
        tenorCorrectCollec = []
        bassCorrectCollec = []
        altoLossCollec = []
        tenorLossCollec = []
        bassLossCollec = []
        totalLossCollec = []
        totalCorrectCollec = []

        altoEpochCorrect = []
        tenorEpochCorrect = []
        bassEpochCorrect = []
        totalEpochCorrect = []

        altoEpochLoss = []
        tenorEpochLoss = []
        bassEpochLoss = []
        totalEpochLoss = []
        totalBaseLine = []
        epochBaseline = []
        voices = Voices()
        alto = voices['alto']
        tenor = voices['tenor']
        bass = voices['bass']
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

            predAlto = torch.reshape(predAlto, (BATCH_SIZE, ALTO_CLASS_NUMBER, SEQUENCE_LENGTH))
            predTenor = torch.reshape(predTenor, (BATCH_SIZE, TENOR_CLASS_NUMBER, SEQUENCE_LENGTH))
            predBass = torch.reshape(predBass, (BATCH_SIZE, BASS_CLASS_NUMBER, SEQUENCE_LENGTH))           

            lossAlto = self.calculateLoss(predAlto, y_alto)
            lossTenor = self.calculateLoss(predTenor, y_tenor)
            lossBass = self.calculateLoss(predBass, y_bass)

            lossTotal = lossAlto + lossTenor + lossBass

            lossTotal.backward()
            self.optimizer.step()

            if i % 200 == 0:
                altoCorrect = self.sumCorrectPredictions(predAlto, y_alto)
                tenorCorrect = self.sumCorrectPredictions(predTenor, y_tenor)
                bassCorrect = self.sumCorrectPredictions(predBass, y_bass)
                totalCorrect = altoCorrect + tenorCorrect + bassCorrect

                altoEpochCorrect.append(altoCorrect)
                tenorEpochCorrect.append(tenorCorrect)
                bassEpochCorrect.append(bassCorrect)
                totalEpochCorrect.append(totalCorrect)
                
                altoEpochLoss.append(lossAlto.item())
                tenorEpochLoss.append(lossTenor.item())
                bassEpochLoss.append(lossBass.item())
                totalEpochLoss.append(lossTotal.item())

                altoBaseline = self.createBaseline(alto)
                tenorBaseline = self.createBaseline(tenor)
                bassBaseline = self.createBaseline(bass)
                altoBaseline = torch.reshape(torch.tensor(alto.encodePart(altoBaseline)), (1, ALTO_CLASS_NUMBER, SEQUENCE_LENGTH))
                tenorBaseline = torch.reshape(torch.tensor(tenor.encodePart(tenorBaseline)).to(device), (1, TENOR_CLASS_NUMBER, SEQUENCE_LENGTH))
                bassBaseline = torch.reshape(torch.tensor(bass.encodePart(bassBaseline)).to(device), (1, BASS_CLASS_NUMBER, SEQUENCE_LENGTH))

                altoBaseline = self.sumCorrectPredictions(altoBaseline, y_alto)
                tenorBaseline = self.sumCorrectPredictions(tenorBaseline, y_tenor)
                bassBaseline = self.sumCorrectPredictions(bassBaseline, y_bass)
                totalBaseline = altoBaseline + tenorBaseline + bassBaseline
                epochBaseline.append(totalBaseline)
            if i == len(x_soprano) - 1:
                altoAvLoss = sum(altoEpochLoss) / len(altoEpochLoss) if altoEpochLoss else 0
                tenorAvLoss = sum(tenorEpochLoss) / len(tenorEpochLoss) if tenorEpochLoss else 0
                bassAvLoss = sum(bassEpochLoss) / len(bassEpochLoss) if bassEpochLoss else 0
                totalAvLoss = sum(totalEpochLoss) / len(totalEpochLoss) if totalEpochLoss else 0

                altoLossCollec.append([epoch, altoAvLoss])
                tenorLossCollec.append([epoch, tenorAvLoss])
                bassLossCollec.append([epoch, bassAvLoss])
                totalLossCollec.append([epoch, totalAvLoss])
                
                altoAvCorrect = sum(altoEpochCorrect) / len(altoEpochCorrect) if altoEpochCorrect else 0
                tenorAvCorrect = sum(tenorEpochCorrect) / len(tenorEpochCorrect) if tenorEpochCorrect else 0
                bassAvCorrect = sum(bassEpochCorrect) / len(bassEpochCorrect) if bassEpochCorrect else 0
                totalAvCorrect = sum(totalEpochCorrect) / len(totalEpochCorrect) if totalEpochCorrect else 0

                avBaseline = sum(epochBaseline) / len(epochBaseline) if epochBaseline else 0
                totalBaseLine.append([epoch, avBaseline])

                altoCorrectCollec.append([epoch, altoAvCorrect])
                tenorCorrectCollec.append([epoch, tenorAvCorrect])
                bassCorrectCollec.append([epoch, bassAvCorrect])
                totalCorrectCollec.append([epoch, totalAvCorrect])
                
                altoEpochCorrect = []
                tenorEpochCorrect = []
                bassEpochCorrect = []
                totalEpochCorrect = []

                altoEpochLoss = []
                tenorEpochLoss = []
                bassEpochLoss = []
                totalEpochLoss = []
                epochBaseline = []

            oracleList = [epoch, 192]
        print(f"Epoch #{epoch} \n Av Loss: {totalAvLoss}\n Av Total Correct Predictions: {totalAvCorrect}\n")
        return [totalLossCollec, altoLossCollec, tenorLossCollec, bassLossCollec, totalCorrectCollec, altoCorrectCollec, tenorCorrectCollec, bassCorrectCollec], oracleList, totalBaseLine
        
