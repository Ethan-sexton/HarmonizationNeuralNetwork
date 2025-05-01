from classes.Trainer import Trainer
from DataProcessing import loadData
from model.HarmonizationNeuralNetwork import HarmonizationNeuralNetwork
import torch
from torch.utils.data import DataLoader 
import matplotlib.pyplot as plt

BATCH_SIZE = 4
EPOCHS = 500

device = torch.device('cpu')
midiData, dataset = loadData()
dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
model = HarmonizationNeuralNetwork().to(device)
trainer = Trainer(model, dl)

dataCollect = []
for epoch in range(1, EPOCHS + 1):
    toPlot = trainer.train(epoch)
    dataCollect.append(toPlot)
    print(f'Epoch {epoch} completed')
model.eval()

totalLossCollec, altoLossCollec, tenorLossCollec, bassLossCollec, totalCorrectCollec, altoCorrectCollec, tenorCorrectCollec, bassCorrectCollec = zip(*dataCollect)

# Flatten data for graphing
totalLossX, totalLossY = zip(*[item for sublist in totalLossCollec for item in sublist])
altoLossX, altoLossY = zip(*[item for sublist in altoLossCollec for item in sublist])
tenorLossX, tenorLossY = zip(*[item for sublist in tenorLossCollec for item in sublist])
bassLossX, bassLossY = zip(*[item for sublist in bassLossCollec for item in sublist])

totalCorrectX, totalCorrectY = zip(*[item for sublist in totalCorrectCollec for item in sublist])
altoCorrectX, altoCorrectY = zip(*[item for sublist in altoCorrectCollec for item in sublist])
tenorCorrectX, tenorCorrectY = zip(*[item for sublist in tenorCorrectCollec for item in sublist])
bassCorrectX, bassCorrectY = zip(*[item for sublist in bassCorrectCollec for item in sublist])

# Plot Loss
figLoss, axLoss = plt.subplots()
axLoss.plot(totalLossX, totalLossY, label='Total Loss', color='blue')
axLoss.plot(altoLossX, altoLossY, label='Alto Loss', color='orange')
axLoss.plot(tenorLossX, tenorLossY, label='Tenor Loss', color='green')
axLoss.plot(bassLossX, bassLossY, label='Bass Loss', color='red')
axLoss.legend()
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.title('Loss per Epoch')
plt.savefig('total-loss.png')

# Plot Correct Predictions
plt.clf()
figCorrect, axCorrect = plt.subplots()
axCorrect.plot(totalCorrectX, totalCorrectY, label='Total Correct', color='blue')
axCorrect.plot(altoCorrectX, altoCorrectY, label='Alto Correct', color='orange')
axCorrect.plot(tenorCorrectX, tenorCorrectY, label='Tenor Correct', color='green')
axCorrect.plot(bassCorrectX, bassCorrectY, label='Bass Correct', color='red')
axCorrect.legend()
plt.xlabel('Epoch Number')
plt.ylabel('Correct Predictions')
plt.title('Correct Predictions per Epoch')
plt.savefig('total-correct.png')

torch.save(model.state_dict(), 'harmonizer.pth')
