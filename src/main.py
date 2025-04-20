from classes.Trainer import Trainer
from DataProcessing import loadData
from model.HarmonizationNeuralNetwork import HarmonizationNeuralNetwork
import torch
from torch.utils.data import DataLoader 
import matplotlib as plt

BATCH_SIZE = 4
EPOCHS = 50

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
torch.save(model.state_dict(), 'harmonizer.pth')
