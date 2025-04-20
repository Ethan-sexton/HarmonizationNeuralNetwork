from classes.Trainer import Trainer
from DataProcessing import loadData
from model.HarmonizationNeuralNetwork import HarmonizationNeuralNetwork
import torch
from torch.utils.data import DataLoader 

BATCH_SIZE = 4
EPOCHS = 50

device = torch.device('cpu')
midiData, dataset = loadData()
dataLoader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
model = HarmonizationNeuralNetwork().to(device)
trainer = Trainer(model, dataLoader)

for epoch in range(1, EPOCHS + 1):
    for x_soprano, y_alto, y_tenor, y_bass in dataLoader:
        trainer.train(epoch)
        print(f'Epoch {epoch} completed')
model.eval()

