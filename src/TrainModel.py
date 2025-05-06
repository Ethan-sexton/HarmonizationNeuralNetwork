from classes.Trainer import Trainer
from DataProcessing import loadData
import torch
from torch.utils.data import DataLoader 
import matplotlib.pyplot as plt

def TrainModel(modelType, epochs, plot:bool):
    BATCH_SIZE = 4
    EPOCHS = epochs
    if modelType == 'RNN':
        from model.RHarmonizationNeuralNetwork import RHarmonizationNeuralNetwork
        model = RHarmonizationNeuralNetwork()
    elif modelType == 'FNN':
        from model.FHarmonizationNeuralNetwork import FHarmonizationNeuralNetwork
        model = FHarmonizationNeuralNetwork()
    device = torch.device('cpu')
    midiData, dataset = loadData()
    dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    modelIdentifier = model.__class__.__name__[0]
    model = model.to(device)
    trainer = Trainer(model, dl)

    dataCollect = []
    oracle = []
    baseline = []
    for epoch in range(1, EPOCHS + 1):
        toPlot = trainer.train(epoch)
        dataCollect.append(toPlot[0])
        oracle.append(toPlot[1])
        baseline.append(toPlot[2])
        print(f'Epoch {epoch} completed')
    model.eval()
    totalLossCollec, altoLossCollec, tenorLossCollec, bassLossCollec, totalCorrectCollec, altoCorrectCollec, tenorCorrectCollec, bassCorrectCollec = zip(*dataCollect)
    if plot:
    # Flatten data for graphing
        totalLossX, totalLossY = zip(*[item for sublist in totalLossCollec for item in sublist])
        altoLossX, altoLossY = zip(*[item for sublist in altoLossCollec for item in sublist])
        tenorLossX, tenorLossY = zip(*[item for sublist in tenorLossCollec for item in sublist])
        bassLossX, bassLossY = zip(*[item for sublist in bassLossCollec for item in sublist])

        totalCorrectX, totalCorrectY = zip(*[item for sublist in totalCorrectCollec for item in sublist])
        altoCorrectX, altoCorrectY = zip(*[item for sublist in altoCorrectCollec for item in sublist])
        tenorCorrectX, tenorCorrectY = zip(*[item for sublist in tenorCorrectCollec for item in sublist])
        bassCorrectX, bassCorrectY = zip(*[item for sublist in bassCorrectCollec for item in sublist])
        oracleX, oracleY = zip(*oracle)
        baselineX, baselineY = zip(*[item for sublist in baseline for item in sublist])
        print(modelIdentifier  )
        # Plot Loss
        figLoss, axLoss = plt.subplots()
        axLoss.plot(totalLossX, totalLossY, label='Total Loss', color='blue')
        axLoss.plot(altoLossX, altoLossY, label='Alto Loss', color='orange')
        axLoss.plot(tenorLossX, tenorLossY, label='Tenor Loss', color='green')
        axLoss.plot(bassLossX, bassLossY, label='Bass Loss', color='red')
        axLoss.legend()
        plt.xlabel('Epoch Number')
        plt.ylabel('Average Loss')
        plt.title('Average Loss per Epoch')
        plt.savefig(f'src\output\plots\{modelIdentifier}{EPOCHS}-total-loss.png')

        # Plot Correct Predictions
        plt.clf()
        figCorrect, axCorrect = plt.subplots()
        axCorrect.plot(totalCorrectX, totalCorrectY, label='Total Correct', color='blue')
        axCorrect.plot(altoCorrectX, altoCorrectY, label='Alto Correct', color='orange')
        axCorrect.plot(tenorCorrectX, tenorCorrectY, label='Tenor Correct', color='green')
        axCorrect.plot(bassCorrectX, bassCorrectY, label='Bass Correct', color='red')
        axCorrect.legend()
        plt.xlabel('Epoch Number')
        plt.ylabel('Average Correct Predictions')
        plt.title('Average Correct Predictions per Epoch')
        plt.savefig(f'src\output\plots\{modelIdentifier}{EPOCHS}-total-correct.png')

        plt.clf()
        # Plot Oracle vs Baseline vs Model
        figCorrect, axCorrect = plt.subplots()
        axCorrect.plot(oracleX, oracleY, label='Oracle', color='blue')
        axCorrect.plot(baselineX, baselineY, label='Baseline', color='orange')
        axCorrect.plot(totalCorrectX, totalCorrectY, label='Model', color='green')
        axCorrect.legend()
        plt.xlabel('Epoch Number')
        plt.ylabel('Average Correct Predictions')
        plt.title('Oracle vs Baseline vs Model')
        plt.savefig(f'src\output\plots\{modelIdentifier}{EPOCHS}-oracle-baseline-model.png')

    filename = ''
    if EPOCHS > 1:
        filename = f'{modelIdentifier}harmonizer-{EPOCHS}-Epochs.pth'
    else:
        filename = f'{modelIdentifier}harmonizer-{EPOCHS}-Epoch.pth'

    torch.save(model.state_dict(), f'src\output\models\{filename}')
