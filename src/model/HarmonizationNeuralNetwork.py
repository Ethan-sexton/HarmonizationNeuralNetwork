import torch
import torch.nn as nn
import torch.nn.functional as F
import Constants as Constants
SEQUENCE_LENGTH = Constants.SEQUENCE_LENGTH
BATCH_SIZE = Constants.BATCH_SIZE
SOPRANO_CLASS_NUMBER = Constants.SOPRANO_CLASS_NUMBER
ALTO_CLASS_NUMBER = Constants.ALTO_CLASS_NUMBER
TENOR_CLASS_NUMBER = Constants.TENOR_CLASS_NUMBER
BASS_CLASS_NUMBER = Constants.BASS_CLASS_NUMBER
DROPUT_RATE = Constants.DROPOUT_RATE

class HarmonizationNeuralNetwork(nn.Module):
    def __init__(self):
        super(HarmonizationNeuralNetwork, self).__init__()

        self.input = nn.Linear(SEQUENCE_LENGTH * SOPRANO_CLASS_NUMBER, 200)
        self.hidden1 = nn.Linear(200, 200)
        self.dropout1 = nn.Dropout(DROPUT_RATE)
        self.hidden2 = nn.Linear(200, 200)
        self.drouput2 = nn.Dropout(DROPUT_RATE)

        self.forwardAlto = nn.Linear(200, SEQUENCE_LENGTH * ALTO_CLASS_NUMBER)
        self.forwardTenor = nn.Linear(200, SEQUENCE_LENGTH * TENOR_CLASS_NUMBER)
        self.forwardBass = nn.Linear(200, SEQUENCE_LENGTH * BASS_CLASS_NUMBER)

    def forward(self, x):
        #Architecture: 2 hidden layers
        x = torch.flatten(x, start_dim=1)

        x = self.input(x)
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.drouput2(x)

        #Alto Voice:
        x_alto = self.forwardAlto(x)
        x_alto = F.relu(x_alto)
        x_alto = torch.reshape(x_alto, (BATCH_SIZE, ALTO_CLASS_NUMBER, SEQUENCE_LENGTH))
        y_alto = F.log_softmax(x_alto, dim=1)

        #Tenor Voice:
        x_tenor = self.forwardTenor(x)
        x_tenor = F.relu(x_tenor)
        x_tenor = torch.reshape(x_tenor, (BATCH_SIZE, TENOR_CLASS_NUMBER, SEQUENCE_LENGTH))
        y_tenor = F.log_softmax(x_tenor, dim=1)

        #Bass Voice:
        x_bass = self.forwardBass(x)
        x_bass = F.relu(x_bass)
        x_bass = torch.reshape(x_bass, (BATCH_SIZE, BASS_CLASS_NUMBER, SEQUENCE_LENGTH))
        y_bass = F.log_softmax(x_bass, dim=1)

        return y_alto, y_tenor, y_bass