import torch
import torch.nn as nn
import torch.nn.functional as F
import Constants
SEQUENCE_LENGTH = Constants.SEQUENCE_LENGTH
BATCH_SIZE = Constants.BATCH_SIZE
CLASS_NUMBER = Constants.CLASS_NUMBER

class HarmonizationNeuralNetwork(nn.Module):
    def __init__(self):
        super(HarmonizationNeuralNetwork, self).__init__()

        self.input = nn.Linear(SEQUENCE_LENGTH * CLASS_NUMBER, 200)
        self.hidden1 = nn.Linear(200, 200)
        self.dropout1 = nn.Dropout(0.2)
        self.hidden2 = nn.Linear(200, 200)
        self.drouput2 = nn.Dropout(0.2)

        self.forwardAlto = nn.Linear(200, SEQUENCE_LENGTH * CLASS_NUMBER)
        self.forwardTenor = nn.Linear(200, SEQUENCE_LENGTH * CLASS_NUMBER)
        self.forwardBass = nn.Linear(200, SEQUENCE_LENGTH * CLASS_NUMBER)

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
        x_alto = torch.reshape(x_alto, (BATCH_SIZE, CLASS_NUMBER, SEQUENCE_LENGTH))
        y_alto = F.log_softmax(x_alto, dim=1)

        #Tenor Voice:
        x_tenor = self.forwardTenor(x)
        x_tenor = F.relu(x_tenor)
        x_tenor = torch.reshape(x_tenor, (BATCH_SIZE, CLASS_NUMBER, SEQUENCE_LENGTH))
        y_tenor = F.log_softmax(x_tenor, dim=1)

        #Bass Voice:
        x_bass = self.forwardBass(x)
        x_bass = F.relu(x_bass)
        x_bass = torch.reshape(x_bass, (BATCH_SIZE, CLASS_NUMBER, SEQUENCE_LENGTH))
        y_bass = F.log_softmax(x_bass, dim=1)

        return y_alto, y_tenor, y_bass