import torch.nn as nn
import torch.nn.functional as F
import Constants 
SEQUENCE_LENGTH = Constants.SEQUENCE_LENGTH
BATCH_SIZE = Constants.BATCH_SIZE
SOPRANO_CLASS_NUMBER = Constants.SOPRANO_CLASS_NUMBER
ALTO_CLASS_NUMBER = Constants.ALTO_CLASS_NUMBER
TENOR_CLASS_NUMBER = Constants.TENOR_CLASS_NUMBER
BASS_CLASS_NUMBER = Constants.BASS_CLASS_NUMBER
DROPOUT_RATE = Constants.DROPOUT_RATE

class RHarmonizationNeuralNetwork(nn.Module):
    def __init__(self, input_size=SOPRANO_CLASS_NUMBER, hidden_size=200, num_layers=2, dropout=DROPOUT_RATE):
        super(RHarmonizationNeuralNetwork, self).__init__()

        # Input fully connected layer
        self.input_fc = nn.Linear(input_size, 128)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

        # Fully connected layers for each voice
        self.alto_fc = nn.Linear(hidden_size, ALTO_CLASS_NUMBER)
        self.tenor_fc = nn.Linear(hidden_size, TENOR_CLASS_NUMBER)
        self.bass_fc = nn.Linear(hidden_size, BASS_CLASS_NUMBER)

    def forward(self, x):
        # Reshape input to (batch_size, sequence_length, input_size)
        x = x.view(-1, SEQUENCE_LENGTH, SOPRANO_CLASS_NUMBER)

        # Pass through input fully connected layer with ReLU activation
        x = F.tanh(self.input_fc(x))
        # Pass through LSTM
        x, _ = self.lstm(x)


        # Apply dropout
        x = self.dropout(x)

        # Pass the output of the LSTM through fully connected layers for each voice
        alto_output = self.alto_fc(x)
        tenor_output = self.tenor_fc(x)
        bass_output = self.bass_fc(x)

        # Apply softmax to get probabilities
        alto_output = F.log_softmax(alto_output, dim=2)
        tenor_output = F.log_softmax(tenor_output, dim=2)
        bass_output = F.log_softmax(bass_output, dim=2)

        return alto_output, tenor_output, bass_output