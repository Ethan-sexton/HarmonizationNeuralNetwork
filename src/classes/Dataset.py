from torch.utils.data import Dataset
import torch
import numpy as np

def processSequence(sequenceData):
    sequences = []
    for sequence in sequenceData:
        sliceI = []
        for slice in sequence:
            indices = np.where(slice == 1.0)[0]
            if len(indices) > 0:
                index = indices[0]
            else:
                # Handle the case where no 1.0 is found (e.g., use a default value like 0)
                index = 0
            sliceI.append(index)
        sequences.append(sliceI)
    print(sequences)
    return sequences
class ModelDataset(Dataset):
    def __init__(self, sequenceData: dict):
        self.x_soprano = torch.tensor(sequenceData['soprano'], dtype=torch.float32)
        self.y_alto = torch.tensor(processSequence(sequenceData['alto']), dtype=torch.long)
        self.y_tenor = torch.tensor(processSequence(sequenceData['tenor']), dtype=torch.long)
        self.y_bass = torch.tensor(processSequence(sequenceData['bass']), dtype=torch.long)
        self.length = min(len(sequenceData['soprano']), len(sequenceData['alto']),
                            len(sequenceData['tenor']), len(sequenceData['bass']))

    def __len__(self): 
        return self.length

    def __getitem__(self, index):
        sopF = self.x_soprano[index]
        altoF = self.y_alto[index]
        tenorF = self.y_tenor[index]
        bassF = self.y_bass[index]

        return (sopF, altoF, tenorF, bassF)   
