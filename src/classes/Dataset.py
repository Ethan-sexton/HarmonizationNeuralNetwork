from torch.utils.data import Dataset
import torch
import numpy as np

def processSequence(sequenceData):
    sequences = []
    for sequence in sequenceData:
        sliceI = []
        for slice in sequence:
            slice = np.array(slice)  # Ensure slice is a NumPy array
            if np.sum(slice) != 1.0:
                print(f"Invalid slice detected: {slice}. Using default value 0.")
                index = 0  # Default to 0 for invalid slices
            else:
                indices = np.where(np.isclose(slice, 1.0))[0]  # Use np.isclose for comparison
                if len(indices) > 0:
                    index = indices[0]
                else:
                    print(f"No 1.0 found in slice: {slice}. Using default value 0.")
                    index = 0
            sliceI.append(index)
        sequences.append(sliceI)
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
