from classes.Range import Range
import pandas as pd
import torch

class Voice:
    def __init__(self, name: str, dataIndex: int, ran: Range):
        self.name = name
        self.dataIndex = dataIndex
        self.range = ran
        self.ENCODE_KEY = generateKey(ran.getRange()[0], ran.getRange()[1])
        print(self.ENCODE_KEY)

    def getPartList(self, df: pd.DataFrame, songIndex: int):
        print("Get part list entered")
        song = df[songIndex]
        print(f"Song: {song}")
        notes = []
        if isinstance(song, list):
            for note in range(len(song)):
                notes.append(song[note][self.dataIndex])
        print(f"Notes: {notes}")
        return notes

    def encodePart(self, part: list):
        print("encode part entered")
        encodedPart = []
        for item in part:
            if item == -1 or item is None or item == 0:
                encodedPart.append([0] * len(self.ENCODE_KEY))
            else:
                encodedPart.append(self.ENCODE_KEY[item])
        return encodedPart
    
    def decodePart(self, part: list):
        decodedPart = []
        for item in part:
            decodedPart.append(keyFromValue(self.ENCODE_KEY, item))
        return decodedPart

    def prepPart(self, df: pd.DataFrame, songIndex: int):
        print("prep part entered")
        #Returns an encoded tensor for specified song
        part = self.getPartList(df, songIndex)
        part = self.encodePart(part)
        songTensor = torch.tensor(part)
        return songTensor

def generateKey(min, max):
    listOfNums = list(range(min, max))
    key = {}
    key.update({1 : [0] * len(listOfNums)})
    for nums in range(len(listOfNums)):
        print(nums)
        copy = [0] * len(listOfNums)
        copy[nums] = 1
        key[listOfNums[nums]] = copy
    return key

def keyFromValue(dict, neededValue):
    for key, value in dict:
        if value == neededValue:
            return key
    raise ValueError(f"{neededValue} not found in dictionary {dict}")