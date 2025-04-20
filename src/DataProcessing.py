import pickle
from classes.Voices import Voices
import torch
import Constants
from classes.Dataset import ModelDataset

def loadData():
    print("load data entered")
    with open('dataset\jsb-chorales-16th.pkl', 'rb') as file:
        midiData = pickle.load(file, encoding="latin1")
    
    sequentialData = dict()
    sequentialData['soprano'] = []
    sequentialData['alto'] = []
    sequentialData['tenor'] = []
    sequentialData['bass'] = []

    allData = midiData['train'] + midiData['test'] + midiData['valid']
    
    # DETERMINE MAX SONG LENGTH
    # for song in allData:
    #         songLength = len(song)
    #         if songLength > maxSongLength:
    #             maxSongLength = songLength
    #         print(f'Song length: {maxSongLength}')

    dictOfVoices = Voices()
    for song in allData:
        for voice in dictOfVoices.values():
            notes = []
            for beat in range(len(song)):
                if (len(song[beat]) > voice.dataIndex):
                    notes.append(song[beat][voice.dataIndex])
            while (len(notes) < Constants.MAX_SONG_LENGTH) and (len(notes) == len(song)):
                notes.append(0)
            encoded = oneHotEnocde(notes)
            split = splitSequence(encoded)
            sequentialData[voice.name] = sequentialData[voice.name] + split

    print(f"Soprano length: {len(sequentialData['soprano'])}")
    print(f"Alto length: {len(sequentialData['alto'])}")
    print(f"Tenor length: {len(sequentialData['tenor'])}")
    print(f"Bass length: {len(sequentialData['bass'])}")
    return midiData, ModelDataset(sequentialData)

def oneHotEnocde(part):
    print("one hot encode entered")
    encodedPart = []
    ENCODE_KEY = Constants.ENOCDE_KEY
    for item in part:
        encodedPart.append(ENCODE_KEY[item])
    return encodedPart

def decodePart(part: list):
    print("decode part entered")
    decodedPart = []
    for item in part:
        decodedPart.append(keyFromValue(Constants.ENCODE_KEY, item))
    return decodedPart

def keyFromValue(dict: dict, neededValue):
    print("key from value entered")
    for key, value in dict:
        if value == neededValue:
            return key
    raise ValueError(f"{neededValue} not found in dictionary {dict}")

def splitSequence(sequence: list):
    print("split sequence entered")
    if len(sequence) == 0:
        print("Empty sequence provided, returning empty list.")
        return []
    sequenceLength = Constants.SEQUENCE_LENGTH
    split = [sequence[i:i + sequenceLength] for i in range(0, len(sequence), sequenceLength)]
    return split