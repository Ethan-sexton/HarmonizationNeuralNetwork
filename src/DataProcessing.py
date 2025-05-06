import pickle
from classes.Voices import Voices
import Constants
from classes.Dataset import ModelDataset

def loadData():
    print("Loading Data...")
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
            encoded = voice.encodePart(notes)
            split = splitSequence(encoded)
            sequentialData[voice.name] = sequentialData[voice.name] + split
    
    return midiData, ModelDataset(sequentialData)

def keyFromValue(dict: dict, neededValue):
    for key, value in dict:

        if value == neededValue:
            return key
        
    raise ValueError(f"Error in DataProcessing\keyFromValue: {neededValue} not found in dictionary {dict}")

def splitSequence(sequence: list):
    splitLen = len(sequence)
    split = []
    for start in range(0, splitLen, Constants.SEQUENCE_LENGTH):
        splitEnd = start + Constants.SEQUENCE_LENGTH

        if splitEnd > splitLen:
            splitEnd = splitLen - 1
            start = splitEnd - Constants.SEQUENCE_LENGTH
        
        sequencePart = sequence[start:splitEnd]
        split.append(sequencePart)
    return split