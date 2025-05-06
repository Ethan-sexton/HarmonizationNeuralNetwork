import torch
from classes.Voices import Voices
from classes.NoteConverter import NoteConverter
import Constants
def ApplyModel(modelPath, midiInput):

    if modelPath[0] == 'F':
        from model.FHarmonizationNeuralNetwork import FHarmonizationNeuralNetwork
        model = FHarmonizationNeuralNetwork()
    elif modelPath[0] == 'R':
        from model.RHarmonizationNeuralNetwork import RHarmonizationNeuralNetwork
        model = RHarmonizationNeuralNetwork()
    else:
        raise ValueError("Invalid model path. Model path should start with 'F' or 'R'.")

    model.load_state_dict(torch.load(f'src\output\models\{modelPath}', weights_only=False, map_location=torch.device('cpu')))
    model.eval()

    modelName = modelPath.split('\\')[-1].split('.')[0]
    converter = NoteConverter()
    notes = converter.midiToNote(f'src\input\{midiInput}')
    voices = Voices()
    sopVoice = voices['soprano']

    #Prep Notes
    paddedNotes = converter.padSequence(notes, Constants.SEQUENCE_LENGTH)
    encoded = sopVoice.encodePart(paddedNotes)
    prepped = converter.splitSequence(encoded)

    notesTensor = torch.tensor(prepped, dtype=torch.float32)  # Add batch dimension

    with torch.no_grad():
        predAlto, predTenor, predBass = model(notesTensor)

    predAltoList = predAlto.squeeze(0).argmax(dim=1).tolist()  # Get the predicted class for each step
    predTenorList = predTenor.squeeze(0).argmax(dim=1).tolist()
    predBassList = predBass.squeeze(0).argmax(dim=1).tolist()

    predAltoList = converter.flattenVoice(predAltoList)
    for item in range(len(predAltoList)):
        if predAltoList[item] != 0:
            predAltoList[item] = predAltoList[item] + 53

    predTenorList = converter.flattenVoice(predTenorList)
    for item in range(len(predTenorList)):
        if predTenorList[item] != 0:
            predTenorList[item] = predTenorList[item] + 47
            
    predBassList = converter.flattenVoice(predBassList)
    for item in  range(len(predBassList)):
        if predBassList[item] != 0:
            predBassList[item] = predBassList[item] + 36


    # Combine the original notes with the predictions
    output = [notes, predAltoList, predTenorList, predBassList]

    # Save the output as a MIDI file
    converter.notesToMidi(output, f'src\output\midi\{modelName}-{midiInput}-output.mid')
