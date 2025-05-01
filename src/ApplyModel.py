import torch
from classes.Voices import Voices
from model.HarmonizationNeuralNetwork import HarmonizationNeuralNetwork
from classes.NoteConverter import NoteConverter
from classes import Constants
model = HarmonizationNeuralNetwork()
model.load_state_dict(torch.load('harmonizer.pth', weights_only=False, map_location=torch.device('cpu')))
model.eval()

converter = NoteConverter()
notes = converter.midiToNote('Composition Project MUS 173.mid')
voices = Voices()
sopVoice = voices['soprano']

#Prep Notes
paddedNotes = converter.padSequence(notes, Constants.SEQUENCE_LENGTH)
encoded = sopVoice.encodePart(paddedNotes)
prepped = converter.splitSequence(encoded)
SEQUENCE_LENGTH = Constants.SEQUENCE_LENGTH

notesTensor = torch.tensor(prepped, dtype=torch.float32)  # Add batch dimension

print(f"Input tensor shape: {notesTensor.shape}")
with torch.no_grad():
    predAlto, predTenor, predBass = model(notesTensor)
print(predAlto)
print(predTenor)
print(predBass)
predAltoList = predAlto.squeeze(0).argmax(dim=1).tolist()  # Get the predicted class for each step
predTenorList = predTenor.squeeze(0).argmax(dim=1).tolist()
predBassList = predBass.squeeze(0).argmax(dim=1).tolist()
print(f'Soprano: {notes}')
print(f"Predicted Alto: {predAltoList}")
print(f"Predicted Tenor: {predTenorList}")
print(f"Predicted Bass: {predBassList}")

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

print(f"Scaled Predicted Alto: {predAltoList}")
print(f"Scaled Predicted Tenor: {predTenorList}")
print(f"Scaled Predicted Bass: {predBassList}")

print(f"Length of Soprano: {len(notes)}")
print(f"Length of Predicted Alto: {len(predAltoList)}")
print(f"Length of Predicted Tenor: {len(predTenorList)}")
print(f"Length of Predicted Bass: {len(predBassList)}")
# Combine the original notes with the predictions
output = [notes, predAltoList, predTenorList, predBassList]
print(f"Output: {output}")
# Save the output as a MIDI file
converter.notesToMidi(output, 'myproj_output.mid')
