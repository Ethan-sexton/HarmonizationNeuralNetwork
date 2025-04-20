from classes.Voice import Voice
from classes.Range import Range
import DataProcessing as dataProc
from model.HarmonizationNeuralNetwork import HarmonizationNeuralNetwork as nn

x_sop, y_alto, y_tenor, y_bass = dataProc.loadData()
print(f'Soprano: {x_sop}')
print(f'Alto: {y_alto}')
print(f'Tenor: {y_tenor}')
print(f'Bass: {y_bass}')
#print(dataProc.decodeData(df))
# sopVoice = Voice('soprano', 0, Range(60,81))
# sopVoice = sopVoice.encodePart(df, 0, 0)
# altoVoice = Voice('alto', 1, Range(57,79))
# altoVoice = altoVoice.encodePart(df, 0, 1)
# tenorVoice = Voice('tenor', 2, Range(49,60))
# tenorVoice = tenorVoice.encodePart(df, 0, 2)
# bassVoice = Voice('bass', 3, Range(41,57))
# bassVoice = bassVoice.encodePart(df, 0, 3)
# print(sopVoice)
# print(altoVoice)
# print(tenorVoice)
# print(bassVoice)

# testModel = nn()
# for epoch in range(5):
#     nn.train()
