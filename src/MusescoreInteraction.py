import pandas as pd
import json

notes = {0: "A", 1: "A#", 2: "B", 3: "C", 4: "C#", 5: "D", 6: "D#", 7: "E", 8:"F", 9:"F#", 10:"G", 11:"G#"}

jsonLocation = 'HarmonizationNeuralNetwork\dataset\Jsb16thSeparated.json'
with open(jsonLocation) as jsonData:
    jsonFile = json.load(jsonData)
    df = pd.DataFrame(jsonFile["test"])
#Examples of fetching different portions from data frame:
# print("Entire Dataframe: ")
# print(df)

# print(f"\nSpecific Song:")
# print(df[0])

# print(f"\nSpecific Measure:") #Assuming Common Time
# measure = df[0].to_list()
# for i in range(16):
#     print(str(measure[i]) + f" {i}")

# print(f"\nSpecific Beat:")
# print(df[0].to_list()[0])

# print(f"\nSpecific Voice:")
# print(df[0][0][0])
# def keyToNote(key):
#     return str(notes[(key - 21) % 12]) + str(int((key - 21) / 12) + 1)
# print(f"\nSpecific Voice for entire song:")
# piece = df[0]
# for j in range(len(piece)):
#     print(piece[j][0])
SopranoVoice = []
#Get Soprano Voice
song = 0
for k in range(len(df.iloc[song])):
    if df.iloc[0][k]:
        SopranoVoice.append(str(df.iloc[song][k][0]))
    print(SopranoVoice)
# Get other voices

#TODO:
# Seperate voices into encoded matrices for each voice - getDummies()?
# Program the NN Model
# Converting DF format to midi and vice-versa

# Getting midi from inside Musescore
# Putting generated midi back into Musescore
