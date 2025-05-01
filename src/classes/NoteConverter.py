import mido
import classes.Voices as Voices
import Constants as Constants
#Takes in Model's output, converts to standard midi
class NoteConverter:
    def __init__(self):
        self.ENCODE_KEY = Constants.ENCODE_KEY 

    def midiToNote(self, fileName):
        mid = mido.MidiFile(fileName)
        ticksPerBeat = mid.ticks_per_beat
        ticksPerSixteenth = ticksPerBeat // 4

        currentTick = 0
        events = []

        for msg in mid.tracks[0]:
            currentTick += msg.time
            if msg.type in ['note_on', 'note_off']:
                events.append((currentTick, msg))
        
        # Normalize timing to start at 0
        if events:
            firstTick = events[0][0]
            events = [(tick - firstTick, msg) for tick, msg in events]

        totalTicks = events[-1][0] if events else 0
        totalSixteenths = (totalTicks // ticksPerSixteenth) + 1

        notes = []
        currentNote = 0
        nextEventIDX = 0

        for i in range(totalSixteenths):
            currentTime = i * ticksPerSixteenth

            while nextEventIDX < len(events) and events[nextEventIDX][0] <= currentTime:
                eventTick, msg = events[nextEventIDX]
                if msg.type == 'note_on' and msg.velocity > 0:
                    currentNote = msg.note
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    currentNote = 0
                nextEventIDX += 1
            notes.append(currentNote)
        return notes
    
    def encodeNotes(self, notes: list):
        encodedPart = []
        for item in notes:
            encodedPart.append(self.ENCODE_KEY[item])
        return encodedPart
    
    def flattenVoice(self, voice):
        if all(isinstance(note, int) for note in voice):
        # If the input is already a flat list, return it as is
            return voice
    # Otherwise, flatten the list of lists
        return [note for sublist in voice for note in sublist]
    def notesToMidi(self, notesData: list, filename='output.midi', tempoIn=120):
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)

        # Set the tempo
        tempo = mido.bpm2tempo(tempoIn)
        track.append(mido.MetaMessage('set_tempo', tempo=tempo))

        ticksPerBeat = mid.ticks_per_beat
        ticksPerSixteenth = ticksPerBeat // 4
        # Use notesData directly as allVoices
        allVoices = notesData

        # Initialize previous notes for each voice
        numVoices = len(allVoices)
        previousNotes = [0] * numVoices

        # Iterate over each time step
        for timeStep in range(len(allVoices[0])):
            timeIncrement = ticksPerSixteenth if timeStep > 0 else 0  # No delay for the first time step
            for voiceIndex in range(numVoices):
                note = allVoices[voiceIndex][timeStep]

                # Ensure note is an integer
                if not isinstance(note, int):
                    raise TypeError(f"Invalid note value: {note}. Note values must be integers.")

                # Handle note-off events for previous notes
                if previousNotes[voiceIndex] != 0 and previousNotes[voiceIndex] != note:
                    track.append(mido.Message('note_off', note=previousNotes[voiceIndex], velocity=64, time=timeIncrement))
                    timeIncrement = 0  # Reset time increment after the first event

                # Handle note-on events for current notes
                if note != 0 and note != previousNotes[voiceIndex]:
                    track.append(mido.Message('note_on', note=note, velocity=64, time=timeIncrement))
                    timeIncrement = 0  # Reset time increment after the first event

                # Update the previous note for this voice
                previousNotes[voiceIndex] = note

        # Add note-off events for any remaining notes
        for note in previousNotes:
            if note != 0:
                track.append(mido.Message('note_off', note=note, velocity=64, time=ticksPerSixteenth))

        # Save the MIDI file
        mid.save(filename)
        print(f"MIDI saved to: {filename}")

    def keyFromValue(self, dict: dict, neededValue):
        print("key from value entered")
        print(neededValue)
        for key, value in dict.items():
            print(value)
            if value == neededValue:
                return key
        print(f"Warning: {neededValue} not found in dictionary. Using default value.")
        return 0
    
    def decodeNotes(self, voices: Voices, notes: list):
        decodedVoices = []
        for voiceName, voiceNotes in zip(voices.keys(), notes):
            voiceDict = voices[voiceName].ENCODE_KEY  # Get the dictionary for the current voice
            decodedPart = []
            for item in voiceNotes:
                decodedValue = self.keyFromValue(voiceDict, item)
                if not isinstance(decodedValue, int):
                    print(f"Warning: Decoded value {decodedValue} is not an integer. Using default value 0.")
                    decodedValue = 0  # Default to 0 if the value is invalid
                decodedPart.append(decodedValue)
            decodedVoices.append(decodedPart)
        return decodedVoices

    def splitSequence(self, sequence: list):
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
    
    def padSequence(self, sequence, targetLength, padValue=0):
        while len(sequence) < targetLength:
            sequence.append(padValue)
        return sequence

