class Range:
    def __init__(self, lowestNote, highestNote):
        self.lowestNote = lowestNote
        self.highestNote = highestNote
    
    def getRange(self):
        return [self.lowestNote, self.highestNote]