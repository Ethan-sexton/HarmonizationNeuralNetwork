from classes.Voice import Voice
from classes.Range import Range

class Voices(dict):
    def __init__(self):
        self['soprano'] = Voice('soprano', 0, Range(60,81))
        self['alto'] = Voice('alto', 1, Range(57,79))
        self['tenor'] = Voice('tenor', 2, Range(49,60))
        self['bass'] = Voice('bass', 3, Range(41,57))