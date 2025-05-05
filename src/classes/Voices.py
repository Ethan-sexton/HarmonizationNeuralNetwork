from classes.Voice import Voice
from classes.Range import Range

class Voices(dict):
    def __init__(self):
        self['soprano'] = Voice('soprano', 0, Range(43,88))
        self['alto'] = Voice('alto', 1, Range(50,76))
        self['tenor'] = Voice('tenor', 2, Range(42,71))
        self['bass'] = Voice('bass', 3, Range(36,68))