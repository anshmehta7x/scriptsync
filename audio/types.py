from dataclasses import dataclass


@dataclass
class SpeechSegment:
    start: float
    end: float
    text: str
