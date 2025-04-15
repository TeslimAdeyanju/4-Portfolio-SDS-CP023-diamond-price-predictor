from enum import Enum

class Cut(str, Enum):
    Fair = "Fair"
    Good = "Good"
    Very_Good = "Very Good"
    Premium = "Premium"
    Ideal = "Ideal"


class Color(str, Enum):
    J = "J"
    I = "I"
    H = "H"
    G = "G"
    F = "F"
    E = "E"
    D = "D"

class Clarity(str, Enum):
    I1 = "I1"
    SI2 = "SI2"
    SI1 = "SI1"
    VS2 = "VS2"
    VS1 = "VS1"
    VVS2 = "VVS2"
    VVS1 = "VVS1"
    IF = "IF"