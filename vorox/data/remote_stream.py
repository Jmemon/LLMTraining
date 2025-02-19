
from smart_open import open

def open_remote(url: str):
    return open(url, "r")
