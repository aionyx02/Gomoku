from enum import IntEnum
from typing import Any


class Stone(IntEnum):
    EMPTY: int
    BLACK: int
    WHITE: int


class GameStatus(IntEnum):
    PLAYING: int
    BLACK_WIN: int
    WHITE_WIN: int
    DRAW: int


class Board:
    def __init__(self, size: int = 15) -> None: ...
    def place_stone(self, x: int, y: int) -> bool: ...
    def get_current_player(self) -> Stone: ...
    def get_status(self) -> GameStatus: ...
    def get_stone(self, x: int, y: int) -> Stone: ...
    def get_observation(self) -> Any: ...
