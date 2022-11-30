# -*- coding: utf-8 -*-
class Switch(object):
    def __init__(self, value: str) -> None:
        self.__value = value
        self.__fall = False

    def __iter__(self) -> bool:
        """Return the match method once, then stop"""
        yield self.match
        return StopIteration

    def match(self, *args: tuple) -> bool:
        """Indicate whether to enter a case suite"""
        if self.__fall or not args:
            return True
        elif self.__value in args:  # changed for v1.5, see below
            self.__fall = True
            return True
        else:
            return False
