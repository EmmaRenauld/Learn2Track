"""Monitoring utilities"""
import logging
import sys
from time import time

COLOR_CODES = {
    'black': '\u001b[30m',
    'red': '\u001b[31m',
    'green': '\u001b[32m',
    'yellow': '\u001b[33m',
    'blue': '\u001b[34m',
    'magenta': '\u001b[35m',
    'cyan': '\u001b[36m',
    'white': '\u001b[37m',
    'reset': '\u001b[0m'
}


class Timer:
    """Times code within a `with` statement, optionally adding color. """

    def __init__(self, txt, color=None):
        try:
            prepend = (COLOR_CODES[color] if color else '')
            append = (COLOR_CODES['reset'] if color else '')
        except KeyError:
            prepend = ''
            append = ''

        self.txt = prepend + txt + append

    def __enter__(self):
        self.start = time()

        txt_line = self.txt + "... "
        logging.info(txt_line)

    def __exit__(self, type, value, tb):
        txt_line = self.txt + " done in {:.2f} sec.".format(time() - self.start)
        logging.info(txt_line)
