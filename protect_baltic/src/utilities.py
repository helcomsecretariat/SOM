"""
Utility methods for easy use
"""


import time
from collections import namedtuple
import traceback


class Timer:
    """
    Simple timer for determining execution time
    """
    def __init__(self) -> None:
        self.start = time.perf_counter()
    def time_passed(self) -> int:
        """Return time passed since start, in seconds"""
        return time.perf_counter() - self.start
    def get_time(self) -> tuple:
        """Returns durations in hours, minutes, seconds as a named tuple"""
        duration = self.time_passed()
        hours = duration // 3600
        minutes = (duration % 3600) // 60
        seconds = duration % 60
        PassedTime = namedtuple('PassedTime', 'hours minutes seconds')
        return PassedTime(hours, minutes, seconds)
    def get_duration(self) -> str:
        """Returns a string of duration in hours, minutes, seconds"""
        t = self.get_time()
        return '%d h %d min %d sec' % (t.hours, t.minutes, t.seconds)
    def get_hhmmss(self) -> str:
        """Returns a string of duration in hh:mm:ss format"""
        t = self.get_time()
        return '[' + ':'.join(f'{int(value):02d}' for value in [t.hours, t.minutes, t.seconds]) + ']'
    def reset(self) -> None:
        """Reset timer"""
        self.start = time.perf_counter()


def exception_traceback(e: Exception):
    """
    Format exception traceback and print
    """
    tb = traceback.format_exception(type(e), e, e.__traceback__)
    print(''.join(tb))

