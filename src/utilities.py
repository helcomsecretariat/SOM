"""
Small utility methods for convenience.
"""

import time
from collections import namedtuple
import traceback
import sys
import numpy as np
import matplotlib.pyplot as plt


class Timer:
    """
    Simple timer class for determining execution time.
    """
    def __init__(self) -> None:
        self.start = time.perf_counter()
    def time_passed(self) -> int:
        """Return time passed since start, in seconds."""
        return time.perf_counter() - self.start
    def get_time(self) -> tuple:
        """Returns durations in hours, minutes, seconds as a named tuple."""
        duration = self.time_passed()
        hours = duration // 3600
        minutes = (duration % 3600) // 60
        seconds = duration % 60
        PassedTime = namedtuple('PassedTime', 'hours minutes seconds')
        return PassedTime(hours, minutes, seconds)
    def get_duration(self) -> str:
        """Returns a string of duration in hours, minutes, seconds."""
        t = self.get_time()
        return '%d h %d min %d sec' % (t.hours, t.minutes, t.seconds)
    def get_hhmmss(self) -> str:
        """Returns a string of duration in hh:mm:ss format."""
        t = self.get_time()
        return '[' + ':'.join(f'{int(value):02d}' for value in [t.hours, t.minutes, t.seconds]) + ']'
    def reset(self) -> None:
        """Reset timer to zero."""
        self.start = time.perf_counter()


def exception_traceback(e: Exception, file = None):
    """
    Format exception traceback and print it.

    Arguments:
        e (Exception): exception from which to print traceback.
        file (str): optional file path to write to, otherwise defaults to sys.stdout.
    """
    tb = traceback.format_exception(type(e), e, e.__traceback__)
    print(''.join(tb), file=file)


def fail_with_message(m: str = None, e: Exception = None, file = None, do_not_exit: bool = False):
    """
    Prints the given exception traceback along with given message, and exits.

    Arguments:
        m (str): optional message to print along with traceback.
        e (Exception): exception from which to print traceback.
        file (str): optional file path to write to, otherwise defaults to sys.stdout.
        do_not_exit (bool): optional flag to not exit.
    """
    if e is not None:
        exception_traceback(e, file)
    if m is not None:
        print(m, file=file)
    print('Terminating.', file=file)
    if not do_not_exit:
        exit()


def display_progress(completion: float, size: int = 20, text: str = 'Progress: '):
    """
    Shows the current simulation progress as a percentage with a progress bar.

    Arguments:
        completion (float): fraction representing completion.
        size (int): total amount of simulations to run.
        text (str): optional text to display before progress bas.
    """
    x = int(size*completion)
    sys.stdout.write("%s[%s%s] %02d %%\r" % (text, "#"*x, "."*(size-x), completion*100))
    sys.stdout.flush()


class Progress:
    def __init__(self, total: int, bar_size: int = 50, text: str = 'Progress: '):
        self.total = total
        self.bar_size = bar_size
        self.text = text
        self.current = 0
    def increase(self):
        self.current += 1
    def display(self):
        completion = self.current / self.total
        x = int(self.bar_size * completion)
        sys.stdout.write("%s[%s%s] %02d %%\r" % (self.text, "#"*x, "."*(self.bar_size-x), completion*100))
    def reset(self):
        self.current = 0


def pert_dist(peak: float, low: float, high: float, size: int) -> np.ndarray:
    '''
    Returns a set of random picks from a PERT distribution.

    Arguments:
        peak (float): distribution peak.
        low (float): distribution lower tail.
        high (float): distribution higher tail.
        size (int): number of picks to return.

    Returns:
        numpy array
    '''
    # weight, controls probability of edge values (higher -> more emphasis on most likely, lower -> extreme values more probable)
    # 4 is standard used in unmodified PERT distributions
    gamma = 4
    # calculate expected value
    # mu = ((low + gamma) * (peak + high)) / (gamma + 2)
    if low == high and low == peak:
        return np.full(int(size), peak)
    r = high - low
    alpha = 1 + gamma * (peak - low) / r
    beta = 1 + gamma * (high - peak) / r
    return low + np.random.default_rng().beta(alpha, beta, size=int(size)) * r


def get_prob_dist(expecteds: np.ndarray, 
                  lower_boundaries: np.ndarray, 
                  upper_boundaries: np.ndarray, 
                  weights: np.ndarray) -> np.ndarray:
    '''
    Returns an aggregated probability distribution from all the individual expert answers provided. 
    Each value in the argument arrays correspond to a PERT distribution characteristic (peak, high, low). 
    Each individual distribution has a weight which impacts its contribution to the final aggregated distribution. 
    All arguments should be 1D arrays with percentage as unit.

    Arguments:
        expecteds (ndarray): individual distribution peaks.
        lower_boundaries (ndarray): individual distributions lows.
        upper_boundaries (ndarray): individual distribution highs.
        weights (ndarray): individual distribution weights.
    
    Returns:
        numpy array
    '''
    # verify that all arrays have the same size
    assert expecteds.size == lower_boundaries.size == upper_boundaries.size == weights.size

    #
    # TODO: remove uncomment in future to not accept faulty data
    # for now, sort arrays to have values in correct order
    #
    # # verify that all lower boundaries are lower than the upper boundaries
    # assert np.sum(lower_boundaries > upper_boundaries) == 0
    # # verify that most likely values are between lower and upper boundaries
    # assert np.sum((expecteds < lower_boundaries) & (expecteds > upper_boundaries)) == 0
    arr = np.full((len(expecteds), 3), np.nan)
    arr[:, 0] = lower_boundaries
    arr[:, 1] = expecteds
    arr[:, 2] = upper_boundaries
    arr = np.array([np.sort(row) for row in arr])
    lower_boundaries = arr[:, 0]
    expecteds = arr[:, 1]
    upper_boundaries = arr[:, 2]
    
    # select values that are not nan, bool matrix
    non_nan = ~np.isnan(expecteds) & ~np.isnan(lower_boundaries) & ~np.isnan(upper_boundaries)
    # multiply those values with weights, True = 1 and False = 0
    weights_non_nan = (non_nan * weights)

    # create a PERT distribution for each expert
    # from each distribution, draw a large number of picks
    # pool the picks together
    number_of_picks = 5000
    picks = []
    for i in range(len(expecteds)):
        peak = expecteds[i]
        low = lower_boundaries[i]
        high = upper_boundaries[i]
        w = weights_non_nan[i]
        if ~non_nan[i]: # note the tilde ~ to check for nan value
            continue    # skip if any value is nan
        dist = pert_dist(peak, low, high, w * number_of_picks)
        picks += dist.tolist()
    
    # return nan if no distributions (= no expert answers)
    if len(picks) == 0:
        return np.nan
        
    # create final probability distribution
    picks = np.array(picks) / 100.0   # convert percentages to fractions
    prob_dist = get_dist_from_picks(picks)

    return prob_dist


def get_pick(dist: np.ndarray) -> float:
    """
    Makes a random pick within [0, 1] weighted by the given discrete distribution.

    Arguments:
        dist (ndarray): probability distribution.

    Returns:
        pick (float): sample from the distribution.
    """
    if dist is not None:
        step = 1 / (dist.size - 1)
        a = np.arange(0, 1 + step, step)
        pick = np.random.choice(a, p=dist)
        return pick
    else:
        return np.nan


def get_dist_from_picks(picks: np.ndarray) -> np.ndarray:
    """
    Takes an array of picks and returns the probability distribution for each percentage unit. Picks need to be fractions in [0, 1].

    Arguments:
        picks (ndarray): array of random samples.

    Returns:
        dist (ndarray): probability distribution.
    """
    picks = np.round(picks, decimals=2)
    unique, count = np.unique(picks, return_counts=True)
    dist = np.zeros(shape=101)  # probability distribution, each element represents a percentage from 0 - 100 %
    # for each percentage, set its value to its frequency in the picks
    for i in range(dist.size):
        for k in range(unique.size):
            if i / 100.0 == unique[k]:
                dist[i] = count[k]
    dist = dist / dist.sum()    # normalize frequencies to sum up to 1
    return dist


def plot_dist(dist: np.ndarray):
    """
    Plot the given distribution

    Arguments:
        dist (ndarray): probability distribution.
    """
    # plot distribution
    y_vals = dist
    step = 1 / y_vals.size
    x_vals = np.arange(0, 1, step)
    plt.plot(x_vals, y_vals)
    # verify that get_pick works
    picks = np.array([get_pick(dist) for i in range(5000)])
    y_vals = get_dist_from_picks(picks)
    step = 1 / y_vals.size
    x_vals = np.arange(0, 1, step)
    plt.plot(x_vals, y_vals)
    plt.show()


def sanitize_string(s: str):
    """
    Makes a string valid for file and directory names.

    Arguments:
        s (str): string to sanitize.
    """
    place_holder = '_'
    for invalid in ['*', '"', '/', '\\', '<', '>', ':', '|', '?']:
        s = s.replace(invalid, place_holder)
    s = s.strip()   # remove leading and trailing whitespace
    return s
