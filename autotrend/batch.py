

from time import time
from typing import List, Tuple
import numpy as np
import pandas as pd
from numpy import linalg
from scipy.optimize import dual_annealing
from scipy.optimize.optimize import OptimizeResult

from .util import get_lines_boundries


def get_optim_splits(x: List[int],
                     y: List[float],
                     min_split_len: int = 1,
                     num_lines: int = 12,
                     verbose: bool = False) -> OptimizeResult:
    """
    Gets optimum piecewise linear function fitted to the data for the specified numbers of lines, by global
    optimization and least square per iteration.

    :param x: indices of values
    :param y: values, e.g., stock prices at the given indices
    :param min_split_len: minimum length considered for each split
    :param num_lines: numbers of lines in piecewise linear model
    :param verbose:

    :return: optimum object returned by the global optimization method.
    """
    def cost_func(splits: List[int]) -> float:

        nonlocal A, params, ssr

        splits = x[:1] + (splits.tolist() if isinstance(splits, np.ndarray) else splits)
        splits.sort()

        params, A, ssr_ = fit_pwl_model(x, y, splits, issorted=True)

        # if unstable condition happens increase the cost to force the algorithm to the opposite
        ssr = ssr + 1000 if not isinstance(ssr_, float) or pd.isnull(ssr_) else ssr_

        return ssr

    n = len(x)
    assert n == len(y), "x and y should have equal lengths!"

    ssr = 0
    A = []
    params = []

    #
    if num_lines <= 1:
        params, A, ssr = fit_pwl_model(x, y, x[:1])
        return OptimizeResult({"fun": ssr,
                               "message": "least square results are returned for 1 line fit, no split.",
                               "succes": True,
                               "x": np.array([])})

    #
    ub = [max(x)] * (num_lines - 1)
    lb = [min_split_len] * (num_lines - 1)

    t0 = time()

    optimum = dual_annealing(func=cost_func,
                             bounds=list(zip(lb, ub)),
                             args=(),
                             maxiter=500,
                             initial_temp=5230.0,
                             restart_temp_ratio=2e-05,
                             visit=2.62,
                             accept=-5.0,
                             maxfun=10000000,
                             seed=None,
                             no_local_search=True,
                             callback=None,
                             x0=None)

    if verbose:
        print(f"Optimization time: {time() - t0} for {num_lines} number of lines and {len(x)} samples")

    return optimum


def get_optim_splits_with_regularization(x: List[int],
                                         y: List[float],
                                         min_split_len: int = 1,
                                         num_lines: int = 12,
                                         use_length_imbalance_cost: bool = False,
                                         use_width_imbalance_cost: bool = False,
                                         use_peaks_imbalance_cost: bool = False,
                                         use_high_frequency_reward: bool = False,
                                         reg_smoothing: float = 1e6,
                                         decay: float = 0.1,
                                         alpha: float = 1.0,
                                         beta: float = 1.0) -> object:
    """
    Gets optimum piecewise linear function fitted to the data for the specified numbers of lines, by global
    optimization and least square per iteration.

    :param x: indices of values
    :param y: values, e.g., stock prices at the given indices
    :param min_split_len: minimum length considered for each split
    :param num_lines: numbers of lines in piecewise linear model
    :param use_length_imbalance_cost: if True, uses length imbalance regularization cost
    :param use_width_imbalance_cost: if True, uses width imbalance regularization cost
    :param use_peaks_imbalance_cost: if True, uses peaks imbalance regularization cost
    :param use_high_frequency_reward: if True, uses high frequency reward cost
    :param reg_smoothing: smoothing factor for regularization
    :param decay: decay factor for exponential sum of adaptive weighting regularization
    :param alpha: initial value for adaptive weighting of imbalance cost regularization terms
    :param beta:  initial value for adaptive weighting of high frequency reward regularization terms

    :return: optimum object returned by the global optimization method.
    """
    def cost_func(splits: List[int]) -> float:

        nonlocal alpha, beta, ssr, imbalance_loss, A, params, lenght_imbalance_cost, width_imbalance_cost, \
            peaks_imbalance_cost, high_frequency_reward

        splits = x[:1] + (splits.tolist() if isinstance(splits, np.ndarray) else splits)
        splits.sort()

        # for x1, x2 in zip(splits, splits + x[-1:]):
        #     if x2 - x1 < min_split_len:
        #         # force min split len
        #         return ssr + 1000

        params, A, ssr_ = fit_pwl_model(x, y, splits, issorted=True)
        y_hat: List[float] = np.matmul(A, params).tolist()

        # if unstable condition happens increase the cost to force the algorithm to the opposite
        ssr = ssr + 1000 if not isinstance(ssr_, float) or pd.isnull(ssr_) else ssr_

        #
        lenght_imbalance_cost = _get_lenght_imbalance_cost(x, splits) if use_length_imbalance_cost else 0.0
        width_imbalance_cost = _get_width_imbalance_cost(x, y, n, y_hat, splits) if use_width_imbalance_cost else 0.0
        peaks_imbalance_cost = _get_peaks_imbalance_cost(x, y, n, y_hat, splits) if use_peaks_imbalance_cost else 0.0
        high_frequency_reward = _get_high_frequency_reward(x, y, n, y_hat, splits) if use_high_frequency_reward else 0.0

        imbalance_loss = lenght_imbalance_cost + width_imbalance_cost + peaks_imbalance_cost

        loss = ssr + alpha * imbalance_loss - beta * high_frequency_reward

        if (isinstance(ssr, float) and isinstance(imbalance_loss, float) and
                (use_length_imbalance_cost or use_width_imbalance_cost or use_peaks_imbalance_cost)):
            alpha = alpha * decay + (1 - decay) * (ssr / (imbalance_loss + reg_smoothing))

        if isinstance(ssr, float) and isinstance(high_frequency_reward, float) and use_high_frequency_reward:
            beta = beta * decay + (1 - decay) * (ssr / (high_frequency_reward + reg_smoothing))

        return loss


    n = len(x)
    assert n == len(y), "x and y should have equal lengths!"

    ssr = 0
    A = []
    params = []
    imbalance_loss = float('inf')
    lenght_imbalance_cost = float('inf')
    width_imbalance_cost = float('inf')
    peaks_imbalance_cost = float('inf')
    high_frequency_reward = 0.0

    #
    if num_lines <= 1:
        params, A, ssr = fit_pwl_model(x, y, x[:1])
        return OptimizeResult({"fun": ssr,
                               "message": "least square results are returned for 1 line fit, no split.",
                               "succes": True,
                               "x": np.array([])})

    #
    ub = [max(x)] * (num_lines - 1)
    lb = [min_split_len] * (num_lines - 1)

    t0 = time()

    optimum = dual_annealing(func=cost_func,
                             bounds=list(zip(lb, ub)),
                             args=(),
                             maxiter=500,
                             initial_temp=5230.0,
                             restart_temp_ratio=2e-05,
                             visit=2.62,
                             accept=-5.0,
                             maxfun=10000000,
                             seed=None,
                             no_local_search=True,
                             callback=None,
                             x0=None)

    print(f"Finished optimization, time: {time() - t0} seconds")

    return optimum


def _get_lenght_imbalance_cost(x: List[int], splits: List[int]) -> float:
    """
    Regularization cost against length imbalance defined by the splits

    :param x: indices of values
    :param splits: array that includes split indexes (fist element should be 0)

    :return: imbalance cost
    """
    lenghts = []
    for x1, x2 in zip(splits, splits[1:] + x[-1:]):
        lenghts.append(x2 - x1)

    lenght_imbalance_cost = np.var(lenghts) if lenghts else 0.0

    return lenght_imbalance_cost


def _get_width_imbalance_cost(x: List[int],
                              y: List[float],
                              n: int,
                              y_hat: List[float],
                              splits: List[int]) -> float:
    """
    Regularization cost against width imbalance defined by the splits and lower/upper bounds per line

    :param x: indices of values
    :param y: e.g., stock prices at the given indices
    :param n: numbers of samples ( == len(x) == len(y) )
    :param y_hat: predicted values for y by the model
    :param splits: array that includes split indexes (fist element should be 0)

    :return: imbalance cost
    """
    widths = []
    for x1, x2 in zip(splits, splits[1:] + x[-1:]):
        diffs = [y[i] - y_hat[i] for i in range(int(x1), int(x2)) if i < n]
        if diffs:
            widths.append(max(diffs) - min(diffs))

    width_imbalance_cost = np.var(widths) if widths else 0.0

    return width_imbalance_cost


def _get_peaks_imbalance_cost(x: List[int],
                              y: List[float],
                              n: int,
                              y_hat: List[float],
                              splits: List[int]) -> float:
    """
    Regularization cost against peaks symmetry imbalance defined by the splits and lower/upper bounds per line

    :param x: indices of values
    :param y: e.g., stock prices at the given indices
    :param n: numbers of samples ( == len(x) == len(y) )
    :param y_hat: predicted values for y by the model
    :param splits: array that includes split indexes (fist element should be 0)

    :return: imbalance cost
    """
    peaks_imbalances = []
    for x1, x2 in zip(splits, splits[1:] + x[-1:]):
        diffs = [y[i] - y_hat[i] for i in range(int(x1), int(x2)) if i < n]
        if diffs:
            peaks_imbalances.append((max(diffs) + min(diffs)) ** 2)

    peaks_imbalance_cost = np.mean(peaks_imbalances) ** 0.5 if peaks_imbalances else 0.0

    return peaks_imbalance_cost


def _get_frequency(y: List[float],
                   n: int,
                   y_hat: List[float],
                   splits: List[int]) -> List[int]:
    """
    Regularization cost against width imbalance defined by the splits and lower/upper bounds per line

    :param y: e.g., stock prices at the given indices
    :param n: numbers of samples ( == len(x) == len(y) )
    :param y_hat: predicted values for y by the model
    :param splits: array that includes split indexes (fist element should be 0)

    :return: frequencies of passing fitted lines per split segments
    """
    frequencies = []
    for x1, x2 in zip(splits[:-1], splits[1:]):
        freq = sum([y[i] < y_hat[i] < y[i + 1] or y[i + 1] < y_hat[i] < y[i]
                    for i in range(int(x1), int(x2) - 1) if i < n - 1])
        frequencies.append(freq)

    return frequencies


def _get_high_frequency_reward(x: List[int],
                               y: List[float],
                               n: int,
                               y_hat: List[float],
                               splits: List[int]) -> float:
    """
    Regularization cost against peaks symmetry imbalance defined by the splits and lower/upper bounds per line

    :param x: indices of values
    :param y: values, e.g., stock prices at the given indices
    :param n: numbers of samples ( == len(x) == len(y) )
    :param y_hat: predicted values for y by the model
    :param splits: array that includes split indexes (fist element should be 0)

    :return: average value of frequency estimation per segments
    """
    splits = splits + x[-1:]
    freqs = _get_frequency(y, n, y_hat, splits)
    freqs_per_len = [freq / max(x2 - x1, 1) for freq, (x1, x2) in zip(freqs, zip(splits[:-1], splits[1:]))]
    frequency_reward = np.mean(freqs_per_len) if freqs_per_len else 0.0

    return frequency_reward


def find_best_num_lines(x: List[int],
                        y: List[float],
                        min_split_len: int,
                        min_num_lines: int,
                        max_num_lines: int,
                        mv_ave_window: int = 120,
                        verbose: bool = False) -> Tuple[List[dict], int]:
    """
    Finds best number of splits by running multiple global optimizations for values between the specified min and max
    limits for the numbers of lines.

    * This approach may be expensive, better to guess or use adaptive method
    * not recomended for long time spans in which stock volatility is high

    :param x: indices of values
    :param y: values, e.g., stock prices at the given indices
    :param min_split_len: minimum numbers of pints to consider per split
    :param min_num_lines: minimum numbers of splits to search
    :param max_num_lines: maximum numbers of splits to search
    :param mv_ave_window: length of moving average window for estimating volatility
    :param verbose:

    :return:
    """

    n = len(x)
    assert n == len(y), "x and y should have equal lengths!"

    # getting moving average for the specified windows and estimating expected volatility
    y_ = [d for d in y if pd.notnull(d) and d is not None]
    n_ = len(y_)
    y_mvave = [float(np.mean(y_[max(0, i - mv_ave_window // 2): i + mv_ave_window // 2 + 1])) for i in range(n_)]
    y_centered = [p - ave if pd.notnull(ave) else 0 for p, ave in zip(y_, y_mvave)]
    ave_range = np.mean([np.max(y_centered[i:i + mv_ave_window]) - np.min(y_centered[i:i + mv_ave_window]) for i in
                         range(n_ - mv_ave_window)])
    if verbose:
        print(f"estimated expected volatility for {mv_ave_window} window: {ave_range}")

    # finding best numbers of lines by doing optimization inside binary search
    optim_fn = get_optim_splits

    optim_splits_results = []
    l, r = min_num_lines, max_num_lines
    num_lines = l

    while l < r:
        
        num_lines = (l + r) // 2

        if verbose: 
            print(f"*** (l, r): ({l}, {r}), numlines: {num_lines}")
        
        optimum = optim_fn(x, y, min_split_len, num_lines)

        if verbose:
            print(f"*** finished optimization. ")

        splits = x[:1] + optimum.x.tolist()
        splits.sort()
        params, A, ssr = fit_pwl_model(x, y, splits, issorted=True)
        y_hat: List[float] = np.matmul(A, params).tolist()

        lines, lower_lines, upper_lines = get_lines_boundries(x, y_hat, y, splits)
        ranges = [ul[1][0] - ll[1][0] for ll, ul in zip(lower_lines, upper_lines)]

        max_range = np.max(ranges)

        optim_splits_results.append({"num_lines": num_lines,
                                     "ave_freqs": _get_high_frequency_reward(x, y, n, y_hat, splits),
                                     "ave_ranges": np.mean(ranges),
                                     "max_ranges": max_range,
                                     "std_ranges": np.std(ranges),
                                     "optimum": optimum})

        if max_range > ave_range:
            l = num_lines + 1
        else:
            r = num_lines

    return sorted(optim_splits_results, key=lambda d: d["num_lines"]), num_lines

    
def find_next_split_bs(y: List[float], max_range: float, symmetric: bool = False):
    """
    Finds next breaking point from the start of the time series, using regression and limit on max range of variation.

    :param y: signal (e.g, stock price)
    :param max_range: max acceptable range of variation
    :param symmetric: if True, intervals will be symmetric around the regression line.

    :return: index of the breaking point
    """
    n = len(y)
    l, r = 0, n
    splits = [0]
    while l <= r:
        m = (l + r) // 2
        x = [i for i in range(m)]
        y_ = y[:m]
        A = get_A(x, splits)
        params, ssr, _, _ = linalg.lstsq(A, y_)
        y_hat: List[float] = np.matmul(A, params).tolist()
        lines, lower_lines, upper_lines = get_lines_boundries(x, y_hat, y_, splits, symmetric=symmetric)
        cur_range = [ul[1][0] - ll[1][0] for ll, ul in zip(lower_lines, upper_lines)][0]

        if cur_range >= max_range:
            r = m - 1
        else:
            l = m + 1

    return m


def find_splits_bs(y, max_range, symmetric=False):
    """
    Finds splits by doing binary search for segments with variability less thatn equal to the max_range.

    :param y: signal (e.g, stock price)
    :param max_range: max acceptable range of variation
    :param symmetric: if True, intervals will be symmetric around the regression line.

    :return: indices of split locations.
    """
    splits = [0]
    ind = 0
    while y:
        i = find_next_split_bs(y, max_range, symmetric=symmetric)
        ind += i
        splits.append(ind)
        y = y[i:]

    return splits


def get_A(x: List[int], splits: List[int]) -> np.ndarray:    
    """
    Returns Predicates matrix for getting piecewise linear function by least square based on defined splits

    :param x: array of indexes for the time series data
    :param splits: array that includes split indexes (fist element should be 0)

    :return:
    """
    def get_vals(ind):
        row = [0] * len(splits)
        for i, split in enumerate(splits):
            if ind <= split:
                break
            row[i] = ind - split
        return [1] + row

    return np.array([get_vals(i) for i in x])


def fit_pwl_model(x: List[int],
                  y: List[float],
                  splits: List[int],
                  issorted: bool = False) -> Tuple[List[float], np.ndarray, float]:
    """
    Piecewise linear model fitted to the data by least square for the specified split locations.

    :param x: indexes of values (e.g., prices)
    :param y: e.g., stock prices at the given indices
    :param splits: array that includes split indexes (first element should be 0)
    :param issorted: whether the splits are sorted or not

    :return: optimum line parameters, predicates matrix, sum of squared of residuals
    """
    if not issorted:
        splits.sort()
    A = get_A(x, splits)
    params, ssr, _, _ = linalg.lstsq(A, y)

    return params, A, ssr


def get_A_disjoint(x: List[int], splits: List[int]) -> np.ndarray:
    """
    Returns Predicates matrix for getting piecewise linear function by least square based on defined splits

    :param x: array of indexes for the time series data
    :param splits: array that includes split indexes (fist element should be 0)

    :return:
    """
    def get_vals(ind):
        row = [0] * len(splits) * 2
        for i, split in enumerate(splits):
            row[2 * i] = 1
            if ind <= split:
                break
            row[2 * i + 1] = ind - split
        return row

    return np.array([get_vals(i) for i in x])


def fit_pwl_disjoint_model(x: List[int],
                           y: List[float],
                           splits: List[int],
                           issorted: bool = False) -> Tuple[List[float], np.ndarray, float]:
    """
    Piecewise linear model fitted to the data by least square for the specified split locations.

    :param x: indexes of values (e.g., prices)
    :param y: e.g., stock prices at the given indices
    :param splits: array that includes split indexes (fist element should be 0)
    :param issorted: whether the splits are sorted or not

    :return: optimum line parameters, predicates matrix, sum of squared of residuals
    """
    if not issorted:
        splits.sort()
    A = get_A_disjoint(x, splits)
    params, ssr, _, _ = linalg.lstsq(A, y)

    return params, A, ssr

