

from typing import List, Tuple

from .util import fit_linear_model, get_biases, get_range, is_in_range, estimate_volatility


def next_split_adaptive(y: List[float],
                        start: int,
                        end: int,
                        n: int,
                        incr: int,
                        min_break_len: int,
                        symmetric: bool,
                        max_range: float,
                        min_points=10) -> Tuple[int, Tuple[float, float], float]:
    """
    Finds next split in the line after the start index (starting from the end >= start).
    
    Algorithm:
      - increase index until distance between boundary lines is not more than the max_range.
      - find the next boundary crossing that violates the lines continuously for min_break_len numbers of data points.
      - return the index of line crossing as the next split point.

    :param y: values (e.g., stock prices).
    :param start: start index of the data to fit the next trendline.
    :param end: end index to start fitting the next trendline (end >= start).
    :param n: index of the last value to use for fitting the next trendline (n >= end).
    :param incr: increments of data points to use in each iteration.
    :param min_break_len: minimum numbers of pints to pass the boundary lines to be counted as valid violation.
    :param symmetric: if True, same distance from the regression line will be used for lower/upper line biases.
    :param max_range: max distance between lower/upper boundary lines, used as a measure of volatility.
    :param min_points: min numbers of points to be used in each trendline.

    :return: index of split, (slope, intercept), current volatility estimation, bias of lower line, bias of upper line
    """
    # if the remaining data is less than min required points, return the fitted line
    if n - start < min_points:
        end = n
        (b, a), _, _ = fit_linear_model(y[start:end])
        bias_lower, bias_upper = get_biases(y, start, end, a, b, symmetric)
        cur_range = get_range(bias_upper, bias_lower, a, vertical=True)
        return end, (a, b), cur_range, bias_lower, bias_upper

    # increase index to have at least min_points
    end = max(end, start + min_points)
    
    # increase index until distance between boundary lines is not more than the max_range
    cur_range = 0    
    while end <= n and cur_range <= max_range:
        (b, a), _, _ = fit_linear_model(y[start:end])
        bias_lower, bias_upper = get_biases(y, start, end, a, b, symmetric)
        cur_range = get_range(bias_upper, bias_lower, a, vertical=True)
        end += incr
    
    while cur_range > max_range and end > start + min_points:
        end -= 1
        (b, a), _, _ = fit_linear_model(y[start:end])
        bias_lower, bias_upper = get_biases(y, start, end, a, b, symmetric)
        cur_range = get_range(bias_upper, bias_lower, a, vertical=True)
    
    if end >= n:
        return n, (a, b), cur_range, bias_lower, bias_upper

    # find the first valid violation of boundary lines
    cnt_out = 0
    k = end
    for k in range(end+1, n):
        if is_in_range(b - bias_lower, b + bias_upper, a, k-start, y[k]):
            cnt_out = 0
            continue
        cnt_out += 1
        if cnt_out == min_break_len:
            end = k - min_break_len
            break

    if k == n - 1:
        end = n

    return end, (a, b), cur_range, bias_lower, bias_upper


def trendlines_adaptive(y: List[float],
                        last_ind: int,
                        incr: int,
                        min_break_len: int,
                        symmetric: bool,
                        mv_ave_window: float,
                        min_points=10,
                        alpha: float = 0.5) -> Tuple[List[int], List[Tuple[float, float]]]:
    """
    Fits trendlines to the data adaptively (recursively). In each iteration, it finds the next split point, starting
    from the begining of the signal. The algorithm is as described above. The volatility of stock estimated/updated by
    exponential averaging signal variation range around previously fitted trendlines.

    :param y: values (e.g., stock prices).
    :param last_ind: Index of the datapoint to start from for fitting the trendline (>= start).
    :param incr: increments of data points to use in each iteration.
    :param min_break_len: minimum numbers of pints to pass the boundary lines to be counted as valid violation.
    :param symmetric: if True, same distance from the regression line will be used for lower/upper line biases.
    :param mv_ave_window: max distance between lower/upper boundary lines, used as a measure of volatility.
    :param min_points: min numbers of points to be used in each trendline.
    :param alpha: controls the weights on past and recent volatility measures for exponential averaging.

    :return: index of split, list of (slope, intercept) of lines, biases of boundary lines
    """
    slopes_intercepts = []
    biases = []
    splits = [0]
    n = len(y)

    max_range = estimate_volatility(y=y, mv_ave_window=mv_ave_window)

    start = 0
    end = last_ind
    while end < n:
        end, (a, b), cur_range, bias_lower, bias_upper = next_split_adaptive(y,
                                                                             start,
                                                                             end,
                                                                             n,
                                                                             incr,
                                                                             min_break_len,
                                                                             symmetric,
                                                                             max_range,
                                                                             min_points)
        splits.append(end)
        slopes_intercepts.append((a, b))
        biases.append((bias_lower, bias_upper))
        # max_range = alpha * cur_range + (1 - alpha) * max_range 
        start = end

    return splits, slopes_intercepts, biases

