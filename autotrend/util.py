

from typing import List, Tuple
import numpy as np
import pandas as pd
from numpy import linalg
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pickle as pk

from .config import DARK_COLORS


def get_lines_boundries(x: List[int],
                        y_hat: List[float],
                        y: List[float],
                        splits: List[float],
                        symmetric: bool = True,
                        issorted: bool = False) -> Tuple[List[List[Tuple[int, float]]],
                                                         List[List[Tuple[int, float]]],
                                                         List[List[Tuple[int, float]]]]:
    """
    Finds boundary lines for the fitted linear functions based on some heuristics

    :param x: index of passed values in original array
    :param y_hat: predicted linear function/signal
    :param y: real values (e.g., stock prices)
    :param splits: array of float specifying split locations
    :param symmetric: if True, intervals will be symmetric around the regression line.
    :param issorted: whether the splits are sorted or not

    :return: arrays of times and values for (lines, lower boundary lines, upper boundary lines)
    """
    # finding line splits
    lines = []

    splits = splits + [x[-1] + 1e6]
    if not issorted:
        splits.sort()

    n = len(x)
    i = 0
    for x1, x2 in zip(splits[:-1], splits[1:]):
        j = i
        while j < n and x[j] < x2:
            j += 1
        lines.append((x[i:j], y_hat[i:j]))
        i = j

    # defining boundary lines
    lines_valid, lower_lines, upper_lines = [], [], []
    for (x_hat, y_hat) in lines:
        if not x_hat:
            continue
        lines_valid.append((x_hat, y_hat))
        dists = [(y[ind] - y_hat[i]) for i, ind in enumerate(x_hat)]
        bias_up = max(dists)
        bias_down = min(dists)
        if symmetric:
            bias_down = bias_up = min(bias_up, -bias_down)
        lower_lines.append((x_hat, [y - bias_down for y in y_hat]))
        upper_lines.append((x_hat, [y + bias_up for y in y_hat]))

    return lines_valid, lower_lines, upper_lines


def get_slope(x1: int, x2: int, y1: float, y2: float) -> float:
    """
    get_slope of line

    :param x1: first point x coordinate
    :param x2: second point x coordinate
    :param y1: first point y coordinate
    :param y2: second point y coordinate

    :return:
    """
    return round((y2 - y1) / (x2 - x1), 2)


def get_peaks_signal(ts: List[datetime], prices: List[float], dt: timedelta) -> Tuple[Tuple[List[datetime], List[float]],
                                                                                      Tuple[List[datetime], List[float]]]:
    """
    Get peaks of signal (stock prices

    :param ts: datetime of (close) prices
    :param prices: stock prices
    :param dt: time difference to consider on each side of the current point for getting the peak stock price

    :return: (datetime of mins, values of mins), (datetime of maxes, values of maxes)
    """
    # can be made O(N) using stacks
    maxes, mins = [], []
    n = len(prices)
    for i in range(n):
        l = i
        while l > 1 and ts[i] - ts[l - 1] <= dt:
            l -= 1
        r = i
        while r < n and ts[r] - ts[i] <= dt:
            r += 1
        max_, min_ = max(prices[l:r]), min(prices[l:r])
        if prices[i] == max_:
            maxes.append((ts[i], prices[i]))
        elif prices[i] == min_:
            mins.append((ts[i], prices[i]))

    return ([t[0] for t in mins], [t[1] for t in mins]), ([t[0] for t in maxes], [t[1] for t in maxes])


def get_moving_aves(prices: List[float]) -> pd.DataFrame:
    """
    Getting regressing (exponential 15 day, 50 day, 200 day) moving averages of signal (stock price)

    :param prices: stock prices

    :return: a dataframe of
    """
    df = pd.DataFrame({"values": prices})
    df.dropna(inplace=True)
    df["ema_15day"] = df["values"].ewm(span=15, adjust=False).mean()
    df["fiftyDayAverage"] = df["values"].rolling(window=50).mean()
    df["twoHundredDayAverage"] = df["values"].rolling(window=200).mean()
    return df[["ema_15day", "fiftyDayAverage", "twoHundredDayAverage"]]


def last_valid_weekday(datetime_obj: datetime) -> datetime:
    """
    returns last valid day of stock market (skips the weekends)

    :param datetime_obj: datetime

    :return: datetime of last valid business day
    """
    dt = 4 - max(datetime_obj.weekday(), 4)
    datetime_obj -= timedelta(days=dt)

    return datetime.strptime(f"{datetime_obj.year}-{datetime_obj.month}-{datetime_obj.day} "
                             f"{datetime_obj.hour}:{datetime_obj.minute}:{datetime_obj.second}",
                             "%Y-%m-%d %H:%M:%S")


def draw_trendlines(ts: List[datetime],
                    y: List[float],
                    lines: List[List[Tuple[int, float]]],
                    lower_lines: List[List[Tuple[int, float]]],
                    upper_lines: List[List[Tuple[int, float]]],
                    volume: List[int] = None,
                    title: str = "",
                    y_log_scale: bool = False):
    """
    Draws stock prices and fitted lines (and volume changes if specified).

    :param ts: datetime of (close) prices
    :param y: stock prices
    :param lines: fitted linear functions
    :param lower_lines: lower boundary lines
    :param upper_lines: upper boundary lines
    :param volume: volumes of stocks traded
    :param title: plot title
    :param y_log_scale: if specified, the y axis of stock prices will be in log scale

    :return:
    """
    # plot stock prices and fitted lines
    n = len(ts)
    x = list(range(1, n + 1))
    fig, ax = plt.subplots(figsize=(35, 7))
    fig.patch.set_facecolor('1')
    _ = ax.plot(x, y)

    colors = np.random.choice(DARK_COLORS, min(len(lines), len(DARK_COLORS)), replace=False)
    for i in range(len(lines)):
        line, lower_line, upper_line = lines[i], lower_lines[i], upper_lines[i]
        _ = plt.plot(line[0], line[1], linewidth=2, color=colors[i % len(colors)], linestyle="--")
        _ = plt.plot(lower_line[0], lower_line[1], linewidth=2, color=colors[i % len(colors)])
        _ = plt.plot(upper_line[0], upper_line[1], linewidth=2, color=colors[i % len(colors)])

    if y_log_scale:
        _ = plt.yscale("log")

    _ = plt.title(title)
    _ = plt.grid(True)
    _ = plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    _ = plt.xticks(x[::20], ts[::20], rotation=90)

    if volume:
        #
        plt.show()
        #
        fig = plt.figure(figsize=(35, 3))
        fig.patch.set_facecolor('1')
        _ = plt.bar(ts, volume, label="volume", alpha=0.3)
        mvas = get_moving_aves(volume)
        colors = ["red", "green", "blue"]
        for color, c in zip(colors, mvas.columns):
            _ = plt.plot(ts, mvas[c], label=c, color=color)
        _ = plt.legend()
        _ = plt.title("volume and moving averages")
        _ = plt.grid(True)
        _ = plt.xticks(ts[::20], rotation=90)
        ax = plt.gca()
        ax.axes.xaxis.set_ticklabels([])
        plt.show()
        #
        fig = plt.figure(figsize=(35, 2))
        fig.patch.set_facecolor('1')
        ave_vol = np.mean(volume)
        std_vol = np.std(volume)
        normal_vol = [(v - ave_vol) / std_vol for v in volume]
        _ = plt.bar(ts, normal_vol)
        _ = ax.plot(ts, )
        _ = plt.grid(True)
        _ = plt.xticks(ts[::20], rotation=90)
        _ = plt.title("normalized volume")
        ax = plt.gca()
        ax.axes.xaxis.set_ticklabels([])
        plt.show()
        #
        fig = plt.figure(figsize=(35, 2))
        fig.patch.set_facecolor('1')
        vol_rate_change = [0.0] + [(v2 - v1) / v1 for v2, v1 in zip(volume[1:], volume[:len(volume) - 1])]
        _ = plt.plot(ts, vol_rate_change)
        _ = plt.grid(True)
        _ = plt.xticks(ts[::20], rotation=90)
        _ = plt.title("volume rate change")

    plt.show()


def estimate_volatility(y: List[float], mv_ave_window: int = 60):
    """
    Estimates values (e.g, stock prices) volatility by getting average of ranges for centered signal around symmetric
    moving average.

    :param y: values (e.g., stock prices)
    :param mv_ave_window: moving average window.

    :return:
    """
    y_ = [d for d in y if pd.notnull(d) and d is not None]
    n_ = len(y_)
    y_mvave = [float(np.mean(y_[max(0, i - mv_ave_window // 2): i + mv_ave_window // 2 + 1])) for i in range(n_)]
    y_centered = [p - ave if pd.notnull(ave) else 0 for p, ave in zip(y_, y_mvave)]
    ave_ranges = [np.max(y_centered[max(0, i-mv_ave_window):i+1]) - np.min(y_centered[max(0, i-mv_ave_window):i+1])
                  for i in range(n_)]

    return max(1e-6, np.mean(ave_ranges))


def get_range(bias_lower: float, bias_upper: float, a: float, vertical=True) -> float:
    """
    distance of boundary lines as estimated range of volatility

    :param bias_lower: lower trendline bias
    :param bias_upper: upper trendline bias
    :param a: slope
    :param vertical: if False, gets closes distance of lines else vertical distance

    :return: distance of lines
    """
    dist = bias_upper + bias_lower
    if not vertical:
        dist *= np.cos(np.arctan(a))

    return dist


def is_in_range(bias_lower: float, bias_upper: float, a: float, x: int, val: float) -> bool:
    """
    Checks if the value is in range of boundary lines .

    :param bias_lower: lower trendline bias
    :param bias_upper: upper trendline bias
    :param a: line slope
    :param x: index of the value in the signal.
    :param val: value (e.g., stock price)

    :return: boolean value
    """
    return linear_model_pred(x, a, bias_lower) <= val <= linear_model_pred(x, a, bias_upper)


def fit_linear_model(y: List[float]) -> Tuple[List[float], np.ndarray, float]:
    """
    Fits a linear model using least square to the passed data points. Uses indices as the predicates.

    :param y: values (e.g., stock prices)

    :return: optimum line parameters, predicates matrix, sum of squared of residuals
    """
    A = np.array([[1, i] for i, _ in enumerate(y)])
    params, ssr, _, _ = linalg.lstsq(A, y, rcond=None)
    params = [d for d in params]

    return params, A, ssr


def linear_model_pred(x: int, a: float, b:float) -> float:
    """
    Linear model predictions

    :param x: predicats (index)
    :param a: slope
    :param b: intercept

    :return: value
    """
    return a * x + b


def get_biases(y: List[float], start: int, end: int, a: float, b: float, symmetric: bool) -> Tuple[float, float]:
    """
    Finds biases for the upper and lower boundary lines.

    :param y: values (e.g., stock prices)
    :param start: start index of the fitted line.
    :param end: end index for finding the biases.
    :param a: slope
    :param b: intercept
    :param symmetric: if True, same (min) distance from the regression line will be used for lower/upper line biases.

    :return: lower and upper line biases.
    """
    diffs = [y[i] - linear_model_pred(i - start, a, b) for i in range(start, end)]
    bias_lower = -min(diffs)
    bias_upper = max(diffs)
    if symmetric:
        bias_upper = bias_lower = min(bias_upper, bias_lower)

    return bias_lower, bias_upper


def get_df_from_yf_historic_data_dump(dump):    
    # reforms historic data dump returned from yf API into a dataframe format
    d = dump['chart']['result'][0]
    dates = [datetime.fromtimestamp(ts) for ts in d["timestamp"]]
    d_ = {'dates': dates,               
          'adjclose': d['indicators']['adjclose'][0]['adjclose']}
    for k, l in d['indicators']['quote'][0].items():
        d_[k] = l

    df = pd.DataFrame(d_)
    df.set_index('dates', inplace=True)

    return df

