

import sys
import streamlit as st
from datetime import date, timedelta
import yfinance as yf
from plotly import graph_objs as go
import numpy as np

sys.path.insert(0, "../")
from autotrend.util import (estimate_volatility, 
                            linear_model_pred, 
                            draw_trendlines)
from autotrend.adaptive import trendlines_adaptive
from autotrend.config import DARK_COLORS


st.set_page_config(layout="wide")
st.title('Identifying Linear Trending Patterns in Stock Prices')


# 
def load_data(ticker, start, end):
    data = yf.download(ticker, start, end)
    return data


def get_draw_trendlines_adaptive(data,
                                 price_column='adjclose',
                                 last_ind=0,
                                 incr=1,
                                 min_break_len=14,
                                 min_points=60,
                                 mv_ave_window=60,
                                 symmetric=True,
                                 alpha=0.01,
                                 title=''):
    #
    ts, prices = list(data.index), list(data[price_column])
    x, y = list(range(len(ts))), prices
    
    # 
    splits, slopes_intercepts, biases = trendlines_adaptive(y=y,
                                                            last_ind=last_ind,
                                                            incr=incr,
                                                            min_break_len=min_break_len,
                                                            symmetric=symmetric,
                                                            mv_ave_window=mv_ave_window,
                                                            min_points=min_points,
                                                            alpha=alpha)

    lines, lower_lines, upper_lines = [], [], []

    #
    i = 0
    for j, (a, b), (bias_lower, bias_upper) in zip(splits[1:], slopes_intercepts, biases):
        x = list(range(i, j))
        lines.append((x, [linear_model_pred(k-i, a, b) for k in range(i, j)]))
        lower_lines.append(
            (x, [linear_model_pred(k-i, a, b - bias_lower) for k in range(i, j)]))
        upper_lines.append(
            (x, [linear_model_pred(k-i, a, b + bias_upper) for k in range(i, j)]))
        i = j

    # 
    colors = np.random.choice(DARK_COLORS, min(len(lines), len(DARK_COLORS)), replace=False)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=y, fillcolor='black'))
    
    for i in range(len(lines)):
        line, lower_line, upper_line = lines[i], lower_lines[i], upper_lines[i]

        _ = fig.add_trace(go.Scatter(x=[ts[i] for i in line[0]], 
                                     y=line[1], 
                                     line=dict(color=colors[i % len(colors)], width=1, dash='dash'))) 
        _ = fig.add_trace(go.Scatter(x=[ts[i] for i in lower_line[0]], 
                                     y=lower_line[1], 
                                     line=dict(color=colors[i % len(colors)], width=1))) #colors[i % len(colors)]))
        _ = fig.add_trace(go.Scatter(x=[ts[i] for i in upper_line[0]], 
                                     y=upper_line[1], 
                                     line=dict(color=colors[i % len(colors)], width=1))) #colors[i % len(colors)]))
    
    fig.layout.update(title_text=title, 
                      width=1600,
                      height=800,
                      xaxis_rangeslider_visible=True, 
                      xaxis = dict(showgrid=True, tickangle=90), 
                      showlegend=False)

    st.plotly_chart(fig)


# 
stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.selectbox('Tickers', stocks)

max_n_years = 10
n_days = st.sidebar.slider('Numbers of days:', 60, 365 * max_n_years, 180)
vol_est_window = st.sidebar.slider('Volatility estimation window:', 30, n_days, 90)
min_points = st.sidebar.slider('min points per segment:', 5, n_days, 50)
min_break_len = st.sidebar.slider('vilation tolerance:', 1, 100, 5)

TODAY = date.today().strftime("%Y-%m-%d")
START =(date.today() - timedelta(days=n_days)).strftime("%Y-%m-%d")


# loading data
data = load_data(ticker=selected_stock, start=START, end=TODAY)


# Ploting raw data
get_draw_trendlines_adaptive(data=data, 
                             price_column='Adj Close', 
                             mv_ave_window=vol_est_window,
                             title=f"Adjusgeted Close Price and Linear Trends for {selected_stock}", 
                             min_points=min_points,
                             min_break_len=min_break_len)
