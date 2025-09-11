# Stock Factor Decomposition

This project focuses on constructing factor decompositions of various stocks, tracking their time evolution, and retroactively identifying statistically significant non-zero alpha periods. The aim is to understand how much of a stock’s performance can be explained by systematic risks versus firm-specific effects, and whether any episodes of genuine abnormal returns can be detected.

The methodology relies on asset-pricing models that express returns as a combination of factor exposures and an intercept term. The simplest version is the Capital Asset Pricing Model (CAPM), while the Fama–French models extend this framework by incorporating additional risk factors. By running regressions of stock returns on these factors, we obtain coefficients that describe sensitivities to different sources of systematic risk. The intercept, or alpha, represents the portion of returns not captured by the factors. Identifying statistically significant alpha is central, as it points to periods where returns exceeded expectations based on risk exposures, and thus may signal temporary arbitrage opportunities or missing elements in the model.

The project is organized into two main directories: the `/notebooks` folder, which contains the `stock-factor-decomposition.ipynb` notebook, and the `/src` folder, which holds the source code file `analysis.py` with all the functions used in the notebook. The time series data is retrieved from *yfinance*. Factors are downloaded from [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu).

---

## What It Does

- Downloads historical stock prices and factors from Yahoo Finance and Kenneth R. French Data Library, respectively. 
- Performs factor decompositions of stocks from different industries over an extended period and highlights industry-specific patterns in factor exposures.
- Analyzes the time evolution of stocks using rolling-window regressions, detects periods of non-zero alpha with t-statistics above 2, and links these episodes to real-world events that drove the spikes.

---

## How to Use

1. Clone this repository:
   ```bash
   git clone <https://github.com/AlexanderGTumanov/stock-factor-decomposition>
   cd <stock-factor-decomposition>

---

## Contents of the Notebook

Notebook `/notebooks/stock-factor-decomposition.ipynb` is divided into two sections. In the first, we perform static analysis over a period of market stability, which allows the model to infer industry labels based solely on stock return behavior. In the second, we extend the analysis to a dynamic setting: using a rolling window approach, we train the model at each position of the window and align cluster assignments across time steps. This setup spans a long time period that includes the 2020 COVID crash, which allows us to study how the clustering of stocks evolves during periods of market stress.

---

## Contents of the `/src` folder

The **analysis.py** file contains all the functions and tools used to perform analysis in the notebook. What follows is a brief description of them.

- **get_returns(tickers, start, end)**:  
   &nbsp;&nbsp;&nbsp;This function retrieves a stock or a collection of stocks from **yfinance** between the dates specified by **start** and **end**, and converts the prices into simple returns. If **tickers** is a string containing a single ticker, the function returns the corresponding time series. If **tickers** is a list of tickers, the function returns a dictionary of the form ``{ticker: time series}``.

- **get_factors(model, start, end)**:  
   &nbsp;&nbsp;&nbsp;This function retrieves **model** has three possible values: `CAPM`, `FF3`, and `FF5`.
