# Stock Factor Decomposition

This project focuses on constructing factor decompositions of various stocks, tracking their time evolution, and retroactively identifying statistically significant non-zero alpha periods. The aim is to understand how much of a stock’s performance can be explained by systematic risks versus firm-specific effects, and whether any episodes of genuine abnormal returns can be detected.

The methodology relies on asset-pricing models that express returns as a combination of factor exposures and an intercept term. The simplest version is the Capital Asset Pricing Model (CAPM), while the Fama–French models extend this framework by incorporating additional risk factors. By running regressions of stock returns on these factors, we obtain coefficients that describe sensitivities to different sources of systematic risk. The intercept, or alpha, represents the portion of returns not captured by the factors. Identifying statistically significant alpha is central, as it points to periods where returns exceeded expectations based on risk exposures, and thus may signal temporary arbitrage opportunities or missing elements in the model.

The project is organized into two main directories: the `/notebooks` folder, which contains the `stock-factor-decomposition.ipynb` notebook, and the `/src` folder, which holds the source code file `analysis.py` with all the functions used in the notebook.

---

## What It Does

- Downloads historical stock prices and factors from Yahoo Finance and [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu), respectively. 
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

- **`get_returns(tickers, start, end)`**:  
   &nbsp;&nbsp;&nbsp;This function retrieves a stock or a collection of stocks from **yfinance** between the dates specified by **start** and **end**, and converts the prices into simple returns. If **tickers** is a string containing a single ticker, the function returns the corresponding time series. If **tickers** is a list of tickers, the function returns a dictionary of the form ``{ticker: time series}``.

- **`get_factors(model, start, end)`**:  
   &nbsp;&nbsp;&nbsp;This function retrieves factor data between the dates specified by **start** and **end** from the [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu). The parameter **model** accepts three possible values: `CAPM`, `FF3`, and `FF5`. The function returns a dictionary of the form ``{factor label: time series}``.

- **`factor_regression(returns, factors, precision = 2, window = None, step = None)`**:  
  &nbsp;&nbsp;&nbsp;This function computes the factor exposures of a given stock, or a dictionary of stocks, represented by **returns**, against the factors provided in the **factors** parameter. The regression model is automatically inferred from the contents of **factors**.  
  &nbsp;&nbsp;&nbsp;The optional parameter **precision** controls rounding of the results. By default, the entire input time series is used for the regression. If **window** is specified, the function performs rolling regressions. The parameter **step** sets the distance between consecutive window positions and defaults to 1% of the total series length.  
  &nbsp;&nbsp;&nbsp;The function returns **coeffs**, **tstats**, **pstats**, and **r2**. The first three are dictionaries mapping factor labels to their estimated coefficients, t-statistics, and p-values, respectively. **r2** gives the R² score of the regression. If **returns** is a dictionary of the form ``{ticker: time series}``, then all output variables follow the same overarching structure.

- **`plot_coeffs_static(coeffs, r2 = None, title = "Factor decomposition")`**:  
  &nbsp;&nbsp;&nbsp;This function plots a histogram of factor loadings provided by the **coeffs** output of **factor_regression** with **window = None**. If **r2** is supplied, it also plots a separate histogram of the corresponding R² scores.

- **`plot_coeffs_static(coeffs, r2 = None, title = "Factor decomposition")`**:  
  &nbsp;&nbsp;&nbsp;This function plots the time evolution of factor loadings provided by the **coeffs** output of **factor_regression** with a specified **window**. Unlike in the static case, **coeffs** cannot contain only the factor decomposition of a *single* stock. If **r2** is supplied, the function also plots a separate chart showing the evolution of the corresponding R² scores.

- **`plot_coeffs(coeffs, r2 = None, title = "Factor decomposition")`**:  
  &nbsp;&nbsp;&nbsp;A universal plotting function that automatically distinguishes between the static and rolling cases based on the input and generates the appropriate plots accordingly.

- **`plot_alpha(coeffs, tstats, title = "Alpha and t-statistic over time")`**:  
  &nbsp;&nbsp;&nbsp;This function plots the time evolution of the annualized *alpha* and its t-statistic from the **coeffs** output of **factor_regression** using a specified **window**. It then highlights the intervals where the t-statistic indicates statistical significance.
