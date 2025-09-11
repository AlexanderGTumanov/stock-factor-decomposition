# Stock Factor Decomposition

This project focuses on constructing factor decompositions of various stocks, tracking their time evolution, and retroactively identifying statistically significant non-zero alpha periods. The aim is to understand how much of a stock’s performance can be explained by systematic risks versus firm-specific effects, and whether any episodes of genuine abnormal returns can be detected.

The methodology relies on asset-pricing models that express returns as a combination of factor exposures and an intercept term. The simplest version is the Capital Asset Pricing Model (CAPM), while the Fama–French models extend this framework by incorporating additional risk factors. By running regressions of stock returns on these factors, we obtain coefficients that describe sensitivities to different sources of systematic risk. The intercept, or alpha, represents the portion of returns not captured by the factors. Identifying statistically significant alpha is central, as it points to periods where returns exceeded expectations based on risk exposures, and thus may signal temporary arbitrage opportunities or missing elements in the model.

The project is organized into two main directories: the `/notebooks` folder, which contains the `stock-factor-decomposition.ipynb` notebook, and the `/src` folder, which holds the source code file `analysis.py` with all the functions used in the notebook. The time series data is retrieved from *yfinance*.

---

## What It Does

- Downloads historical stock prices from Yahoo Finance and convert them into simple returns.
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


## Contents of the `/data` folder

This folder contains the `default_tickers.py` module. In it, there is a balanced list of tickers from selected industries: *tech*, *energy*, *finance*, *healthcare*, *utilities*, *materials*, and *real estate*.

---

## Contents of the `/src` folder

The **model.py** file contains all the functions and tools used to perform analysis in the notebook. What follows is a brief description of them.

- **build_dataset(tickers, start, end, industries = None, shuffle = False, normalize = True, verbose = True)**:  
   &nbsp;&nbsp;&nbsp;This function retrieves a collection of time series from **yfinance** between the dates given by **start** and **end** that correspond to tickers contained in **tickers**. It then converts them to log returns. **tickers** must be organized in the form of a dictionary: **{'industry label': [tickers]}**. The industry labels are not used during training, but are recorded for comparisons with the model's predictions. **industries** is an optional variable that can be given as a list of labels from the dictionary. Only stocks with these labels will be considered. If not provided, all the stocks from **tickers** will be retrieved. The dataset constructed will be balanced: i.e., the number of stocks per label will be the same. If some of the stocks fail to be retrieved, the function will discard stocks from other industries until the balance is restored. If **shuffle** is **True**, this will happen randomly. If **normalize** is **True** (recommended), all the log return series retrieved will be normalized. If **verbose** is **True**, then the function will print the size of the dataset once it's constructed.
  
  This function returns **X**, **y**, **t**, and **index**.  
  &nbsp;&nbsp;&nbsp;**X** is a torch tensor that contains all the log return series.  
  &nbsp;&nbsp;&nbsp;**y** and **t** are not necessary for the model to work, but become useful post-training. The former contains all the industry labels of the stocks in **X**, while the latter records their tickers.  
  &nbsp;&nbsp;&nbsp;**index** contains the index of the extracted time series.

- **prepare_dataloaders(X, batch_size = 16, valid_split = 0.2, seed = 42)**:  
  &nbsp;&nbsp;&nbsp;Packages **X** into training and validation data loaders. **batch_size** controls the batch size, while **valid_split** specifies the proportion of data reserved for validation. The function returns *DataLoader()* objects **train_loader**, **valid_loader**.

- **DECModel(nn.Module)**, **Encoder(nn.Module)**, **ClusteringLayer(nn.Module)**:  
  &nbsp;&nbsp;&nbsp;Deep Embedding Clustering (DEC) neural network. Consists of two layers: **Encoder** and **ClusteringLayer**. **Encoder** is a sequential layer that reduces the dimensionality of the input data by transferring it into a low-dimensional latent space. When fully trained, this space retains only the degrees of freedom that are essential for the clustering task. **ClusteringLayer** uses a Student's t-distribution kernel to compute the soft assignment of each latent point to a cluster. **DECModel** combines the two and returns **q**, which is the soft cluster assignment for each point, and **z**, the latent representations of the input data after passing through the encoder.

- **target_distribution(q)**:  
  &nbsp;&nbsp;&nbsp;Computes the sharpened version of the soft cluster assignments **q**, used as the target distribution during DEC training to improve cluster purity. It amplifies confident assignments and suppresses uncertain ones.

- **train_model(train_loader, valid_loader, max_epochs = None, min_epochs = 100, patience = 50, lr = 1e-3, n_clusters = None, centers = None)**:  
  &nbsp;&nbsp;&nbsp;Creates and trains the model on the data from **train_loader** and **valid_loader**. By default, the number of epochs is unrestricted. Instead, the model is coded to stop once the validation loss curve flattens or starts growing. **max_epochs** can be provided to hard-limit the number of epochs, while **min_epochs** is necessary because of the characteristic profile of the DEC loss curves: they tend to go up and spike early on, before starting to slowly go down. The **patience** parameter prevents the model's training from stopping due to random loss spikes. **lr** controls the learning rate, and **n_clusters** determines the number of clusters the stocks will be divided into. The **centers** parameter can be used to initialize cluster centers pre-training. If not provided, the centers will be initialized through KMeans. This parameter is very useful when determining the time evolution of cluster arrangements: cluster output at a given time window's position can be used as a seed for the next one. This greatly speeds up the model, because due to the temporal continuity, the new model's starting point lands very close to the vacuum configuration that it seeks. The function outputs the trained model **model**, along with a pandas dataframe **history** containing loss and validation histories throughout the epochs.

- **predict_clusters(model, X)**:  
  &nbsp;&nbsp;&nbsp;Runs dataset **X** through the model and returns **z**, the latent-space representation of **X** obtained from the encoder, and **clusters**, the predicted cluster assignments.

- **tickers_by_cluster(clusters, y, t)**:  
  &nbsp;&nbsp;&nbsp;Builds a dataset with columns corresponding to clusters and rows to industry labels. Each entry contains the list of tickers from the given industry that were classified into the given cluster.

- **mixing_table(clusters, y, verbose = False)**:  
  &nbsp;&nbsp;&nbsp;Builds a dataset with columns corresponding to clusters and rows to industry labels. Each entry contains the proportion of the given industry within the given cluster. If **verbose** is True, the table is also printed.
