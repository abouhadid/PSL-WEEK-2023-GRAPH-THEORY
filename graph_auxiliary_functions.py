#######################################
##### simple operations on graphs #####
#######################################
from tqdm import tqdm
import numpy

def neighbourhood_dist(graph):
    degrees = []
    for v in tqdm(graph.nodes(data=False)):
        temp = 0
        for n in graph.neighbors(v):
            temp += graph.get_edge_data(v,n)["weight"]
        degrees.append(temp)
    return numpy.array(degrees)


def create_graph(df, start, end, threshold=0):
    corr_mat = df[(df.index >= start) & (df.index < end)].corr().dropna(how = 'all', axis=0).dropna(how = 'all', axis=1)
    corr_np_mat = corr_mat.to_numpy() - numpy.eye(len(corr_mat.index)) # removing self loops

    corr_np_mat[numpy.abs(corr_np_mat) < threshold] = 0 # censoring small correlations
    
    # deleting unlinked nodes
    corr_np_mat = corr_np_mat[~numpy.all(corr_np_mat == 0, axis=1)]
    corr_np_mat = corr_np_mat[:,~numpy.all(corr_np_mat == 0, axis=0)]
    return numpy.maximum(0, corr_np_mat)

#####################################################
##### Time series extraction from yahoo finance #####
#####################################################

# importing the time series

import yfinance as yf
import pandas
import bs4 as bs
import datetime as dt
import os
import pickle
import requests

# Imports as csv files the open, close, high, low and volume of the SP500 stocks
# For our purposes, we will only keep the open and close
# Code from stack overflow (https://stackoverflow.com/questions/58890570/python-yahoo-finance-download-all-sp-500-stocks)
def save_sp500_tickers():
    # Requests the composition of the sp500 on wikipedia and saves the tickers
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker[:-1])
    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)
    return tickers

def get_data_from_yahoo(reload_sp500=False, start=dt.datetime(2020, 1, 1), end=dt.datetime.now()):
    # Loads the csv files based on the web extraction from save_sp500_tickers
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    for ticker in tickers:
        # just in case your connection breaks, we'd like to save our progress!
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            df = yf.download(ticker, start=start, end=end)
            df.reset_index(inplace=True)
            df.set_index("Date", inplace=True)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))

def load_stocks(columns="Close"):
    # Loads the csv files and formats them as a pandas dataframe.
    df = pandas.DataFrame()
    for file in tqdm(os.listdir("stock_dfs/")):
        # loading the csv files and collating them into a single dataframe
        stock = pandas.read_csv("stock_dfs/"+file)
        stock.set_index('Date', inplace=True)
        df = df.join(stock[columns].astype(float), how="outer", rsuffix="_"+file.replace(".csv", ""))

    # dataframe cleaning and computing returns instead of prices
    df.fillna(method='bfill', inplace=True)
    df = df.diff()/df
    df.drop(df.index[0], inplace=True)
    df.fillna(0, inplace=True)
    df = df.loc[:, (df != 0).any(axis=0)]
    return df*1e2

############################################
##### Graph Dataset Creation functions #####
############################################

from torch_geometric.data import Data # base data structure for graph data
import random
import torch

def adjacency_to_edge_index(mat):
    # Function transforming an edjacency matrix into an edge index list and a edge weight list
    edge = [[],[]]
    weight = []
    for i in range(len(mat)):
        for j in range(len(mat)):
            if mat[i,j]!= 0:
                edge[0].append(i)
                edge[1].append(j)
                weight.append(mat[i,j])
    return edge, weight

def create_dataset(df, step=5, horizon=1, train_split=0.6, val_split = 0.2):
    # Function creating a dataset of stock market graphs with label the return at t + horizon.
    dataset = []
    for i in tqdm(range(len(df.index)//step - 1)):
        #for every step, compute the graph, the weights and save the features
        corr_mat = create_graph(df, df.index[5*i], df.index[5*(i+1)])
        X_features = df.iloc[5*i: 5*(i+1)].to_numpy().T # features are the returns of the 5 days
        edge_index, weight = adjacency_to_edge_index(corr_mat)

        y = (df.iloc[5*(i+1)+horizon].to_numpy())

        data = Data(x=torch.Tensor(X_features).float(),
                    edge_index=torch.Tensor(edge_index).long(),
                    edge_attr=torch.Tensor(weight).float(),
                    y=torch.Tensor(y).float())
        
        dataset.append(data)

    # shuffling the features and seperating them into train and test sets
    random.shuffle(dataset)
    train_sep = int(train_split*len(dataset))
    val_sep = int((train_split+val_split) * len(dataset))

    return dataset[:train_sep], dataset[train_sep:val_sep], dataset[val_sep:]

def create_classification_dataset(df, step=5, horizon=1, train_split=0.6, val_split = 0.2):
    # Function creating a dataset of stock market graphs with label the return at t + horizon.
    dataset = []
    for i in tqdm(range(len(df.index)//step - 1)):
        #for every step, compute the graph, the weights and save the features
        corr_mat = create_graph(df, df.index[5*i], df.index[5*(i+1)])
        X_features = df.iloc[5*i: 5*(i+1)].to_numpy().T # features are the returns of the 5 days
        edge_index, weight = adjacency_to_edge_index(corr_mat)

        # classification problem: will the returns be larger than the mean of the week?
        avg_pred = X_features.mean(axis=1)
        true_pred = (df.iloc[5*(i+1)+horizon].to_numpy())
        y = (avg_pred < true_pred)

        data = Data(x=torch.Tensor(X_features).float(),
                    edge_index=torch.Tensor(edge_index).long(),
                    edge_attr=torch.Tensor(weight).float(),
                    y=torch.Tensor(y).float())
        
        dataset.append(data)

    # shuffling the features and seperating them into train and test sets
    random.shuffle(dataset)
    train_sep = int(train_split*len(dataset))
    val_sep = int((train_split+val_split) * len(dataset))

    return dataset[:train_sep], dataset[train_sep:val_sep], dataset[val_sep:]
  