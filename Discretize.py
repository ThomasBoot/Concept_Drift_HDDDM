import numpy as np
import pandas as pd

class Discretizer:

    def __init__(self, method):
        # method is either equalsize bins or equalquantile bins
        self.method = method

    def fit(self, data, names, to_ignore):
        """
        Determines how variables should be discretized.
        :param data: The dataset of interest.
        :param names: Indicates the variables of interest. If None, all variables are considered.
        :param to_ignore: Variables that do not need discretization, i.e. categorical variables.
        :return: An overview of numerical columns to be discretized.
        """

        columns = list(set(list(data)) - set(to_ignore))

        tmp = []
        for col in columns:
            tmp.append(len(data[col].unique()))
        tmp = np.array(tmp)
        threshold = np.mean(tmp)

        if names is None:
            self.numerical_cols = [col for col in columns if
                                   len(data[col].unique()) >= threshold and data[col].isnull().any() == False and data[col].dtype != 'object']
        else:
            self.numerical_cols = list(set(data.columns).intersection(names))

        return self

    def equalsize(self, col):

        bin_col, self.bins_output[col.name] = pd.cut(col, self.n_bins, retbins=True, duplicates='drop')
        return bin_col

    def equalquantile(self, col):

        bin_col, self.bins_output[col.name] = pd.qcut(col, self.n_bins, retbins=True, duplicates='drop')
        return bin_col

    def transform(self, data, n_bins):
        """
        Discretizes numerical data based on the preferred method.
        :param data: The data of interest.
        :param n_bins: The amount of bins to discretize in.
        :return: The dataset, in which numerical data is binned.
        """

        self.n_bins = n_bins
        self.bins_output = {}
        if self.method == "equalsize":
            data[self.numerical_cols] = data[self.numerical_cols].apply(self.equalsize,
                                                                        axis=0)  # apply function to each column
        if self.method == "equalquantile":
            data[self.numerical_cols] = data[self.numerical_cols].apply(self.equalquantile, axis=0)

        return data, self.bins_output