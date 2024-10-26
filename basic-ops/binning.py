# Discretization by Binning methods 
# Distance Binning and Frequency Binning

import pandas as pd
import numpy as np
 
df = pd.read_csv("../dataset/AmesHousingDataset.csv")

# 1. Distance binning
# Formula -> interval = (max-min) / Number of Bins
# Let us consider the 'Age' continuous value column for binning
min_value = df['Year Built'].min()
max_value = df['Year Built'].max()
print(min_value)
print(max_value)

# Suppose the bin size is 5
# linspace returns evenly spaced numbers over a specified interval. 
# Returns num evenly spaced samples, calculated over the interval [start, stop].
bins = np.linspace(min_value,max_value,5)
print(bins)

labels = ['Vintage', 'Old', 'Fairly New', 'Brand New']

# We can use the cut() function to convert the numeric values of the column Age into the categorical values.
# We need to specify the bins and the labels.
df['bins_dist'] = pd.cut(df['Year Built'], bins=bins, labels=labels, include_lowest=True)
print(df['bins_dist'].head())
# print(df['bins_dist'].values.tolist())

# If you want equal distribution of the items in your bins, use qcut . 
# If you want to define your own numeric bin ranges, then use cut

# 2. Frequency Binning
df['bin_freq'] = pd.qcut(df['Year Built'], q=4, precision=1, labels=labels)
print(df['bin_freq'].head())
# print(df['bin_freq'].values.tolist())