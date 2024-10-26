import pandas as pd

# Load the dataset
dataset_source_path='../dataset/AmesHousingDataset.csv'
stats_summary_path='../info/stats.csv'

df = pd.read_csv(dataset_source_path)
print(df.head())

# Summary statistics
print("\nSummary Statistics:")
df.describe(include='all').transpose().to_csv(stats_summary_path)

# Checking for missing values
# print("\nMissing Values:")
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(df.isnull().sum())

# Data type information
print("\nData Types:")
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(df.dtypes)