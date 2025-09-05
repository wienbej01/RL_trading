import pandas as pd

# Load the features (use repo-relative default path)
features_path = 'data/features/SPY_features.parquet'
features = pd.read_parquet(features_path)

# Print feature info
print("Features Info:")
features.info()

# Load VIX data
vix_path = 'data/external/vix.parquet'
vix_data = pd.read_parquet(vix_path)

# Print VIX info
print("\nVIX Data Info:")
vix_data.info()
print("\nVIX Data Head:")
print(vix_data.head())

# Calculate the correlation matrix
correlation_matrix = features.corr()

# Print the correlation matrix
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Select a subset of 10 features
selected_features = [
    'returns',
    'sma_50',
    'rsi_14',
    'bb_width',
    'vol_of_vol',
    'sma_slope',
    'obv',
    'fvg',
    'vix_close_^vix',
    'time_from_open'
]

print("\nSelected Features:")
print(selected_features)
