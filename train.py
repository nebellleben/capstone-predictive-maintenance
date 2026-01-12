#!/usr/bin/env python
# coding: utf-8

# ## Aircraft Predictive Maintenance
# 
# Dataset:
# - https://www.kaggle.com/datasets/maternusherold/pred-maintanance-data?resource=download
# - Copied to /dataset
# 
# Include:
# - Data preparation and data cleaning
# - EDA, feature importance analysis
# - Model selection process and parameter tuning
# 
# ## Problem Overview
# Predict the Remaining Useful Life (RUL) of aircraft engines using sensor data. This is a time-series regression problem.
# 

# In[ ]:


# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


# ## 1. Data Loading

# In[ ]:


# Define data directory and column names
DATA_DIR = Path('dataset')

# Column names: engine_id, cycle, and 21 sensors
column_names = ['engine_id', 'cycle'] + [f'sensor_{i:02d}' for i in range(1, 22)]

# Load training data
# Note: Using sep=r'\s+' to handle multiple spaces correctly
print("Loading training data...")
train_df = pd.read_csv(
    DATA_DIR / 'PM_train.txt',
    sep=r'\s+',
    header=None,
    names=column_names,
    usecols=range(23)  # Only use first 23 columns
)

print(f"Training data shape: {train_df.shape}")
print(f"Number of engines: {train_df['engine_id'].nunique()}")
print(f"Columns: {train_df.columns.tolist()}")
train_df.head()


# In[ ]:


# Load test data
print("Loading test data...")
test_df = pd.read_csv(
    DATA_DIR / 'PM_test.txt',
    sep=r'\s+',
    header=None,
    names=column_names,
    usecols=range(23)  # Only use first 23 columns
)

print(f"Test data shape: {test_df.shape}")
print(f"Number of engines: {test_df['engine_id'].nunique()}")
test_df.head()


# In[ ]:


# Load ground truth RUL values for test set
print("Loading ground truth RUL values...")
truth_df = pd.read_csv(
    DATA_DIR / 'PM_truth.txt',
    header=None,
    names=['RUL']
)

print(f"Ground truth shape: {truth_df.shape}")
truth_df.head()


# ## 2. Initial Data Exploration

# In[ ]:


# Basic information about training data
print("=== Training Data Info ===")
print(f"Shape: {train_df.shape}")
print(f"\nNumber of unique engines: {train_df['engine_id'].nunique()}")
print(f"Engine IDs range: {train_df['engine_id'].min()} to {train_df['engine_id'].max()}")
print(f"\nCycle range: {train_df['cycle'].min()} to {train_df['cycle'].max()}")
print(f"\nData types:\n{train_df.dtypes}")
print(f"\nMissing values:\n{train_df.isnull().sum().sum()} total missing values")
print(f"\nMissing values per column:\n{train_df.isnull().sum()}")


# In[ ]:


# Distribution of cycles per engine
cycles_per_engine = train_df.groupby('engine_id')['cycle'].max()
print("=== Cycles per Engine ===")
print(f"Min cycles: {cycles_per_engine.min()}")
print(f"Max cycles: {cycles_per_engine.max()}")
print(f"Mean cycles: {cycles_per_engine.mean():.2f}")
print(f"Median cycles: {cycles_per_engine.median():.2f}")

plt.figure(figsize=(10, 6))
plt.hist(cycles_per_engine, bins=50, edgecolor='black')
plt.xlabel('Max Cycles per Engine')
plt.ylabel('Frequency')
plt.title('Distribution of Maximum Cycles per Engine')
plt.grid(True, alpha=0.3)
plt.show()


# In[ ]:


# Basic statistics for sensors
print("=== Sensor Statistics ===")
sensor_cols = [f'sensor_{i:02d}' for i in range(1, 22)]
train_df[sensor_cols].describe()


# In[ ]:


# Check for constant or low-variance sensors
print("=== Sensor Variance Analysis ===")
sensor_variance = train_df[sensor_cols].var().sort_values()
print("Sensors with lowest variance:")
print(sensor_variance.head(10))
print("\nSensors with highest variance:")
print(sensor_variance.tail(10))

# Identify constant sensors (variance = 0 or very close to 0)
constant_sensors = sensor_variance[sensor_variance < 1e-10]
if len(constant_sensors) > 0:
    print(f"\nConstant sensors (variance < 1e-10): {constant_sensors.index.tolist()}")
else:
    print("\nNo constant sensors found")


# ## 3. Data Preprocessing and Cleaning

# In[ ]:


# Create a copy for preprocessing
train_clean = train_df.copy()
test_clean = test_df.copy()

# Check for missing values
print("=== Missing Values Check ===")
print(f"Training missing values: {train_clean.isnull().sum().sum()}")
print(f"Test missing values: {test_clean.isnull().sum().sum()}")

# Check for infinite values
print(f"\nTraining infinite values: {np.isinf(train_clean.select_dtypes(include=[np.number])).sum().sum()}")
print(f"Test infinite values: {np.isinf(test_clean.select_dtypes(include=[np.number])).sum().sum()}")

# If there are missing values, we'll handle them (forward fill for time-series)
if train_clean.isnull().sum().sum() > 0:
    train_clean = train_clean.fillna(method='ffill').fillna(method='bfill')
    print("\nFilled missing values using forward fill and backward fill")


# In[ ]:


# Identify constant sensors (sensors with zero or very low variance)
# These sensors don't provide useful information for prediction
sensor_variance = train_clean[sensor_cols].var()
constant_threshold = 1e-10
constant_sensors = sensor_variance[sensor_variance < constant_threshold].index.tolist()

print(f"=== Constant Sensors (variance < {constant_threshold}) ===")
if constant_sensors:
    print(f"Found {len(constant_sensors)} constant sensors: {constant_sensors}")
    print("These will be excluded from feature engineering")
else:
    print("No constant sensors found")

# Store for later use
sensors_to_exclude = constant_sensors.copy()


# In[ ]:


# Calculate RUL for training data
# RUL = max_cycles_per_engine - current_cycle
print("=== Calculating RUL ===")

# Get maximum cycle for each engine
max_cycles = train_clean.groupby('engine_id')['cycle'].max().reset_index()
max_cycles.columns = ['engine_id', 'max_cycle']

# Merge with training data
train_clean = train_clean.merge(max_cycles, on='engine_id', how='left')

# Calculate RUL
train_clean['RUL'] = train_clean['max_cycle'] - train_clean['cycle']

print(f"RUL range: {train_clean['RUL'].min()} to {train_clean['RUL'].max()}")
print(f"Mean RUL: {train_clean['RUL'].mean():.2f}")
print(f"Median RUL: {train_clean['RUL'].median():.2f}")

# Apply piecewise linear degradation (cap RUL at 125 cycles)
# This is a common approach in predictive maintenance
RUL_CAP = 125
train_clean['RUL_capped'] = train_clean['RUL'].clip(upper=RUL_CAP)

print(f"\nAfter capping at {RUL_CAP} cycles:")
print(f"RUL_capped range: {train_clean['RUL_capped'].min()} to {train_clean['RUL_capped'].max()}")
print(f"Mean RUL_capped: {train_clean['RUL_capped'].mean():.2f}")

# Use capped RUL as target
train_clean['target'] = train_clean['RUL_capped']


# In[ ]:


# Visualize RUL distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Original RUL
axes[0].hist(train_clean['RUL'], bins=50, edgecolor='black', alpha=0.7)
axes[0].axvline(RUL_CAP, color='r', linestyle='--', linewidth=2, label=f'Cap at {RUL_CAP}')
axes[0].set_xlabel('RUL (cycles)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Original RUL Distribution')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Capped RUL
axes[1].hist(train_clean['RUL_capped'], bins=50, edgecolor='black', alpha=0.7, color='green')
axes[1].set_xlabel('RUL (cycles)')
axes[1].set_ylabel('Frequency')
axes[1].set_title(f'Capped RUL Distribution (max {RUL_CAP})')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# In[ ]:


# Data validation: Check for data leakage and temporal ordering
print("=== Data Validation ===")

# Check temporal ordering (cycles should be increasing for each engine)
for engine_id in train_clean['engine_id'].unique()[:5]:  # Check first 5 engines
    engine_data = train_clean[train_clean['engine_id'] == engine_id].sort_values('cycle')
    cycles = engine_data['cycle'].values
    is_sorted = np.all(cycles[:-1] <= cycles[1:])
    if not is_sorted:
        print(f"WARNING: Engine {engine_id} cycles are not sorted!")
    else:
        print(f"Engine {engine_id}: Cycles are properly ordered")

# Check RUL calculation
# RUL should decrease as cycle increases
sample_engine = train_clean[train_clean['engine_id'] == 1].sort_values('cycle')
print(f"\nSample engine 1: Cycle {sample_engine['cycle'].iloc[0]} -> RUL {sample_engine['RUL'].iloc[0]}")
print(f"Sample engine 1: Cycle {sample_engine['cycle'].iloc[-1]} -> RUL {sample_engine['RUL'].iloc[-1]}")
print(f"RUL decreases: {sample_engine['RUL'].iloc[0] > sample_engine['RUL'].iloc[-1]}")

print("\nData validation complete!")


# ## 4. Feature Engineering

# In[ ]:


# Feature engineering function
def create_features(df, sensors_to_exclude=None):
    """
    Create engineered features from raw sensor data.

    Features include:
    - Time-based features
    - Rolling statistics
    - Degradation indicators
    - Engine-level aggregations
    - Sensor interactions
    """
    df = df.copy()

    # Get sensor columns (exclude constant sensors)
    if sensors_to_exclude is None:
        sensors_to_exclude = []
    sensor_cols = [col for col in df.columns if col.startswith('sensor_') and col not in sensors_to_exclude]

    # Sort by engine_id and cycle to ensure proper ordering
    df = df.sort_values(['engine_id', 'cycle']).reset_index(drop=True)

    # === Time-based features ===
    df['cycle_norm'] = df.groupby('engine_id')['cycle'].transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1))

    # === Rolling statistics (window = 5 cycles) ===
    window = 5
    for sensor in sensor_cols:
        # Rolling mean
        df[f'{sensor}_rolling_mean'] = df.groupby('engine_id')[sensor].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        # Rolling std
        df[f'{sensor}_rolling_std'] = df.groupby('engine_id')[sensor].transform(
            lambda x: x.rolling(window=window, min_periods=1).std().fillna(0)
        )

    # === Degradation indicators (rate of change) ===
    for sensor in sensor_cols:
        # Rate of change (difference from previous cycle)
        df[f'{sensor}_diff'] = df.groupby('engine_id')[sensor].diff().fillna(0)
        # Rate of change normalized by cycle
        df[f'{sensor}_diff_norm'] = df[f'{sensor}_diff'] / (df['cycle'] + 1)

    # === Engine-level aggregations ===
    engine_stats = df.groupby('engine_id')[sensor_cols].agg(['max', 'min', 'mean', 'std']).reset_index()
    engine_stats.columns = ['engine_id'] + [f'{col}_{stat}' for col in sensor_cols for stat in ['max', 'min', 'mean', 'std']]

    # Merge engine-level stats
    df = df.merge(engine_stats, on='engine_id', how='left')

    # === Sensor interactions (ratios between related sensors) ===
    # Create ratios for sensors that might be related (e.g., temperature/pressure ratios)
    # We'll create a few key ratios based on sensor indices
    if len(sensor_cols) >= 4:
        # Ratio of sensor 1 to sensor 2 (if they exist)
        if f'sensor_01' in sensor_cols and f'sensor_02' in sensor_cols:
            df['sensor_ratio_01_02'] = df['sensor_01'] / (df['sensor_02'] + 1e-10)
        # Ratio of sensor 3 to sensor 4
        if f'sensor_03' in sensor_cols and f'sensor_04' in sensor_cols:
            df['sensor_ratio_03_04'] = df['sensor_03'] / (df['sensor_04'] + 1e-10)

    return df

print("Feature engineering function created!")


# In[ ]:


# Apply feature engineering to training data
print("Creating features for training data...")
train_features = create_features(train_clean, sensors_to_exclude=sensors_to_exclude)

print(f"Original columns: {len(train_clean.columns)}")
print(f"After feature engineering: {len(train_features.columns)}")
print(f"New features created: {len(train_features.columns) - len(train_clean.columns)}")

# Display feature columns
feature_cols = [col for col in train_features.columns 
                if col not in ['engine_id', 'cycle', 'max_cycle', 'RUL', 'RUL_capped', 'target'] 
                and not col.startswith('sensor_')]
print(f"\nEngineered feature columns ({len(feature_cols)}):")
print(feature_cols[:20])  # Show first 20


# In[ ]:


# Prepare feature matrix for modeling
# Select all features except metadata columns
exclude_cols = ['engine_id', 'cycle', 'max_cycle', 'RUL', 'RUL_capped', 'target']
feature_cols_all = [col for col in train_features.columns if col not in exclude_cols]

# Also exclude original sensor columns if we want to use only engineered features
# Or keep them - we'll let the model decide
# For now, keep all features
X_train = train_features[feature_cols_all].copy()
y_train = train_features['target'].copy()

print(f"Feature matrix shape: {X_train.shape}")
print(f"Target shape: {y_train.shape}")
print(f"\nFeature columns: {len(feature_cols_all)}")
print(f"Sample features: {feature_cols_all[:10]}")


# In[ ]:


# Check for any remaining issues in features
print("=== Feature Quality Check ===")
print(f"Missing values in features: {X_train.isnull().sum().sum()}")
print(f"Infinite values in features: {np.isinf(X_train.select_dtypes(include=[np.number])).sum().sum()}")

# Fill any remaining NaN values (shouldn't be any, but just in case)
if X_train.isnull().sum().sum() > 0:
    X_train = X_train.fillna(0)
    print("Filled NaN values with 0")

# Replace infinite values
if np.isinf(X_train.select_dtypes(include=[np.number])).sum().sum() > 0:
    X_train = X_train.replace([np.inf, -np.inf], 0)
    print("Replaced infinite values with 0")

print("Feature engineering complete!")


# ## 5. Exploratory Data Analysis (EDA)

# In[ ]:


# Univariate Analysis: RUL Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# RUL histogram
axes[0].hist(y_train, bins=50, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('RUL (cycles)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Target Variable (RUL) Distribution')
axes[0].grid(True, alpha=0.3)

# RUL box plot
axes[1].boxplot(y_train, vert=True)
axes[1].set_ylabel('RUL (cycles)')
axes[1].set_title('RUL Box Plot')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"RUL Statistics:")
print(f"Mean: {y_train.mean():.2f}")
print(f"Median: {y_train.median():.2f}")
print(f"Std: {y_train.std():.2f}")
print(f"Min: {y_train.min():.2f}")
print(f"Max: {y_train.max():.2f}")


# In[ ]:


# Univariate Analysis: Sample sensor distributions
# Plot distributions of a few key sensors
sample_sensors = ['sensor_01', 'sensor_02', 'sensor_03', 'sensor_04']
if all(col in X_train.columns for col in sample_sensors):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, sensor in enumerate(sample_sensors):
        axes[idx].hist(X_train[sensor], bins=50, edgecolor='black', alpha=0.7)
        axes[idx].set_xlabel(sensor)
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_title(f'{sensor} Distribution')
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# In[ ]:


# Bivariate Analysis: Correlation matrix for original sensors
original_sensors = [col for col in X_train.columns if col.startswith('sensor_') and not any(x in col for x in ['rolling', 'diff', 'ratio', '_max', '_min', '_mean', '_std'])]

if len(original_sensors) > 0:
    # Sample a subset for visualization (too many sensors)
    sensor_sample = original_sensors[:10] if len(original_sensors) >= 10 else original_sensors

    corr_matrix = X_train[sensor_sample].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix: Original Sensors (Sample)')
    plt.tight_layout()
    plt.show()


# In[ ]:


# Time-Series Analysis: Sensor degradation patterns
# Visualize how sensors change over cycles for a few sample engines
sample_engines = [1, 2, 3]
sample_sensors_viz = ['sensor_01', 'sensor_02', 'sensor_03']

fig, axes = plt.subplots(len(sample_engines), 1, figsize=(14, 4*len(sample_engines)))

for idx, engine_id in enumerate(sample_engines):
    engine_data = train_features[train_features['engine_id'] == engine_id].sort_values('cycle')

    for sensor in sample_sensors_viz:
        if sensor in engine_data.columns:
            axes[idx].plot(engine_data['cycle'], engine_data[sensor], 
                          label=sensor, alpha=0.7, linewidth=2)

    axes[idx].set_xlabel('Cycle')
    axes[idx].set_ylabel('Sensor Value')
    axes[idx].set_title(f'Engine {engine_id}: Sensor Trends Over Time')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# In[ ]:


# Time-Series Analysis: RUL vs Cycle for sample engines
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: RUL over cycles for sample engines
sample_engines = [1, 2, 3, 4, 5]
for engine_id in sample_engines:
    engine_data = train_features[train_features['engine_id'] == engine_id].sort_values('cycle')
    axes[0].plot(engine_data['cycle'], engine_data['RUL'], 
                label=f'Engine {engine_id}', alpha=0.7, linewidth=2)

axes[0].set_xlabel('Cycle')
axes[0].set_ylabel('RUL (cycles)')
axes[0].set_title('RUL Degradation Over Cycles (Sample Engines)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Average RUL by cycle across all engines
avg_rul_by_cycle = train_features.groupby('cycle')['RUL'].mean()
axes[1].plot(avg_rul_by_cycle.index, avg_rul_by_cycle.values, 
            linewidth=2, color='red', alpha=0.7)
axes[1].set_xlabel('Cycle')
axes[1].set_ylabel('Average RUL (cycles)')
axes[1].set_title('Average RUL Across All Engines')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# In[ ]:


# Feature Importance: Correlation with RUL
from scipy.stats import pearsonr

# Calculate correlation of each feature with RUL
feature_correlations = []
for col in X_train.columns:
    if X_train[col].dtype in [np.float64, np.int64]:
        try:
            corr, p_value = pearsonr(X_train[col], y_train)
            feature_correlations.append({
                'feature': col,
                'correlation': abs(corr),
                'p_value': p_value
            })
        except:
            pass

# Create DataFrame and sort by correlation
corr_df = pd.DataFrame(feature_correlations)
corr_df = corr_df.sort_values('correlation', ascending=False)

# Display top 20 features by correlation
print("=== Top 20 Features by Correlation with RUL ===")
print(corr_df.head(20))

# Visualize top features
top_n = 15
plt.figure(figsize=(10, 8))
top_features = corr_df.head(top_n)
plt.barh(range(len(top_features)), top_features['correlation'].values)
plt.yticks(range(len(top_features)), top_features['feature'].values)
plt.xlabel('Absolute Correlation with RUL')
plt.title(f'Top {top_n} Features by Correlation with RUL')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# In[ ]:


# Feature Importance: Mutual Information (from scikit-learn)
from sklearn.feature_selection import mutual_info_regression

# Sample data for faster computation (mutual info can be slow on large datasets)
sample_size = min(5000, len(X_train))
sample_idx = np.random.choice(len(X_train), sample_size, replace=False)
X_sample = X_train.iloc[sample_idx]
y_sample = y_train.iloc[sample_idx]

# Calculate mutual information
print("Calculating mutual information scores (this may take a moment)...")
mi_scores = mutual_info_regression(X_sample, y_sample, random_state=42)

# Create DataFrame
mi_df = pd.DataFrame({
    'feature': X_train.columns,
    'mutual_info': mi_scores
}).sort_values('mutual_info', ascending=False)

# Display top 20 features
print("\n=== Top 20 Features by Mutual Information ===")
print(mi_df.head(20))

# Visualize top features
top_n = 15
plt.figure(figsize=(10, 8))
top_features_mi = mi_df.head(top_n)
plt.barh(range(len(top_features_mi)), top_features_mi['mutual_info'].values)
plt.yticks(range(len(top_features_mi)), top_features_mi['feature'].values)
plt.xlabel('Mutual Information Score')
plt.title(f'Top {top_n} Features by Mutual Information')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# In[ ]:


# Summary: Combine correlation and mutual information for feature ranking
feature_importance = corr_df.merge(mi_df, on='feature', how='outer')
feature_importance['correlation'] = feature_importance['correlation'].fillna(0)
feature_importance['mutual_info'] = feature_importance['mutual_info'].fillna(0)

# Normalize scores (0-1 scale) and create combined score
feature_importance['corr_norm'] = (feature_importance['correlation'] - feature_importance['correlation'].min()) / (feature_importance['correlation'].max() - feature_importance['correlation'].min() + 1e-10)
feature_importance['mi_norm'] = (feature_importance['mutual_info'] - feature_importance['mutual_info'].min()) / (feature_importance['mutual_info'].max() - feature_importance['mutual_info'].min() + 1e-10)

# Combined score (weighted average)
feature_importance['combined_score'] = 0.5 * feature_importance['corr_norm'] + 0.5 * feature_importance['mi_norm']
feature_importance = feature_importance.sort_values('combined_score', ascending=False)

print("=== Top 20 Features by Combined Importance Score ===")
print(feature_importance[['feature', 'correlation', 'mutual_info', 'combined_score']].head(20))

# Save feature importance for later use
feature_importance.to_csv('feature_importance.csv', index=False)
print("\nFeature importance saved to 'feature_importance.csv'")


# ## 6. Model Training

# In[ ]:


# Train-validation split (engine-based split to avoid data leakage)
# Use 80% of engines for training, 20% for validation
print("=== Creating Train/Validation Split ===")

# Get engine IDs
engine_ids = train_features['engine_id'].values

# Split by engines, not by samples
unique_engines = np.unique(engine_ids)
np.random.seed(42)
np.random.shuffle(unique_engines)
n_train_engines = int(len(unique_engines) * 0.8)
train_engines = set(unique_engines[:n_train_engines])
val_engines = set(unique_engines[n_train_engines:])

train_mask = np.array([eid in train_engines for eid in engine_ids])
val_mask = np.array([eid in val_engines for eid in engine_ids])

X_train_split, y_train_split = X_train[train_mask], y_train[train_mask]
X_val_split, y_val_split = X_train[val_mask], y_train[val_mask]

print(f"Train engines: {len(train_engines)}")
print(f"Validation engines: {len(val_engines)}")
print(f"Train samples: {X_train_split.shape[0]}")
print(f"Validation samples: {X_val_split.shape[0]}")


# In[ ]:


# Evaluation function
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(model, X_train, y_train, X_val, y_val, model_name):
    """Evaluate model performance."""
    # Training predictions
    y_train_pred = model.predict(X_train)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)

    # Validation predictions
    y_val_pred = model.predict(X_val)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_r2 = r2_score(y_val, y_val_pred)

    print(f"\n{model_name} Performance:")
    print(f"  Train - MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
    print(f"  Val   - MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")

    return {
        'model_name': model_name,
        'train_mae': train_mae,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'val_mae': val_mae,
        'val_rmse': val_rmse,
        'val_r2': val_r2,
        'model': model
    }

print("Evaluation function defined!")


# ### 6.1 Baseline Models

# In[ ]:


# Train baseline models
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

print("="*60)
print("Training Baseline Models")
print("="*60)

baseline_results = []

# 1. Linear Regression
print("\n1. Linear Regression")
lr = LinearRegression()
lr.fit(X_train_split, y_train_split)
baseline_results.append(evaluate_model(lr, X_train_split, y_train_split, X_val_split, y_val_split, "Linear Regression"))

# 2. Ridge Regression
print("\n2. Ridge Regression")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_split, y_train_split)
baseline_results.append(evaluate_model(ridge, X_train_split, y_train_split, X_val_split, y_val_split, "Ridge Regression"))

# 3. Random Forest
print("\n3. Random Forest")
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train_split, y_train_split)
baseline_results.append(evaluate_model(rf, X_train_split, y_train_split, X_val_split, y_val_split, "Random Forest"))


# ### 6.2 Hyperparameter Optimization

# In[ ]:


# Hyperparameter optimization functions
import xgboost as xgb
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler

def optimize_xgboost(trial, X_train, y_train, X_val, y_val):
    """Optimize XGBoost hyperparameters."""
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'hist',
        'random_state': 42,
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
    }

    model = xgb.XGBRegressor(**params, n_jobs=-1)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)

    return mae


def optimize_lightgbm(trial, X_train, y_train, X_val, y_val):
    """Optimize LightGBM hyperparameters."""
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'random_state': 42,
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
    }

    model = lgb.LGBMRegressor(**params, n_jobs=-1, verbose=-1)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])

    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)

    return mae

print("Optimization functions defined!")


# In[ ]:


# Optimize XGBoost
print("="*60)
print("Optimizing XGBoost")
print("="*60)

study_xgb = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
study_xgb.optimize(
    lambda trial: optimize_xgboost(trial, X_train_split, y_train_split, X_val_split, y_val_split),
    n_trials=30,
    show_progress_bar=True
)

print(f"\nBest XGBoost MAE: {study_xgb.best_value:.4f}")
print(f"Best XGBoost params: {study_xgb.best_params}")

# Train best XGBoost model
xgb_best = xgb.XGBRegressor(**study_xgb.best_params, objective='reg:squarederror', 
                             eval_metric='rmse', tree_method='hist', random_state=42, n_jobs=-1)
xgb_best.fit(X_train_split, y_train_split, eval_set=[(X_val_split, y_val_split)], verbose=False)
xgb_result = evaluate_model(xgb_best, X_train_split, y_train_split, X_val_split, y_val_split, "XGBoost (Optimized)")


# In[ ]:


# Optimize LightGBM
print("="*60)
print("Optimizing LightGBM")
print("="*60)

study_lgb = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
study_lgb.optimize(
    lambda trial: optimize_lightgbm(trial, X_train_split, y_train_split, X_val_split, y_val_split),
    n_trials=30,
    show_progress_bar=True
)

print(f"\nBest LightGBM MAE: {study_lgb.best_value:.4f}")
print(f"Best LightGBM params: {study_lgb.best_params}")

# Train best LightGBM model
lgb_best = lgb.LGBMRegressor(**study_lgb.best_params, objective='regression', 
                              metric='mae', boosting_type='gbdt', random_state=42, 
                              n_jobs=-1, verbose=-1)
lgb_best.fit(X_train_split, y_train_split, eval_set=[(X_val_split, y_val_split)], 
             callbacks=[lgb.early_stopping(50, verbose=False)])
lgb_result = evaluate_model(lgb_best, X_train_split, y_train_split, X_val_split, y_val_split, "LightGBM (Optimized)")


# ### 6.3 Model Comparison and Selection

# In[ ]:


# Combine all results
all_results = baseline_results + [xgb_result, lgb_result]

# Create results summary
results_summary = pd.DataFrame([
    {
        'model': r['model_name'],
        'train_mae': r['train_mae'],
        'train_rmse': r['train_rmse'],
        'train_r2': r['train_r2'],
        'val_mae': r['val_mae'],
        'val_rmse': r['val_rmse'],
        'val_r2': r['val_r2']
    }
    for r in all_results
]).sort_values('val_mae')

print("\n" + "="*60)
print("Model Comparison Summary")
print("="*60)
print(results_summary)

# Find best model
best_result = min(all_results, key=lambda x: x['val_mae'])

print("\n" + "="*60)
print("Best Model")
print("="*60)
print(f"Model: {best_result['model_name']}")
print(f"Validation MAE: {best_result['val_mae']:.4f}")
print(f"Validation RMSE: {best_result['val_rmse']:.4f}")
print(f"Validation R²: {best_result['val_r2']:.4f}")


# ### 6.4 Model Persistence

# In[ ]:


# Save best model and metadata
import joblib

print("Saving model and metadata...")

# Save best model
joblib.dump(best_result['model'], 'model.pkl')
print("✓ Saved model.pkl")

# Save feature columns
joblib.dump(feature_cols_all, 'feature_columns.pkl')
print("✓ Saved feature_columns.pkl")

# Save sensors to exclude
joblib.dump(sensors_to_exclude, 'sensors_to_exclude.pkl')
print("✓ Saved sensors_to_exclude.pkl")

# Save results summary
results_summary.to_csv('model_results.csv', index=False)
print("✓ Saved model_results.csv")

print("\nAll model artifacts saved successfully!")


# ### 6.5 LSTM Models (Deep Learning)

# In[ ]:


# LSTM requires sequence data - we need to reshape our data
# For time-series with LSTM, we'll use a sliding window approach

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler

print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")


# In[ ]:


# Prepare sequence data for LSTM
def create_sequences(df, sequence_length=10):
    """
    Create sequences for LSTM from time-series data.
    Each sequence contains the last N cycles for an engine.
    """
    sequences = []
    targets = []

    # Group by engine
    for engine_id in df['engine_id'].unique():
        engine_data = df[df['engine_id'] == engine_id].sort_values('cycle')

        # Get features and target
        features = engine_data[feature_cols_all].values
        target_values = engine_data['target'].values

        # Create sequences
        for i in range(len(features) - sequence_length + 1):
            sequences.append(features[i:i+sequence_length])
            targets.append(target_values[i+sequence_length-1])  # Predict for last cycle in sequence

    return np.array(sequences), np.array(targets)

# Use shorter sequences for faster training
SEQUENCE_LENGTH = 10

print(f"Creating sequences with length {SEQUENCE_LENGTH}...")
X_train_seq, y_train_seq = create_sequences(
    train_features[train_mask], 
    sequence_length=SEQUENCE_LENGTH
)
X_val_seq, y_val_seq = create_sequences(
    train_features[val_mask], 
    sequence_length=SEQUENCE_LENGTH
)

print(f"Training sequences shape: {X_train_seq.shape}")
print(f"Validation sequences shape: {X_val_seq.shape}")
print(f"Features per timestep: {X_train_seq.shape[2]}")


# In[ ]:


# Normalize features for LSTM
scaler = StandardScaler()

# Reshape for scaling
X_train_seq_reshaped = X_train_seq.reshape(-1, X_train_seq.shape[2])
X_val_seq_reshaped = X_val_seq.reshape(-1, X_val_seq.shape[2])

# Fit and transform
X_train_seq_scaled = scaler.fit_transform(X_train_seq_reshaped)
X_val_seq_scaled = scaler.transform(X_val_seq_reshaped)

# Reshape back to sequences
X_train_seq_scaled = X_train_seq_scaled.reshape(X_train_seq.shape)
X_val_seq_scaled = X_val_seq_scaled.reshape(X_val_seq.shape)

print("Features normalized for LSTM")


# In[ ]:


# Build LSTM Model 1: Simple LSTM
print("="*60)
print("Training LSTM Model 1: Simple LSTM")
print("="*60)

model_lstm_simple = Sequential([
    LSTM(50, input_shape=(SEQUENCE_LENGTH, X_train_seq.shape[2])),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)
])

model_lstm_simple.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

# Callbacks
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001
)

# Train
history_simple = model_lstm_simple.fit(
    X_train_seq_scaled, y_train_seq,
    validation_data=(X_val_seq_scaled, y_val_seq),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Evaluate
y_train_pred_lstm_simple = model_lstm_simple.predict(X_train_seq_scaled, verbose=0).flatten()
y_val_pred_lstm_simple = model_lstm_simple.predict(X_val_seq_scaled, verbose=0).flatten()

lstm_simple_result = {
    'model_name': 'LSTM (Simple)',
    'train_mae': mean_absolute_error(y_train_seq, y_train_pred_lstm_simple),
    'train_rmse': np.sqrt(mean_squared_error(y_train_seq, y_train_pred_lstm_simple)),
    'train_r2': r2_score(y_train_seq, y_train_pred_lstm_simple),
    'val_mae': mean_absolute_error(y_val_seq, y_val_pred_lstm_simple),
    'val_rmse': np.sqrt(mean_squared_error(y_val_seq, y_val_pred_lstm_simple)),
    'val_r2': r2_score(y_val_seq, y_val_pred_lstm_simple),
    'model': model_lstm_simple
}

print(f"\nSimple LSTM Performance:")
print(f"  Train - MAE: {lstm_simple_result['train_mae']:.4f}, RMSE: {lstm_simple_result['train_rmse']:.4f}, R²: {lstm_simple_result['train_r2']:.4f}")
print(f"  Val   - MAE: {lstm_simple_result['val_mae']:.4f}, RMSE: {lstm_simple_result['val_rmse']:.4f}, R²: {lstm_simple_result['val_r2']:.4f}")


# In[ ]:


# Build LSTM Model 2: Stacked LSTM with Dropout
print("\n" + "="*60)
print("Training LSTM Model 2: Stacked LSTM")
print("="*60)

model_lstm_stacked = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, X_train_seq.shape[2])),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

model_lstm_stacked.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

# Train
history_stacked = model_lstm_stacked.fit(
    X_train_seq_scaled, y_train_seq,
    validation_data=(X_val_seq_scaled, y_val_seq),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Evaluate
y_train_pred_lstm_stacked = model_lstm_stacked.predict(X_train_seq_scaled, verbose=0).flatten()
y_val_pred_lstm_stacked = model_lstm_stacked.predict(X_val_seq_scaled, verbose=0).flatten()

lstm_stacked_result = {
    'model_name': 'LSTM (Stacked)',
    'train_mae': mean_absolute_error(y_train_seq, y_train_pred_lstm_stacked),
    'train_rmse': np.sqrt(mean_squared_error(y_train_seq, y_train_pred_lstm_stacked)),
    'train_r2': r2_score(y_train_seq, y_train_pred_lstm_stacked),
    'val_mae': mean_absolute_error(y_val_seq, y_val_pred_lstm_stacked),
    'val_rmse': np.sqrt(mean_squared_error(y_val_seq, y_val_pred_lstm_stacked)),
    'val_r2': r2_score(y_val_seq, y_val_pred_lstm_stacked),
    'model': model_lstm_stacked
}

print(f"\nStacked LSTM Performance:")
print(f"  Train - MAE: {lstm_stacked_result['train_mae']:.4f}, RMSE: {lstm_stacked_result['train_rmse']:.4f}, R²: {lstm_stacked_result['train_r2']:.4f}")
print(f"  Val   - MAE: {lstm_stacked_result['val_mae']:.4f}, RMSE: {lstm_stacked_result['val_rmse']:.4f}, R²: {lstm_stacked_result['val_r2']:.4f}")


# In[ ]:


# Plot training history
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Simple LSTM
axes[0, 0].plot(history_simple.history['loss'], label='Train Loss')
axes[0, 0].plot(history_simple.history['val_loss'], label='Val Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss (MSE)')
axes[0, 0].set_title('Simple LSTM: Training History (Loss)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(history_simple.history['mae'], label='Train MAE')
axes[0, 1].plot(history_simple.history['val_mae'], label='Val MAE')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('MAE')
axes[0, 1].set_title('Simple LSTM: Training History (MAE)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Stacked LSTM
axes[1, 0].plot(history_stacked.history['loss'], label='Train Loss')
axes[1, 0].plot(history_stacked.history['val_loss'], label='Val Loss')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Loss (MSE)')
axes[1, 0].set_title('Stacked LSTM: Training History (Loss)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(history_stacked.history['mae'], label='Train MAE')
axes[1, 1].plot(history_stacked.history['val_mae'], label='Val MAE')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('MAE')
axes[1, 1].set_title('Stacked LSTM: Training History (MAE)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# ### 6.6 Final Model Comparison

# In[ ]:


# Combine all results including LSTM models
all_results_final = baseline_results + [xgb_result, lgb_result, lstm_simple_result, lstm_stacked_result]

# Create comprehensive results summary
results_final = pd.DataFrame([
    {
        'Model': r['model_name'],
        'Train MAE': r['train_mae'],
        'Train RMSE': r['train_rmse'],
        'Train R²': r['train_r2'],
        'Val MAE': r['val_mae'],
        'Val RMSE': r['val_rmse'],
        'Val R²': r['val_r2'],
        'Overfit (MAE diff)': r['train_mae'] - r['val_mae']
    }
    for r in all_results_final
]).sort_values('Val MAE')

print("\n" + "="*80)
print("COMPREHENSIVE MODEL COMPARISON")
print("="*80)
print(results_final.to_string(index=False))

# Find overall best model
best_model_final = min(all_results_final, key=lambda x: x['val_mae'])

print("\n" + "="*80)
print("BEST MODEL")
print("="*80)
print(f"Model: {best_model_final['model_name']}")
print(f"Validation MAE: {best_model_final['val_mae']:.4f}")
print(f"Validation RMSE: {best_model_final['val_rmse']:.4f}")
print(f"Validation R²: {best_model_final['val_r2']:.4f}")


# In[ ]:


# Visualize model comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Validation MAE Comparison
models = results_final['Model'].values
val_mae = results_final['Val MAE'].values
colors = ['red' if mae == val_mae.min() else 'steelblue' for mae in val_mae]

axes[0].barh(models, val_mae, color=colors)
axes[0].set_xlabel('Validation MAE (lower is better)')
axes[0].set_title('Model Comparison: Validation MAE')
axes[0].invert_yaxis()
axes[0].grid(True, alpha=0.3, axis='x')

# 2. Validation RMSE Comparison
val_rmse = results_final['Val RMSE'].values
colors_rmse = ['red' if rmse == val_rmse.min() else 'steelblue' for rmse in val_rmse]

axes[1].barh(models, val_rmse, color=colors_rmse)
axes[1].set_xlabel('Validation RMSE (lower is better)')
axes[1].set_title('Model Comparison: Validation RMSE')
axes[1].invert_yaxis()
axes[1].grid(True, alpha=0.3, axis='x')

# 3. Validation R² Comparison
val_r2 = results_final['Val R²'].values
colors_r2 = ['green' if r2 == val_r2.max() else 'steelblue' for r2 in val_r2]

axes[2].barh(models, val_r2, color=colors_r2)
axes[2].set_xlabel('Validation R² (higher is better)')
axes[2].set_title('Model Comparison: Validation R²')
axes[2].invert_yaxis()
axes[2].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()


# In[ ]:


# Save final best model
import joblib

print("Saving best model and metadata...")

# Save best model (handling different model types)
if 'LSTM' in best_model_final['model_name']:
    # Save Keras model
    best_model_final['model'].save('model_lstm.keras')
    print("✓ Saved model_lstm.keras")
    # Also save as pickle for compatibility
    joblib.dump(best_result['model'], 'model.pkl')
    print("✓ Saved model.pkl (non-LSTM best)")
else:
    joblib.dump(best_model_final['model'], 'model.pkl')
    print("✓ Saved model.pkl")

# Save feature columns
joblib.dump(feature_cols_all, 'feature_columns.pkl')
print("✓ Saved feature_columns.pkl")

# Save sensors to exclude
joblib.dump(sensors_to_exclude, 'sensors_to_exclude.pkl')
print("✓ Saved sensors_to_exclude.pkl")

# Save comprehensive results
results_final.to_csv('model_results_comprehensive.csv', index=False)
print("✓ Saved model_results_comprehensive.csv")

# Save scaler for LSTM
if 'LSTM' in best_model_final['model_name']:
    joblib.dump(scaler, 'scaler.pkl')
    print("✓ Saved scaler.pkl (for LSTM)")

print("\nAll model artifacts saved successfully!")

