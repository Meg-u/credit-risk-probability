import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Load your dataset
df = pd.read_csv('data/raw/data.csv')

# Define features to engineer
# Aggregate Features
def create_aggregate_features(df):

    aggregates = df.groupby('CustomerId')['Amount'].agg(['sum', 'mean', 'count','std']).add_prefix('Transaction_')
    df = df.merge(aggregates.reset_index(), on='CustomerId', how='left')
    return df

# Extract Features from TransactionStartTime
df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
df['TransactionHour'] = df['TransactionStartTime'].dt.hour
df['TransactionDay'] = df['TransactionStartTime'].dt.day
df['TransactionMonth'] = df['TransactionStartTime'].dt.month
df['TransactionYear'] = df['TransactionStartTime'].dt.year

# Define preprocessing for numeric and categorical data
numeric_features = ['Amount', 'Value', 'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear']
categorical_features = ['CurrencyCode', 'CountryCode', 'ProductCategory', 'ChannelId', 'PricingStrategy']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Apply transformations
X = df.drop('FraudResult', axis=1)  
y = df['FraudResult']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply the preprocessor
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# At this point, X_train_processed and X_test_processed are ready for model training

train_df = pd.DataFrame(X_train_processed, columns=preprocessor.get_feature_names_out())
train_df['FraudResult'] = y_train

# Save to CSV
train_df.to_csv('data/processed/processed_train_data.csv', index=False)