# Data preprocessing and feature engineering

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath):
    """Load dataset from CSV file"""
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    Returns distance in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Radius of earth in kilometers
    r = 6371
    return c * r


def clean_data(df):
    """Clean and preprocess the dataset"""
    print("\nCleaning data...")
    
    # Create a copy
    df = df.copy()
    
    # Check for missing values
    print(f"Missing values:\n{df.isnull().sum()}")
    
    # Remove rows with invalid coordinates (0,0)
    initial_rows = len(df)
    df = df[~((df['Restaurant_latitude'] == 0) & (df['Restaurant_longitude'] == 0))]
    df = df[~((df['Delivery_location_latitude'] == 0) & (df['Delivery_location_longitude'] == 0))]
    print(f"Removed {initial_rows - len(df)} rows with invalid coordinates")
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle outliers in delivery time (keep reasonable range: 5-120 minutes)
    df = df[(df['Delivery Time_taken(min)'] >= 5) & (df['Delivery Time_taken(min)'] <= 120)]
    
    # Handle outliers in ratings (should be between 1-5)
    df = df[(df['Delivery_person_Ratings'] >= 1) & (df['Delivery_person_Ratings'] <= 5)]
    
    # Handle outliers in age (reasonable age range: 18-65)
    df = df[(df['Delivery_person_Age'] >= 18) & (df['Delivery_person_Age'] <= 65)]
    
    # Clean text columns (remove extra spaces)
    df['Type_of_order'] = df['Type_of_order'].str.strip()
    df['Type_of_vehicle'] = df['Type_of_vehicle'].str.strip()
    
    print(f"Cleaned: {df.shape[0]} rows remaining")
    
    return df


def engineer_features(df):
    """Create new features from existing data"""
    print("\nEngineering features...")
    
    df = df.copy()
    
    # 1. Calculate delivery distance using Haversine formula
    df['delivery_distance_km'] = haversine_distance(
        df['Restaurant_latitude'],
        df['Restaurant_longitude'],
        df['Delivery_location_latitude'],
        df['Delivery_location_longitude']
    )
    
    # 2. Calculate latitude and longitude differences
    df['lat_diff'] = abs(df['Delivery_location_latitude'] - df['Restaurant_latitude'])
    df['lon_diff'] = abs(df['Delivery_location_longitude'] - df['Restaurant_longitude'])
    
    # 3. Create area/zone features (approximate city center distance)
    # Assuming center as mean of all coordinates
    df['restaurant_distance_from_center'] = np.sqrt(
        (df['Restaurant_latitude'] - df['Restaurant_latitude'].mean())**2 + 
        (df['Restaurant_longitude'] - df['Restaurant_longitude'].mean())**2
    )
    
    df['delivery_distance_from_center'] = np.sqrt(
        (df['Delivery_location_latitude'] - df['Delivery_location_latitude'].mean())**2 + 
        (df['Delivery_location_longitude'] - df['Delivery_location_longitude'].mean())**2
    )
    
    # 4. Create rating bins (performance categories)
    df['rating_category'] = pd.cut(
        df['Delivery_person_Ratings'], 
        bins=[0, 3.5, 4.2, 5.0], 
        labels=['Low', 'Medium', 'High']
    )
    
    # 5. Create age groups
    df['age_group'] = pd.cut(
        df['Delivery_person_Age'], 
        bins=[0, 25, 35, 65], 
        labels=['Young', 'Mid', 'Senior']
    )
    
    # 6. Interaction features
    df['distance_rating_interaction'] = df['delivery_distance_km'] * df['Delivery_person_Ratings']
    df['distance_age_interaction'] = df['delivery_distance_km'] * df['Delivery_person_Age']
    
    print(f"Total features: {len(df.columns)}")
    
    return df


def encode_categorical(df, fit_encoders=None):
    """Encode categorical variables"""
    print("\nEncoding categorical variables...")
    
    df = df.copy()
    
    # Label encoding for ordinal/binary categories
    categorical_cols = ['Type_of_order', 'Type_of_vehicle', 'rating_category', 'age_group']
    
    if fit_encoders is None:
        encoders = {}
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
    else:
        encoders = fit_encoders
        for col in categorical_cols:
            if col in df.columns and col in encoders:
                df[col + '_encoded'] = encoders[col].transform(df[col].astype(str))
    
    # One-hot encoding for nominal categories (alternative approach)
    # df = pd.get_dummies(df, columns=['Type_of_order', 'Type_of_vehicle'], drop_first=True)
    
    print("Encoding complete")
    
    return df, encoders


def prepare_for_modeling(df):
    """Prepare final dataset for model training"""
    print("\nPreparing features...")
    
    # Define feature columns
    feature_cols = [
        'Delivery_person_Age',
        'Delivery_person_Ratings',
        'Restaurant_latitude',
        'Restaurant_longitude',
        'Delivery_location_latitude',
        'Delivery_location_longitude',
        'delivery_distance_km',
        'lat_diff',
        'lon_diff',
        'restaurant_distance_from_center',
        'delivery_distance_from_center',
        'distance_rating_interaction',
        'distance_age_interaction',
        'Type_of_order_encoded',
        'Type_of_vehicle_encoded',
        'rating_category_encoded',
        'age_group_encoded'
    ]
    
    # Target column
    target_col = 'Delivery Time_taken(min)'
    
    # Select features and target
    X = df[feature_cols]
    y = df[target_col]
    
    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    
    return X, y, feature_cols


def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and testing sets"""
    print(f"\nSplitting: {(1-test_size)*100:.0f}% train, {test_size*100:.0f}% test")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test, fit_scaler=None):
    """Scale features using StandardScaler"""
    print("\nScaling features...")
    
    if fit_scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        scaler = fit_scaler
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    
    print("Scaling complete")
    
    return X_train_scaled, X_test_scaled, scaler


def preprocess_pipeline(filepath, for_training=True):
    """Complete preprocessing pipeline"""
    print("="*60)
    print("Starting preprocessing...")
    print("="*60)
    
    # Load data
    df = load_data(filepath)
    
    # Clean data
    df = clean_data(df)
    
    # Engineer features
    df = engineer_features(df)
    
    # Encode categorical
    df, encoders = encode_categorical(df)
    
    # Prepare for modeling
    X, y, feature_cols = prepare_for_modeling(df)
    
    if for_training:
        # Split data
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # Scale features
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
        
        print("\n" + "="*60)
        print("Preprocessing complete!")
        print("="*60)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler, encoders, feature_cols, df
    else:
        print("\n" + "="*60)
        print("Preprocessing complete!")
        print("="*60)
        
        return X, y, encoders, feature_cols, df


if __name__ == "__main__":
    # Test the preprocessing pipeline
    import os
    filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Dataset.csv')
    X_train, X_test, y_train, y_test, scaler, encoders, feature_cols, df = preprocess_pipeline(filepath)
    
    print(f"\n📊 Final shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")
