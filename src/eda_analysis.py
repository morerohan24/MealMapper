# Exploratory Data Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def load_and_basic_info(filepath):
    """Load dataset and display basic information"""
    print("="*70)
    print("📊 EXPLORATORY DATA ANALYSIS")
    print("="*70)
    
    df = pd.read_csv(filepath)
    
    print(f"\n📌 Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\n📋 Column Names:")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i}. {col}")
    
    print(f"\n🔍 Data Types:")
    print(df.dtypes)
    
    print(f"\n📊 Basic Statistics:")
    print(df.describe())
    
    print(f"\n❓ Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("   ✅ No missing values found!")
    else:
        print(missing[missing > 0])
    
    print(f"\n🔄 Duplicate Rows: {df.duplicated().sum()}")
    
    return df


def analyze_target_variable(df, save_path=None):
    if save_path is None:
        import os
        save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'visualizations', 'target_distribution.png')
    """Analyze the target variable distribution"""
    print("\n" + "="*70)
    print("🎯 TARGET VARIABLE ANALYSIS: Delivery Time")
    print("="*70)
    
    target = df['Delivery Time_taken(min)']
    
    print(f"\n📊 Statistics:")
    print(f"   Mean:     {target.mean():.2f} minutes")
    print(f"   Median:   {target.median():.2f} minutes")
    print(f"   Std Dev:  {target.std():.2f} minutes")
    print(f"   Min:      {target.min():.2f} minutes")
    print(f"   Max:      {target.max():.2f} minutes")
    print(f"   Q1 (25%): {target.quantile(0.25):.2f} minutes")
    print(f"   Q3 (75%): {target.quantile(0.75):.2f} minutes")
    
    # Create visualizations
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Target Variable: Delivery Time Analysis', fontsize=16, fontweight='bold')
    
    # Histogram
    axes[0].hist(target, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0].axvline(target.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {target.mean():.2f}')
    axes[0].axvline(target.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {target.median():.2f}')
    axes[0].set_xlabel('Delivery Time (minutes)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    axes[1].boxplot(target, vert=True)
    axes[1].set_ylabel('Delivery Time (minutes)', fontsize=12)
    axes[1].set_title('Box Plot (Outlier Detection)', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    # Q-Q plot (check normality)
    from scipy import stats
    stats.probplot(target, dist="norm", plot=axes[2])
    axes[2].set_title('Q-Q Plot (Normality Check)', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Target distribution plot saved to {save_path}")
    plt.close()


def analyze_categorical_features(df, save_path=None):
    if save_path is None:
        import os
        save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'visualizations', 'categorical_analysis.png')
    """Analyze categorical features"""
    print("\n" + "="*70)
    print("📊 CATEGORICAL FEATURES ANALYSIS")
    print("="*70)
    
    categorical_cols = ['Type_of_order', 'Type_of_vehicle']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Categorical Features Analysis', fontsize=16, fontweight='bold')
    
    # Type of Order
    order_counts = df['Type_of_order'].value_counts()
    print(f"\n🍽️ Type of Order Distribution:")
    print(order_counts)
    
    axes[0, 0].bar(order_counts.index, order_counts.values, color='coral')
    axes[0, 0].set_xlabel('Order Type', fontsize=12)
    axes[0, 0].set_ylabel('Count', fontsize=12)
    axes[0, 0].set_title('Order Type Distribution', fontsize=14)
    axes[0, 0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(order_counts.values):
        axes[0, 0].text(i, v + 100, str(v), ha='center', va='bottom')
    
    # Type of Vehicle
    vehicle_counts = df['Type_of_vehicle'].value_counts()
    print(f"\n🏍️ Type of Vehicle Distribution:")
    print(vehicle_counts)
    
    axes[0, 1].bar(vehicle_counts.index, vehicle_counts.values, color='lightgreen')
    axes[0, 1].set_xlabel('Vehicle Type', fontsize=12)
    axes[0, 1].set_ylabel('Count', fontsize=12)
    axes[0, 1].set_title('Vehicle Type Distribution', fontsize=14)
    axes[0, 1].tick_params(axis='x', rotation=45)
    for i, v in enumerate(vehicle_counts.values):
        axes[0, 1].text(i, v + 100, str(v), ha='center', va='bottom')
    
    # Delivery time by order type
    order_time = df.groupby('Type_of_order')['Delivery Time_taken(min)'].mean().sort_values()
    axes[1, 0].barh(order_time.index, order_time.values, color='skyblue')
    axes[1, 0].set_xlabel('Average Delivery Time (minutes)', fontsize=12)
    axes[1, 0].set_ylabel('Order Type', fontsize=12)
    axes[1, 0].set_title('Avg Delivery Time by Order Type', fontsize=14)
    for i, v in enumerate(order_time.values):
        axes[1, 0].text(v + 0.5, i, f'{v:.1f}', va='center')
    
    # Delivery time by vehicle type
    vehicle_time = df.groupby('Type_of_vehicle')['Delivery Time_taken(min)'].mean().sort_values()
    axes[1, 1].barh(vehicle_time.index, vehicle_time.values, color='lightyellow')
    axes[1, 1].set_xlabel('Average Delivery Time (minutes)', fontsize=12)
    axes[1, 1].set_ylabel('Vehicle Type', fontsize=12)
    axes[1, 1].set_title('Avg Delivery Time by Vehicle Type', fontsize=14)
    for i, v in enumerate(vehicle_time.values):
        axes[1, 1].text(v + 0.5, i, f'{v:.1f}', va='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Categorical analysis plot saved to {save_path}")
    plt.close()


def analyze_numerical_features(df, save_path=None):
    if save_path is None:
        import os
        save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'visualizations', 'numerical_analysis.png')
    """Analyze numerical features"""
    print("\n" + "="*70)
    print("📊 NUMERICAL FEATURES ANALYSIS")
    print("="*70)
    
    numerical_cols = ['Delivery_person_Age', 'Delivery_person_Ratings']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Numerical Features Analysis', fontsize=16, fontweight='bold')
    
    # Age distribution
    print(f"\n👤 Delivery Person Age:")
    print(df['Delivery_person_Age'].describe())
    
    axes[0, 0].hist(df['Delivery_person_Age'], bins=30, edgecolor='black', alpha=0.7, color='purple')
    axes[0, 0].set_xlabel('Age (years)', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('Delivery Person Age Distribution', fontsize=14)
    axes[0, 0].axvline(df['Delivery_person_Age'].mean(), color='red', linestyle='--', linewidth=2)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Rating distribution
    print(f"\n⭐ Delivery Person Ratings:")
    print(df['Delivery_person_Ratings'].describe())
    
    axes[0, 1].hist(df['Delivery_person_Ratings'], bins=30, edgecolor='black', alpha=0.7, color='gold')
    axes[0, 1].set_xlabel('Rating (out of 5)', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title('Delivery Person Rating Distribution', fontsize=14)
    axes[0, 1].axvline(df['Delivery_person_Ratings'].mean(), color='red', linestyle='--', linewidth=2)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Age vs Delivery Time
    axes[1, 0].scatter(df['Delivery_person_Age'], df['Delivery Time_taken(min)'], alpha=0.3, s=10)
    axes[1, 0].set_xlabel('Delivery Person Age (years)', fontsize=12)
    axes[1, 0].set_ylabel('Delivery Time (minutes)', fontsize=12)
    axes[1, 0].set_title('Age vs Delivery Time', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Rating vs Delivery Time
    axes[1, 1].scatter(df['Delivery_person_Ratings'], df['Delivery Time_taken(min)'], alpha=0.3, s=10, color='green')
    axes[1, 1].set_xlabel('Delivery Person Rating', fontsize=12)
    axes[1, 1].set_ylabel('Delivery Time (minutes)', fontsize=12)
    axes[1, 1].set_title('Rating vs Delivery Time', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Numerical analysis plot saved to {save_path}")
    plt.close()


def analyze_correlations(df, save_path=None):
    if save_path is None:
        import os
        save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'visualizations', 'correlation_matrix.png')
    """Analyze correlations between numerical features"""
    print("\n" + "="*70)
    print("🔗 CORRELATION ANALYSIS")
    print("="*70)
    
    # Select numerical columns
    numerical_cols = [
        'Delivery_person_Age',
        'Delivery_person_Ratings',
        'Restaurant_latitude',
        'Restaurant_longitude',
        'Delivery_location_latitude',
        'Delivery_location_longitude',
        'Delivery Time_taken(min)'
    ]
    
    corr_matrix = df[numerical_cols].corr()
    
    print("\n📊 Correlation with Delivery Time:")
    delivery_time_corr = corr_matrix['Delivery Time_taken(min)'].sort_values(ascending=False)
    print(delivery_time_corr)
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Correlation matrix saved to {save_path}")
    plt.close()


def analyze_location_data(df, save_path=None):
    if save_path is None:
        import os
        save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'visualizations', 'location_analysis.png')
    """Analyze location-based patterns"""
    print("\n" + "="*70)
    print("📍 LOCATION-BASED ANALYSIS")
    print("="*70)
    
    # Calculate distances
    from math import radians, cos, sin, asin, sqrt
    
    def haversine_vectorized(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        km = 6371 * c
        return km
    
    df['distance_km'] = haversine_vectorized(
        df['Restaurant_latitude'].values,
        df['Restaurant_longitude'].values,
        df['Delivery_location_latitude'].values,
        df['Delivery_location_longitude'].values
    )
    
    print(f"\n📏 Delivery Distance Statistics:")
    print(df['distance_km'].describe())
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Location-Based Analysis', fontsize=16, fontweight='bold')
    
    # Distance distribution
    axes[0, 0].hist(df['distance_km'], bins=50, edgecolor='black', alpha=0.7, color='teal')
    axes[0, 0].set_xlabel('Distance (km)', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('Delivery Distance Distribution', fontsize=14)
    axes[0, 0].axvline(df['distance_km'].mean(), color='red', linestyle='--', linewidth=2)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Distance vs Delivery Time
    axes[0, 1].scatter(df['distance_km'], df['Delivery Time_taken(min)'], alpha=0.3, s=10, color='orange')
    axes[0, 1].set_xlabel('Distance (km)', fontsize=12)
    axes[0, 1].set_ylabel('Delivery Time (minutes)', fontsize=12)
    axes[0, 1].set_title('Distance vs Delivery Time', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Restaurant locations
    axes[1, 0].scatter(df['Restaurant_longitude'], df['Restaurant_latitude'], 
                      alpha=0.1, s=5, c='blue')
    axes[1, 0].set_xlabel('Longitude', fontsize=12)
    axes[1, 0].set_ylabel('Latitude', fontsize=12)
    axes[1, 0].set_title('Restaurant Locations', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Delivery locations
    axes[1, 1].scatter(df['Delivery_location_longitude'], df['Delivery_location_latitude'], 
                      alpha=0.1, s=5, c='red')
    axes[1, 1].set_xlabel('Longitude', fontsize=12)
    axes[1, 1].set_ylabel('Latitude', fontsize=12)
    axes[1, 1].set_title('Delivery Locations', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Location analysis plot saved to {save_path}")
    plt.close()
    
    return df


def generate_insights(df):
    """Generate key business insights"""
    print("\n" + "="*70)
    print("💡 KEY BUSINESS INSIGHTS")
    print("="*70)
    
    # Calculate distance if not already done
    if 'distance_km' not in df.columns:
        from math import radians, cos, sin, asin, sqrt
        
        def haversine_vectorized(lat1, lon1, lat2, lon2):
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            km = 6371 * c
            return km
        
        df['distance_km'] = haversine_vectorized(
            df['Restaurant_latitude'].values,
            df['Restaurant_longitude'].values,
            df['Delivery_location_latitude'].values,
            df['Delivery_location_longitude'].values
        )
    
    print("\n1️⃣ Average Delivery Time by Order Type:")
    print(df.groupby('Type_of_order')['Delivery Time_taken(min)'].mean().sort_values(ascending=False))
    
    print("\n2️⃣ Average Delivery Time by Vehicle Type:")
    print(df.groupby('Type_of_vehicle')['Delivery Time_taken(min)'].mean().sort_values(ascending=False))
    
    print("\n3️⃣ Impact of Partner Rating on Delivery Time:")
    rating_bins = pd.cut(df['Delivery_person_Ratings'], bins=[0, 3.5, 4.2, 5.0], labels=['Low', 'Medium', 'High'])
    print(df.groupby(rating_bins)['Delivery Time_taken(min)'].mean())
    
    print("\n4️⃣ Impact of Partner Age on Delivery Time:")
    age_bins = pd.cut(df['Delivery_person_Age'], bins=[0, 25, 35, 65], labels=['Young', 'Mid', 'Senior'])
    print(df.groupby(age_bins)['Delivery Time_taken(min)'].mean())
    
    print("\n5️⃣ Distance Impact:")
    distance_bins = pd.cut(df['distance_km'], bins=[0, 5, 10, 20, 100], labels=['<5km', '5-10km', '10-20km', '>20km'])
    print(df.groupby(distance_bins)['Delivery Time_taken(min)'].mean())
    
    print("\n6️⃣ Fastest Vehicle Type:")
    fastest_vehicle = df.groupby('Type_of_vehicle')['Delivery Time_taken(min)'].mean().idxmin()
    print(f"   ⚡ Fastest: {fastest_vehicle}")
    
    print("\n7️⃣ Most Ordered Food Type:")
    most_ordered = df['Type_of_order'].value_counts().idxmax()
    print(f"   🍽️ Most Popular: {most_ordered}")


def main():
    """Main EDA pipeline"""
    print("\n" + "🔍"*35)
    print(" "*20 + "EXPLORATORY DATA ANALYSIS")
    print(" "*25 + "Food Delivery Dataset")
    print("🔍"*35 + "\n")
    
    # Load and basic info
    import os
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Dataset.csv')
    df = load_and_basic_info(dataset_path)
    
    # Analyze target variable
    analyze_target_variable(df)
    
    # Analyze categorical features
    analyze_categorical_features(df)
    
    # Analyze numerical features
    analyze_numerical_features(df)
    
    # Analyze correlations
    analyze_correlations(df)
    
    # Analyze location data
    df = analyze_location_data(df)
    
    # Generate insights
    generate_insights(df)
    
    print("\n" + "="*70)
    print("✅ EXPLORATORY DATA ANALYSIS COMPLETED!")
    print("="*70)
    print("\n📁 All visualizations saved in 'visualizations/' folder")
    print("💡 Key insights generated for presentation")


if __name__ == "__main__":
    main()
