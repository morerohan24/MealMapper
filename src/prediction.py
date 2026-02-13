# Prediction utilities

import pandas as pd
import numpy as np
import joblib
from preprocessing import haversine_distance


def load_model_artifacts(model_dir=None):
    if model_dir is None:
        import os
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models') + os.sep
    """Load trained model and preprocessing artifacts"""
    try:
        model = joblib.load(model_dir + 'best_model.pkl')
        scaler = joblib.load(model_dir + 'scaler.pkl')
        encoders = joblib.load(model_dir + 'encoders.pkl')
        feature_cols = joblib.load(model_dir + 'feature_cols.pkl')
        
        print("✅ Model artifacts loaded successfully!")
        return model, scaler, encoders, feature_cols
    
    except Exception as e:
        print(f"❌ Error loading model artifacts: {e}")
        print("Make sure you've run the training pipeline first!")
        return None, None, None, None


def prepare_single_input(
    delivery_person_age,
    delivery_person_rating,
    restaurant_lat,
    restaurant_lon,
    delivery_lat,
    delivery_lon,
    order_type,
    vehicle_type,
    encoders
):
    """
    Prepare a single input for prediction
    
    Parameters:
    -----------
    delivery_person_age : int
        Age of delivery partner (18-65)
    delivery_person_rating : float
        Rating of delivery partner (1.0-5.0)
    restaurant_lat : float
        Restaurant latitude
    restaurant_lon : float
        Restaurant longitude
    delivery_lat : float
        Delivery location latitude
    delivery_lon : float
        Delivery location longitude
    order_type : str
        Type of order ('Snack', 'Meal', 'Buffet', 'Drinks')
    vehicle_type : str
        Type of vehicle ('motorcycle', 'scooter', 'electric_scooter')
    encoders : dict
        Dictionary of label encoders
    
    Returns:
    --------
    features : numpy array
        Prepared features for prediction
    """
    
    # Calculate delivery distance
    distance_km = haversine_distance(
        restaurant_lat, restaurant_lon,
        delivery_lat, delivery_lon
    )
    
    # Calculate coordinate differences
    lat_diff = abs(delivery_lat - restaurant_lat)
    lon_diff = abs(delivery_lon - restaurant_lon)
    
    # Distance from center (using approximate mean values)
    mean_lat = 16.0
    mean_lon = 77.0
    
    restaurant_distance_from_center = np.sqrt(
        (restaurant_lat - mean_lat)**2 + (restaurant_lon - mean_lon)**2
    )
    
    delivery_distance_from_center = np.sqrt(
        (delivery_lat - mean_lat)**2 + (delivery_lon - mean_lon)**2
    )
    
    # Create rating category
    if delivery_person_rating <= 3.5:
        rating_category = 'Low'
    elif delivery_person_rating <= 4.2:
        rating_category = 'Medium'
    else:
        rating_category = 'High'
    
    # Create age group
    if delivery_person_age <= 25:
        age_group = 'Young'
    elif delivery_person_age <= 35:
        age_group = 'Mid'
    else:
        age_group = 'Senior'
    
    # Interaction features
    distance_rating_interaction = distance_km * delivery_person_rating
    distance_age_interaction = distance_km * delivery_person_age
    
    # Encode categorical variables
    order_type_encoded = encoders['Type_of_order'].transform([order_type])[0]
    vehicle_type_encoded = encoders['Type_of_vehicle'].transform([vehicle_type])[0]
    rating_category_encoded = encoders['rating_category'].transform([rating_category])[0]
    age_group_encoded = encoders['age_group'].transform([age_group])[0]
    
    # Create feature array in correct order
    features = [
        delivery_person_age,
        delivery_person_rating,
        restaurant_lat,
        restaurant_lon,
        delivery_lat,
        delivery_lon,
        distance_km,
        lat_diff,
        lon_diff,
        restaurant_distance_from_center,
        delivery_distance_from_center,
        distance_rating_interaction,
        distance_age_interaction,
        order_type_encoded,
        vehicle_type_encoded,
        rating_category_encoded,
        age_group_encoded
    ]
    
    return np.array(features).reshape(1, -1), distance_km


def predict_delivery_time(features, model, scaler):
    """Make prediction on prepared features"""
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    
    return prediction


def predict_single_order(
    delivery_person_age,
    delivery_person_rating,
    restaurant_lat,
    restaurant_lon,
    delivery_lat,
    delivery_lon,
    order_type,
    vehicle_type
):
    """
    Complete prediction pipeline for a single order
    
    Example usage:
    >>> predicted_time = predict_single_order(
    ...     delivery_person_age=30,
    ...     delivery_person_rating=4.5,
    ...     restaurant_lat=12.9716,
    ...     restaurant_lon=77.5946,
    ...     delivery_lat=13.0358,
    ...     delivery_lon=77.6431,
    ...     order_type='Meal',
    ...     vehicle_type='motorcycle'
    ... )
    """
    
    # Load model artifacts
    model, scaler, encoders, feature_cols = load_model_artifacts()
    
    if model is None:
        return None
    
    # Prepare input
    features, distance = prepare_single_input(
        delivery_person_age,
        delivery_person_rating,
        restaurant_lat,
        restaurant_lon,
        delivery_lat,
        delivery_lon,
        order_type,
        vehicle_type,
        encoders
    )
    
    # Make prediction
    predicted_time = predict_delivery_time(features, model, scaler)
    
    return predicted_time, distance


def batch_predict(input_df):
    """
    Make predictions on a batch of orders
    
    Parameters:
    -----------
    input_df : pandas DataFrame
        DataFrame with columns matching training data
    
    Returns:
    --------
    predictions : numpy array
        Array of predicted delivery times
    """
    
    # Load model artifacts
    model, scaler, encoders, feature_cols = load_model_artifacts()
    
    if model is None:
        return None
    
    # Prepare all inputs
    predictions = []
    
    for idx, row in input_df.iterrows():
        features, _ = prepare_single_input(
            row['Delivery_person_Age'],
            row['Delivery_person_Ratings'],
            row['Restaurant_latitude'],
            row['Restaurant_longitude'],
            row['Delivery_location_latitude'],
            row['Delivery_location_longitude'],
            row['Type_of_order'],
            row['Type_of_vehicle'],
            encoders
        )
        
        pred = predict_delivery_time(features, model, scaler)
        predictions.append(pred)
    
    return np.array(predictions)


def demo_prediction():
    """Demo prediction with sample data"""
    print("\n" + "="*70)
    print("🔮 DELIVERY TIME PREDICTION DEMO")
    print("="*70)
    
    # Sample scenarios
    scenarios = [
        {
            'name': 'Quick Snack Delivery',
            'delivery_person_age': 28,
            'delivery_person_rating': 4.8,
            'restaurant_lat': 12.9716,
            'restaurant_lon': 77.5946,
            'delivery_lat': 12.9916,
            'delivery_lon': 77.6146,
            'order_type': 'Snack',
            'vehicle_type': 'electric_scooter'
        },
        {
            'name': 'Standard Meal Delivery',
            'delivery_person_age': 32,
            'delivery_person_rating': 4.5,
            'restaurant_lat': 19.0760,
            'restaurant_lon': 72.8777,
            'delivery_lat': 19.1136,
            'delivery_lon': 72.9083,
            'order_type': 'Meal',
            'vehicle_type': 'motorcycle'
        },
        {
            'name': 'Long Distance Buffet',
            'delivery_person_age': 35,
            'delivery_person_rating': 4.2,
            'restaurant_lat': 28.7041,
            'restaurant_lon': 77.1025,
            'delivery_lat': 28.5355,
            'delivery_lon': 77.3910,
            'order_type': 'Buffet',
            'vehicle_type': 'scooter'
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'='*70}")
        print(f"Scenario {i}: {scenario['name']}")
        print(f"{'='*70}")
        
        print(f"\n📋 Order Details:")
        print(f"   Partner Age: {scenario['delivery_person_age']}")
        print(f"   Partner Rating: {scenario['delivery_person_rating']}/5.0")
        print(f"   Order Type: {scenario['order_type']}")
        print(f"   Vehicle: {scenario['vehicle_type'].replace('_', ' ').title()}")
        
        result = predict_single_order(
            delivery_person_age=scenario['delivery_person_age'],
            delivery_person_rating=scenario['delivery_person_rating'],
            restaurant_lat=scenario['restaurant_lat'],
            restaurant_lon=scenario['restaurant_lon'],
            delivery_lat=scenario['delivery_lat'],
            delivery_lon=scenario['delivery_lon'],
            order_type=scenario['order_type'],
            vehicle_type=scenario['vehicle_type']
        )
        
        if result:
            predicted_time, distance = result
            avg_speed = (distance / predicted_time) * 60 if predicted_time > 0 else 0
            
            print(f"\n🎯 Prediction Results:")
            print(f"   Estimated Delivery Time: {predicted_time:.1f} minutes")
            print(f"   Delivery Distance: {distance:.2f} km")
            print(f"   Average Speed: {avg_speed:.1f} km/h")
    
    print("\n" + "="*70)
    print("✅ Demo completed!")
    print("="*70)


if __name__ == "__main__":
    # Run demo
    demo_prediction()
    
    # Interactive prediction
    print("\n" + "="*70)
    print("🎮 INTERACTIVE PREDICTION")
    print("="*70)
    
    print("\nEnter order details:")
    
    try:
        age = int(input("Delivery partner age (18-65): "))
        rating = float(input("Delivery partner rating (1.0-5.0): "))
        
        print("\nRestaurant location:")
        rest_lat = float(input("  Latitude: "))
        rest_lon = float(input("  Longitude: "))
        
        print("\nDelivery location:")
        del_lat = float(input("  Latitude: "))
        del_lon = float(input("  Longitude: "))
        
        print("\nOrder type (Snack/Meal/Buffet/Drinks):")
        order_type = input("  Type: ").strip()
        
        print("\nVehicle type (motorcycle/scooter/electric_scooter):")
        vehicle_type = input("  Type: ").strip()
        
        print("\n" + "="*70)
        print("🔄 Making prediction...")
        print("="*70)
        
        result = predict_single_order(
            age, rating, rest_lat, rest_lon, del_lat, del_lon, order_type, vehicle_type
        )
        
        if result:
            predicted_time, distance = result
            avg_speed = (distance / predicted_time) * 60 if predicted_time > 0 else 0
            
            print(f"\n🎯 PREDICTION RESULTS:")
            print(f"{'='*70}")
            print(f"⏱️  Estimated Delivery Time: {predicted_time:.1f} minutes")
            print(f"📏 Delivery Distance:       {distance:.2f} km")
            print(f"⚡ Average Speed:            {avg_speed:.1f} km/h")
            print(f"{'='*70}")
    
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
