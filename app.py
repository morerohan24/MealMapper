# Streamlit app for delivery time prediction

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Delivery Time Predictor",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_artifacts():
    """Load trained model and preprocessing artifacts"""
    try:
        model = joblib.load('models/best_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        encoders = joblib.load('models/encoders.pkl')
        feature_cols = joblib.load('models/feature_cols.pkl')
        return model, scaler, encoders, feature_cols
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        st.stop()


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points using Haversine formula"""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r


def engineer_features(input_data, encoders):
    """Engineer features from input data"""
    # Calculate distances
    input_data['delivery_distance_km'] = haversine_distance(
        input_data['Restaurant_latitude'],
        input_data['Restaurant_longitude'],
        input_data['Delivery_location_latitude'],
        input_data['Delivery_location_longitude']
    )
    
    # Calculate coordinate differences
    input_data['lat_diff'] = abs(input_data['Delivery_location_latitude'] - input_data['Restaurant_latitude'])
    input_data['lon_diff'] = abs(input_data['Delivery_location_longitude'] - input_data['Restaurant_longitude'])
    
    # Distance from center (using approximate mean values from training)
    mean_lat = 16.0
    mean_lon = 77.0
    
    input_data['restaurant_distance_from_center'] = np.sqrt(
        (input_data['Restaurant_latitude'] - mean_lat)**2 + 
        (input_data['Restaurant_longitude'] - mean_lon)**2
    )
    
    input_data['delivery_distance_from_center'] = np.sqrt(
        (input_data['Delivery_location_latitude'] - mean_lat)**2 + 
        (input_data['Delivery_location_longitude'] - mean_lon)**2
    )
    
    # Create rating and age categories
    if input_data['Delivery_person_Ratings'] <= 3.5:
        rating_cat = 'Low'
    elif input_data['Delivery_person_Ratings'] <= 4.2:
        rating_cat = 'Medium'
    else:
        rating_cat = 'High'
    input_data['rating_category'] = rating_cat
    
    if input_data['Delivery_person_Age'] <= 25:
        age_cat = 'Young'
    elif input_data['Delivery_person_Age'] <= 35:
        age_cat = 'Mid'
    else:
        age_cat = 'Senior'
    input_data['age_group'] = age_cat
    
    # Interaction features
    input_data['distance_rating_interaction'] = input_data['delivery_distance_km'] * input_data['Delivery_person_Ratings']
    input_data['distance_age_interaction'] = input_data['delivery_distance_km'] * input_data['Delivery_person_Age']
    
    # Encode categorical variables
    input_data['Type_of_order_encoded'] = encoders['Type_of_order'].transform([input_data['Type_of_order']])[0]
    input_data['Type_of_vehicle_encoded'] = encoders['Type_of_vehicle'].transform([input_data['Type_of_vehicle']])[0]
    input_data['rating_category_encoded'] = encoders['rating_category'].transform([input_data['rating_category']])[0]
    input_data['age_group_encoded'] = encoders['age_group'].transform([input_data['age_group']])[0]
    
    return input_data


def predict_delivery_time(input_data, model, scaler, encoders, feature_cols):
    """Make prediction on input data"""
    # Engineer features
    input_data = engineer_features(input_data, encoders)
    
    # Select required features
    feature_values = [input_data[col] for col in feature_cols]
    features_df = pd.DataFrame([feature_values], columns=feature_cols)
    
    # Scale features
    features_scaled = scaler.transform(features_df)
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    
    return prediction, input_data['delivery_distance_km']


def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<p class="main-header">Food Delivery Time Predictor</p>', unsafe_allow_html=True)
    st.write("Predict delivery times using ML")
    
    # Load model
    model, scaler, encoders, feature_cols = load_model_artifacts()
    
    # Sidebar
    st.sidebar.header("Input Details")
    
    # Navigation
    page = st.sidebar.radio("Select Page", ["Predict", "Model Info", "About"])
    
    if page == "Predict":
        st.header("Predict Delivery Time")
        st.write("Enter order details below:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Delivery Partner")
            partner_age = st.number_input("Partner Age", min_value=18, max_value=65, value=30)
            partner_rating = st.slider("Partner Rating (out of 5)", min_value=1.0, max_value=5.0, value=4.5, step=0.1)
            
            st.subheader("Order Details")
            order_type = st.selectbox("Order Type", ["Snack", "Meal", "Buffet", "Drinks"])
            vehicle_type = st.selectbox("Vehicle Type", ["motorcycle", "scooter", "electric_scooter"])
        
        with col2:
            st.subheader("Restaurant Location")
            rest_lat = st.number_input("Restaurant Latitude", value=12.9716, format="%.4f")
            rest_lon = st.number_input("Restaurant Longitude", value=77.5946, format="%.4f")
            
            st.subheader("Delivery Location")
            del_lat = st.number_input("Delivery Latitude", value=13.0716, format="%.4f")
            del_lon = st.number_input("Delivery Longitude", value=77.6946, format="%.4f")
        
        st.markdown("---")
        st.subheader("Quick Presets")
        preset_col1, preset_col2, preset_col3 = st.columns(3)
        
        with preset_col1:
            if st.button("Bangalore"):
                rest_lat, rest_lon = 12.9716, 77.5946
                del_lat, del_lon = 13.0358, 77.6431
        
        with preset_col2:
            if st.button("Mumbai"):
                rest_lat, rest_lon = 19.0760, 72.8777
                del_lat, del_lon = 19.1136, 72.9083
        
        with preset_col3:
            if st.button("Delhi"):
                rest_lat, rest_lon = 28.7041, 77.1025
                del_lat, del_lon = 28.5355, 77.3910
        
        st.markdown("---")
        
        # Predict button
        if st.button("Predict", type="primary", use_container_width=True):
            with st.spinner("Calculating delivery time..."):
                # Prepare input data
                input_data = {
                    'Delivery_person_Age': partner_age,
                    'Delivery_person_Ratings': partner_rating,
                    'Restaurant_latitude': rest_lat,
                    'Restaurant_longitude': rest_lon,
                    'Delivery_location_latitude': del_lat,
                    'Delivery_location_longitude': del_lon,
                    'Type_of_order': order_type,
                    'Type_of_vehicle': vehicle_type
                }
                
                # Make prediction
                predicted_time, distance = predict_delivery_time(
                    input_data.copy(), model, scaler, encoders, feature_cols
                )
                
                # Display results
                st.success("Prediction Complete!")
                
                # Metrics
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric(
                        label="Estimated Time",
                        value=f"{predicted_time:.1f} min"
                    )
                
                with metric_col2:
                    st.metric(
                        label="Distance",
                        value=f"{distance:.2f} km"
                    )
                
                with metric_col3:
                    avg_speed = (distance / predicted_time) * 60 if predicted_time > 0 else 0
                    st.metric(
                        label="Avg Speed",
                        value=f"{avg_speed:.1f} km/h"
                    )
                
                st.markdown("---")
                st.subheader("Route Map")
                
                # Create map
                map_data = pd.DataFrame({
                    'lat': [rest_lat, del_lat],
                    'lon': [rest_lon, del_lon],
                    'location': ['Restaurant', 'Delivery Address']
                })
                
                fig = px.scatter_mapbox(
                    map_data,
                    lat='lat',
                    lon='lon',
                    text='location',
                    color='location',
                    color_discrete_map={'Restaurant': 'red', 'Delivery Address': 'blue'},
                    zoom=11,
                    height=400
                )
                
                fig.update_layout(
                    mapbox_style="open-street-map",
                    margin={"r": 0, "t": 0, "l": 0, "b": 0}
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Additional info
                st.markdown("---")
                st.write("### Summary")
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.write(f"Order: {order_type}")
                    st.write(f"Vehicle: {vehicle_type.replace('_', ' ').title()}")
                    st.write(f"Rating: {partner_rating}/5.0")
                
                with col_b:
                    if predicted_time < 25:
                        category = "Fast"
                    elif predicted_time < 35:
                        category = "Normal"
                    else:
                        category = "Slow"
                    
                    st.write(f"Category: {category}")
                    st.write(f"Distance: {distance:.2f} km")
                    st.write(f"ETA: {predicted_time:.1f} min")
    
    elif page == "Model Info":
        st.header("Model Information")
        
        st.write("### Features Used")
        st.write("""
        - Delivery partner age and rating
        - Restaurant and delivery locations
        - Order type and vehicle type
        - Distance calculation
        - Other engineered features
        """)
        
        st.write("### Models Compared")
        st.write("""
        - Linear Regression
        - Ridge/Lasso
        - Random Forest
        - XGBoost
        - Gradient Boosting
        """)
        
        st.write("### Key Findings")
        st.write("""
        1. Distance is the main predictor
        2. Partner rating matters
        3. Vehicle type affects speed
        4. Mid-age partners perform best
        """)
        
        st.info(f"Features: {len(feature_cols)}")
    
    elif page == "About":
        st.header("About")
        
        st.write("### Project")
        st.write("ML-based delivery time prediction for food delivery platforms")
        
        st.write("### Tech Stack")
        st.write("""
        - Python, scikit-learn, XGBoost
        - Pandas, NumPy
        - Streamlit
        """)
        
        st.write("### Dataset")
        st.write("""
        - 45,000+ delivery records
        - Features: partner info, locations, order details
        - Target: delivery time (minutes)
        """)
        
        st.write("### Approach")
        st.write("""
        1. Data preprocessing and cleaning
        2. Feature engineering
        3. Trained multiple models
        4. Selected best model based on RMSE
        5. Deployed with Streamlit
        """)


if __name__ == "__main__":
    main()
