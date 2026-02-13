# Model training and evaluation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

from preprocessing import preprocess_pipeline


def evaluate_model(y_true, y_pred, model_name="Model"):
    """Calculate and display model performance metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{'='*50}")
    print(f"📊 {model_name} Performance Metrics")
    print(f"{'='*50}")
    print(f"🎯 RMSE (Root Mean Squared Error): {rmse:.4f} minutes")
    print(f"🎯 MAE (Mean Absolute Error):     {mae:.4f} minutes")
    print(f"🎯 R² Score:                       {r2:.4f}")
    print(f"{'='*50}")
    
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}


def train_multiple_models(X_train, y_train, X_test, y_test):
    """Train and compare multiple regression models"""
    print("\n" + "="*70)
    print("🤖 TRAINING MULTIPLE REGRESSION MODELS")
    print("="*70)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'Decision Tree': DecisionTreeRegressor(max_depth=15, random_state=42),
        'Random Forest': RandomForestRegressor(
            n_estimators=100, 
            max_depth=20, 
            random_state=42, 
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100, 
            max_depth=10, 
            learning_rate=0.1, 
            random_state=42
        ),
        'XGBoost': XGBRegressor(
            n_estimators=100, 
            max_depth=10, 
            learning_rate=0.1, 
            random_state=42,
            n_jobs=-1
        )
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\n🔄 Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Evaluate on training set
        print(f"\n📈 Training Set Performance:")
        train_metrics = evaluate_model(y_train, y_pred_train, f"{name} (Train)")
        
        # Evaluate on test set
        print(f"\n📉 Test Set Performance:")
        test_metrics = evaluate_model(y_test, y_pred_test, f"{name} (Test)")
        
        # Store results
        results[name] = {
            'train': train_metrics,
            'test': test_metrics
        }
        trained_models[name] = model
    
    return trained_models, results


def plot_model_comparison(results, save_path=None):
    if save_path is None:
        import os
        save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'visualizations', 'model_comparison.png')
    """Create visualization comparing model performances"""
    print("\n📊 Creating model comparison visualizations...")
    
    # Prepare data for plotting
    model_names = list(results.keys())
    rmse_scores = [results[name]['test']['RMSE'] for name in model_names]
    mae_scores = [results[name]['test']['MAE'] for name in model_names]
    r2_scores = [results[name]['test']['R2'] for name in model_names]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # RMSE comparison
    axes[0].barh(model_names, rmse_scores, color='coral')
    axes[0].set_xlabel('RMSE (minutes)', fontsize=12)
    axes[0].set_title('Root Mean Squared Error', fontsize=14)
    axes[0].invert_yaxis()
    for i, v in enumerate(rmse_scores):
        axes[0].text(v + 0.1, i, f'{v:.2f}', va='center')
    
    # MAE comparison
    axes[1].barh(model_names, mae_scores, color='skyblue')
    axes[1].set_xlabel('MAE (minutes)', fontsize=12)
    axes[1].set_title('Mean Absolute Error', fontsize=14)
    axes[1].invert_yaxis()
    for i, v in enumerate(mae_scores):
        axes[1].text(v + 0.1, i, f'{v:.2f}', va='center')
    
    # R² comparison
    axes[2].barh(model_names, r2_scores, color='lightgreen')
    axes[2].set_xlabel('R² Score', fontsize=12)
    axes[2].set_title('R² Score (Higher is Better)', fontsize=14)
    axes[2].invert_yaxis()
    for i, v in enumerate(r2_scores):
        axes[2].text(v + 0.01, i, f'{v:.3f}', va='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Model comparison saved to {save_path}")
    plt.close()


def plot_predictions(y_true, y_pred, model_name, save_path):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.5, s=10)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Delivery Time (minutes)', fontsize=12)
    plt.ylabel('Predicted Delivery Time (minutes)', fontsize=12)
    plt.title(f'{model_name}: Actual vs Predicted', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Prediction plot saved to {save_path}")
    plt.close()


def plot_residuals(y_true, y_pred, model_name, save_path):
    """Plot residual distribution"""
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{model_name}: Residual Analysis', fontsize=16, fontweight='bold')
    
    # Residual plot
    axes[0].scatter(y_pred, residuals, alpha=0.5, s=10)
    axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('Predicted Delivery Time (minutes)', fontsize=12)
    axes[0].set_ylabel('Residuals (minutes)', fontsize=12)
    axes[0].set_title('Residual Plot', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Residual distribution
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Residuals (minutes)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Residual Distribution', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Residual plot saved to {save_path}")
    plt.close()


def plot_feature_importance(model, feature_names, save_path):
    """Plot feature importance for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1][:15]  # Top 15 features
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importance[indices], color='teal')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title('Top 15 Most Important Features', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        
        for i, v in enumerate(importance[indices]):
            plt.text(v + 0.001, i, f'{v:.4f}', va='center')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Feature importance plot saved to {save_path}")
        plt.close()
    else:
        print(f"⚠️ Model does not have feature_importances_ attribute")


def select_best_model(results, trained_models):
    """Select the best model based on test RMSE"""
    print("\n" + "="*70)
    print("🏆 SELECTING BEST MODEL")
    print("="*70)
    
    best_model_name = min(results, key=lambda x: results[x]['test']['RMSE'])
    best_model = trained_models[best_model_name]
    best_metrics = results[best_model_name]['test']
    
    print(f"\n✨ Best Model: {best_model_name}")
    print(f"📊 Test RMSE: {best_metrics['RMSE']:.4f} minutes")
    print(f"📊 Test MAE:  {best_metrics['MAE']:.4f} minutes")
    print(f"📊 Test R²:   {best_metrics['R2']:.4f}")
    
    return best_model_name, best_model, best_metrics


def save_model_artifacts(model, scaler, encoders, feature_cols, save_dir=None):
    if save_dir is None:
        import os
        save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models') + os.sep
    """Save trained model and preprocessing artifacts"""
    print(f"\n💾 Saving model artifacts to {save_dir}...")
    
    joblib.dump(model, save_dir + 'best_model.pkl')
    joblib.dump(scaler, save_dir + 'scaler.pkl')
    joblib.dump(encoders, save_dir + 'encoders.pkl')
    joblib.dump(feature_cols, save_dir + 'feature_cols.pkl')
    
    print("✅ Model artifacts saved successfully!")
    print(f"   - best_model.pkl")
    print(f"   - scaler.pkl")
    print(f"   - encoders.pkl")
    print(f"   - feature_cols.pkl")


def main():
    """Main training pipeline"""
    print("\n" + "🚀"*35)
    print(" "*20 + "FOOD DELIVERY TIME PREDICTION")
    print(" "*25 + "ML Training Pipeline")
    print("🚀"*35 + "\n")
    
    # Preprocess data
    import os
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Dataset.csv')
    X_train, X_test, y_train, y_test, scaler, encoders, feature_cols, df = preprocess_pipeline(
        dataset_path, 
        for_training=True
    )
    
    # Train multiple models
    trained_models, results = train_multiple_models(X_train, y_train, X_test, y_test)
    
    # Create model comparison visualization
    plot_model_comparison(results)
    
    # Select best model
    best_model_name, best_model, best_metrics = select_best_model(results, trained_models)
    
    # Create visualizations for best model
    y_pred_test = best_model.predict(X_test)
    import os
    viz_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'visualizations')
    
    plot_predictions(
        y_test, 
        y_pred_test, 
        best_model_name, 
        os.path.join(viz_dir, f'{best_model_name.replace(" ", "_")}_predictions.png')
    )
    
    plot_residuals(
        y_test, 
        y_pred_test, 
        best_model_name, 
        os.path.join(viz_dir, f'{best_model_name.replace(" ", "_")}_residuals.png')
    )
    
    plot_feature_importance(
        best_model, 
        feature_cols, 
        os.path.join(viz_dir, f'{best_model_name.replace(" ", "_")}_feature_importance.png')
    )
    
    # Save model and artifacts
    save_model_artifacts(best_model, scaler, encoders, feature_cols)
    
    print("\n" + "="*70)
    print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\n🎯 Final Model: {best_model_name}")
    print(f"📊 RMSE: {best_metrics['RMSE']:.4f} minutes")
    print(f"📊 MAE:  {best_metrics['MAE']:.4f} minutes")
    print(f"📊 R²:   {best_metrics['R2']:.4f}")
    print("\n✨ Ready for deployment!")


if __name__ == "__main__":
    main()
