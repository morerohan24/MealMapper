# Complete pipeline - runs EDA and model training

import sys
import os

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    os.system('chcp 65001 >nul 2>&1')
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("\nFood Delivery Time Prediction - ML Pipeline\n")
print("Running EDA and Model Training...")
print("This will take 15-20 minutes...\n")

print("Step 1: Running EDA...")

try:
    from eda_analysis import main as eda_main
    eda_main()
    print("EDA completed!\n")
except Exception as e:
    print(f"Error in EDA: {e}")
    import traceback
    traceback.print_exc()

print("Step 2: Training models...")

try:
    from model_training import main as training_main
    training_main()
    print("Model training completed!\n")
except Exception as e:
    print(f"Error in training: {e}")
    import traceback
    traceback.print_exc()

print("\nPipeline completed!")
print("\nGenerated files:")
print("  - models/ (trained models)")
print("  - visualizations/ (charts)")
print("\nRun the app: streamlit run app.py\n")
