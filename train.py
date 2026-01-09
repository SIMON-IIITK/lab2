import pandas as pd
import json
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
#  LAB 1 EXPERIMENT CONFIGURATION
#  (Edit this section for each experiment)
# ==========================================
MODEL_TYPE = "LinearRegression"   # Options: LinearRegression, Ridge, RandomForest
SCALING = False                   # Set True for Ridge, False for others (per Lab 1)
TEST_SIZE = 0.2                   # 80/20 Split
RANDOM_STATE = 42

# Hyperparameters
ALPHA = 1.0                       # For Ridge
N_ESTIMATORS = 50                 # For Random Forest
MAX_DEPTH = 10                    # For Random Forest

# Ensure output directory exists
os.makedirs("output", exist_ok=True)

def main():
    print("Loading data...")
    # Load dataset
    try:
        data = pd.read_csv('dataset/winequality-red.csv', sep=';')
    except:
        print("Error: Dataset file not found.")
        return

    X = data.drop('quality', axis=1)
    y = data['quality']

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # Preprocessing (Scaling) - Only if SCALING is True
    if SCALING:
        print("Applying StandardScaler...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        print("No scaling applied (Raw data)...")

    # Model Selection
    print(f"Training {MODEL_TYPE}...")
    if MODEL_TYPE == "LinearRegression":
        model = LinearRegression()
    elif MODEL_TYPE == "Ridge":
        model = Ridge(alpha=ALPHA)
    elif MODEL_TYPE == "RandomForest":
        model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS, 
            max_depth=MAX_DEPTH, 
            random_state=RANDOM_STATE
        )
    else:
        raise ValueError("Unknown model type")

    # Train and Evaluate
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE: {mse}")
    print(f"R2 Score: {r2}")

    # Save Artifacts
    joblib.dump(model, 'output/model.pkl')
    
    metrics = {
        "model_type": MODEL_TYPE,
        "mse": mse,
        "r2_score": r2,
        "config": {
            "scaling": SCALING,
            "alpha": ALPHA,
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH
        }
    }
    
    with open('output/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()