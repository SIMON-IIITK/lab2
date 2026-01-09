import pandas as pd
import json
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# EXPERIMENT CONFIGURATION
# (Edit these for Task 5 experiments)
# ==========================================
MODEL_TYPE = "LinearRegression"  # Options: LinearRegression, Ridge, Lasso
TEST_SIZE = 0.3
RANDOM_STATE = 42
ALPHA = 0.1  # Only for Ridge/Lasso

# Ensure output directory exists
os.makedirs("output", exist_ok=True)

def main():
    print("Loading data...")
    # Load dataset (Note: this dataset uses ';' as separator)
    data = pd.read_csv('dataset/winequality-red.csv', sep=';')

    # 1. Prepare Data
    X = data.drop('quality', axis=1)
    y = data['quality']

    # 2. Split and Preprocess
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Train Model
    print(f"Training {MODEL_TYPE}...")
    if MODEL_TYPE == "LinearRegression":
        model = LinearRegression()
    elif MODEL_TYPE == "Ridge":
        model = Ridge(alpha=ALPHA)
    elif MODEL_TYPE == "Lasso":
        model = Lasso(alpha=ALPHA)
    
    model.fit(X_train_scaled, y_train)

    # 4. Evaluate
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE: {mse}")
    print(f"R2 Score: {r2}")

    # 5. Save Artifacts
    # Save Model
    joblib.dump(model, 'output/model.pkl')
    
    # Save Metrics to JSON
    metrics = {
        "model_type": MODEL_TYPE,
        "mse": mse,
        "r2_score": r2,
        "hyperparameters": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "alpha": ALPHA
        }
    }
    
    with open('output/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()