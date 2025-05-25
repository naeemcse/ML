# House Price Prediction with Comprehensive Logging
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Add a file handler to save logs to a file (in Colab's temporary storage)
log_filename = f"house_price_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
file_handler = logging.FileHandler(log_filename)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

logger.info("Starting House Price Prediction Project")
print("Project started. Check logs for details.")

# Step 1: Load and Explore Data
logger.info("Step 1: Loading and exploring data")
try:
    # Load dataset (using Boston housing dataset as example)
    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing()
    data = pd.DataFrame(housing.data, columns=housing.feature_names)
    data['PRICE'] = housing.target
    
    logger.info(f"Data loaded successfully. Shape: {data.shape}")
    print("Data loaded successfully. First 5 rows:")
    print(data.head())
    
    # Basic statistics
    logger.info("Displaying basic statistics")
    print("\nData Statistics:")
    print(data.describe())
    
except Exception as e:
    logger.error(f"Error loading data: {str(e)}")
    raise

# Step 2: Data Preprocessing
logger.info("Step 2: Data preprocessing")
try:
    # Check for missing values
    logger.info("Checking for missing values")
    missing_values = data.isnull().sum()
    print("\nMissing Values:")
    print(missing_values)
    
    # No missing values in this dataset, but here's how we'd handle them:
    if missing_values.any():
        logger.warning(f"Missing values found: {missing_values[missing_values > 0]}")
        # data = data.fillna(data.mean())  # Example imputation
    else:
        logger.info("No missing values found")
    
    # Feature scaling
    logger.info("Scaling features using StandardScaler")
    scaler = StandardScaler()
    features = data.drop('PRICE', axis=1)
    scaled_features = scaler.fit_transform(features)
    
    # Create scaled DataFrame
    scaled_data = pd.DataFrame(scaled_features, columns=housing.feature_names)
    scaled_data['PRICE'] = data['PRICE']
    
    logger.info("Feature scaling completed")
    print("\nFirst 5 rows after scaling:")
    print(scaled_data.head())
    
except Exception as e:
    logger.error(f"Error during preprocessing: {str(e)}")
    raise

# Step 3: Train-Test Split
logger.info("Step 3: Creating train-test split")
try:
    X = scaled_data.drop('PRICE', axis=1)
    y = scaled_data['PRICE']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    logger.info(f"Train-test split completed. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"\nTrain set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    
except Exception as e:
    logger.error(f"Error during train-test split: {str(e)}")
    raise

# Step 4: Model Training
logger.info("Step 4: Training Linear Regression model")
try:
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    logger.info("Model training completed")
    print("\nModel trained successfully")
    print(f"Coefficients: {model.coef_}")
    print(f"Intercept: {model.intercept_}")
    
except Exception as e:
    logger.error(f"Error during model training: {str(e)}")
    raise

# Step 5: Model Evaluation
logger.info("Step 5: Evaluating model performance")
try:
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"Model evaluation complete. RMSE: {rmse:.2f}, R2: {r2:.2f}")
    print("\nModel Performance:")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R2): {r2:.2f}")
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs Predicted House Prices")
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.show()
    logger.info("Actual vs Predicted plot displayed")
    
except Exception as e:
    logger.error(f"Error during model evaluation: {str(e)}")
    raise

# Step 6: Make Predictions on New Data
logger.info("Step 6: Making predictions on new data")
try:
    # Create some sample data for prediction (using mean values from dataset)
    sample_data = X_train.mean().values.reshape(1, -1)
    predicted_price = model.predict(sample_data)
    
    logger.info(f"Prediction made for sample data. Predicted price: {predicted_price[0]:.2f}")
    print("\nSample Prediction:")
    print(f"For features: {sample_data}")
    print(f"Predicted house price: ${predicted_price[0]*100000:,.2f}")
    
except Exception as e:
    logger.error(f"Error during prediction: {str(e)}")
    raise

logger.info("House Price Prediction Project completed successfully")
print("\nProject completed successfully. Check logs for detailed information.")

# Save the log file to Google Drive (optional)
try:
    from google.colab import drive
    drive.mount('/content/drive')
    !cp "{log_filename}" "/content/drive/MyDrive/"
    logger.info(f"Log file saved to Google Drive: {log_filename}")
except Exception as e:
    logger.warning(f"Could not save log to Google Drive: {str(e)}")