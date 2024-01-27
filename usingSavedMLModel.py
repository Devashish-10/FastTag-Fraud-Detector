import joblib
import pandas as pd
# Loading the saved model
model = joblib.load('model_FasttagFraudDetection.pkl')
# Getting input from the user for real-time prediction
user_data = {
    'Hour': int(input("Enter Hour:")),
    'DayOfWeek': int(input("Enter Day of Week (0-6, where 0 is Monday and 6 is Sunday):")),
    'Month': int(input("Enter Month (1-12):")),
    'distance_from_city_center': float(input("Enter Distance from City Center in kilometers:")),
    'Vehicle_Speed': float(input("Enter Vehicle Speed:")),
    'Transaction_Amount': float(input("Enter Transaction Amount:")),
    'Amount_paid': float(input("Enter Amount Paid:")),
    'Vehicle_Type_Car': int(input("Is the Vehicle Type Car? (1 for Yes, 0 for No):")),
    'Vehicle_Type_Truck': int(input("Is the Vehicle Type Truck? (1 for Yes, 0 for No):")),
    'Lane_Type_Express': int(input("Is the Lane Type Express? (1 for Yes, 0 for No):")),
    'Lane_Type_Local': int(input("Is the Lane Type Local? (1 for Yes, 0 for No):"))
}
# Converting the user input into a DataFrame
user_data_df = pd.DataFrame([user_data])
# Making predictions using the loaded model
prediction = model.predict(user_data_df)
# Printing the prediction made by our model
print("Predicted Fraud Indicator:", prediction[0])
