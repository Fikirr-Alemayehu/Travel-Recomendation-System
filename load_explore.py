# ===== LOADANDEXPLORE =====

import pandas as pd

# Load datasets with correct filenames
users = pd.read_csv("travel_data/Final_Updated_Expanded_Users.csv")
destinations = pd.read_csv("travel_data/Expanded_Destinations.csv")
trips = pd.read_csv("travel_data/Final_Updated_Expanded_UserHistory.csv")

# Display basic information
print("Users dataset shape:", users.Name)
print("Destinations dataset shape:", destinations.shape)
print("Trips dataset shape:", trips.shape)

# Preview the datasets
print(users.head())
print(destinations.head())
print(trips.head())

