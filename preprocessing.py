# ===== PREPROCESSING =====

from sklearn.preprocessing import LabelEncoder
import numpy as np

# Column names
user_col = 'UserID'
destination_col = 'DestinationID'
rating_col = 'ExperienceRating'  # rating in trips dataset

# Drop rows with missing ratings (instead of filling with 1)
print("Before dropna:", trips.shape)
trips = trips.dropna(subset=[rating_col])
print("After dropna:", trips.shape)

# Encode IDs
user_encoder = LabelEncoder()
destination_encoder = LabelEncoder()

trips['user_id_encoded'] = user_encoder.fit_transform(trips[user_col])
trips['destination_id_encoded'] = destination_encoder.fit_transform(trips[destination_col])

# Duplicate checker
dup_count = trips.duplicated(
    subset=['user_id_encoded', 'destination_id_encoded'],
    keep=False
).sum()
print("Duplicate (user,item) rows found:", dup_count)

# Resolve duplicates by averaging ratings
trips = (trips
         .groupby(['user_id_encoded','destination_id_encoded'], as_index=False)[rating_col]
         .mean())

# Build ratings dataframe for Surprise
ratings = trips.rename(columns={rating_col: 'Rating'})[['user_id_encoded','destination_id_encoded','Rating']]

ratings.head()

