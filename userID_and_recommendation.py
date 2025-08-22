# ===== USERIDANDRECOMENDATION =====

# Get unique user IDs from test_predictions
unique_users = set([pred.uid for pred in test_predictions])
print("Sample user IDs in test_predictions:", list(unique_users)[:10])

# Predict ratings for all destinations for a given user
all_items = ratings['destination_id_encoded'].unique()
user_id = 10  # Example user

# Re-assign the model variable to the SVD object
model = algorithms["SVD"]

user_predictions = [model.predict(user_id, iid) for iid in all_items]

# Sort and pick top-5
top_5 = sorted(user_predictions, key=lambda x: x.est, reverse=True)[:5]

# Map back to original Destination IDs
top_destinations = [destination_encoder.inverse_transform([rec.iid])[0] for rec in top_5]
print(f"Top 5 recommended destinations for user {user_id}: {top_destinations}")

