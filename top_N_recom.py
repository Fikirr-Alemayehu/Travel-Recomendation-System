# ===== TOPNRECOMMENDATIONS =====

# Pick any encoded user ID that exists
all_items_encoded = ratings['destination_id_encoded'].unique()
some_user = next(iter(user_pred_map.keys()))  # pick a user from the test predictions

# Predict all items for that user (for a “full catalog” top-N)
user_preds_full = [model.predict(some_user, iid) for iid in all_items_encoded]
top_5 = sorted(user_preds_full, key=lambda x: x.est, reverse=True)[:5]

# Map back to original DestinationID and names
top_destinations_orig = [destination_encoder.inverse_transform([pred.iid])[0] for pred in top_5]
dest_names = [destinations.loc[destinations['DestinationID'] == d, 'Name'].values[0] for d in top_destinations_orig]
scores = [pred.est for pred in top_5]

plt.figure(figsize=(10,6))
plt.barh(dest_names, scores, color='skyblue')
plt.xlabel("Predicted Rating")
plt.title(f"Top 5 Destinations for User {some_user}\nHR@10={avg_hr:.3f}, NDCG@10={avg_ndcg:.3f}")
plt.gca().invert_yaxis()
plt.show()

