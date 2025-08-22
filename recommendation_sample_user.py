# ===== RECOMMENDATIONWITHSAMPLEUSER =====

from collections import defaultdict
import numpy as np

# Function to calculate Hit Ratio@K and NDCG@K for each user
def evaluate_per_user(predictions, K=10):
    user_ratings = defaultdict(list)

    for uid, iid, true_r, est, _ in predictions:
        user_ratings[uid].append((iid, est, true_r))

    results = {}

    for uid, ratings in user_ratings.items():
        # Sort predictions by estimated rating
        ratings.sort(key=lambda x: x[1], reverse=True)

        # Take Top-K
        top_k = ratings[:K]
        top_k_items = [iid for (iid, _, _) in top_k]

        # For HR@K: check if any ground truth item is in Top-K
        hr = 1 if any(true_r >= 4.0 and iid in top_k_items for (iid, _, true_r) in ratings) else 0

        # For NDCG@K: gain = 1 if relevant item appears in Top-K, discounted by log rank
        ndcg = 0
        for rank, (iid, _, true_r) in enumerate(top_k, start=1):
            if true_r >= 4.0:
                ndcg = 1 / np.log2(rank + 1)
                break

        results[uid] = (hr, ndcg)

    return results


# Compute per-user metrics
user_metrics = evaluate_per_user(test_predictions, K=10)

# Example: visualize recommendations for multiple users
sample_users = [1, 514, 3]  # replace with user IDs in your dataset

for user_id in sample_users:
    user_predictions = [pred for pred in test_predictions if pred.uid == user_id]

    if not user_predictions:
        print(f"No predictions available for User {user_id}")
        continue

    # Sort predictions by estimated rating
    top_5 = sorted(user_predictions, key=lambda x: x.est, reverse=True)[:5]
    top_destinations = [rec.iid for rec in top_5]

    # Get destination names
    dest_names = [
        destinations.loc[destinations['DestinationID'] == d, 'Name'].values[0]
        for d in top_destinations
    ]
    scores = [rec.est for rec in top_5]

    # Get per-user HR and NDCG
    hr, ndcg = user_metrics.get(user_id, (0, 0))

    # Plot with per-user metrics in title
    plt.figure(figsize=(10, 6))
    plt.barh(dest_names, scores, color='skyblue')
    plt.xlabel("Predicted Rating")
    plt.title(
        f"Top 5 Recommended Destinations for User {user_id}\n"
        f"Hit Ratio@10: {hr:.3f}, NDCG@10: {ndcg:.3f}"
    )
    plt.gca().invert_yaxis()
    plt.show()

# ===== PERUSERVISUALIZATION =====

import random

def plot_recommendations_for_user(test_predictions, user_id, top_k=5):
    user_preds = [pred for pred in test_predictions if pred.uid == user_id]
    if not user_preds:
        print(f"No predictions for user {user_id}")
        return

    user_preds_sorted = sorted(user_preds, key=lambda x: x.est, reverse=True)[:top_k]
    dest_ids = [int(pred.iid) for pred in user_preds_sorted]
    scores = [pred.est for pred in user_preds_sorted]

    dest_names = []
    for d in dest_ids:
        if d in destinations['DestinationID'].values:
            dest_names.append(destinations.loc[destinations['DestinationID'] == d, 'Name'].values[0])
        else:
            dest_names.append(f"Destination {d}")

    plt.figure(figsize=(10,6))
    plt.barh(dest_names, scores, color='skyblue')
    plt.xlabel("Predicted Rating")
    plt.title(f"Top {top_k} Recommended Destinations for User {user_id}\n" f"Hit Ratio@10: {hr:.3f}, NDCG@10: {ndcg:.3f}")
    plt.gca().invert_yaxis()
    plt.show()

# Pick random sample users
unique_users = list(set([pred.uid for pred in predictions]))
sample_users = random.sample(unique_users, 3)

print("Random sample users:", sample_users)
for uid in sample_users:
    plot_recommendations_for_user(predictions, uid, top_k=5)



