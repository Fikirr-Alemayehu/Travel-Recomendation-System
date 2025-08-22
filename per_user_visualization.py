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



