# ===== EVALUATEMODELWITHLEAVEOUT =====

from collections import defaultdict

def hit_ratio_at_k(pred_list, actual_items, k=10):
    """pred_list: [(iid, est), ...] sorted desc by est; actual_items: list of iids."""
    top_k = [iid for iid, _ in pred_list[:k]]
    return int(any(i in top_k for i in actual_items))

def ndcg_at_k(pred_list, actual_items, k=10):
    dcg = 0.0
    for rank, (iid, _) in enumerate(pred_list[:k], start=1):
        if iid in actual_items:
            dcg += 1.0 / np.log2(rank + 1)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(actual_items), k) + 1))
    return dcg / idcg if idcg > 0 else 0.0

# Get predictions for all users in the testset
test_predictions = model.test(testset)

# Build user->predictions map from Surprise predictions
user_pred_map = defaultdict(list)
for p in test_predictions:
    user_pred_map[p.uid].append((p.iid, p.est))

# Sort predictions per user
for uid in user_pred_map:
    user_pred_map[uid].sort(key=lambda x: x[1], reverse=True)

# Build DataFrame for actual test interactions
testset_df = pd.DataFrame(testset, columns=['uid','iid','r_ui'])

# Evaluate averages
hr_list, ndcg_list = [], []
for uid, pred_items in user_pred_map.items():
    actual_items = testset_df.loc[testset_df['uid'] == uid, 'iid'].tolist()
    if not actual_items:
        continue
    hr_list.append(hit_ratio_at_k(pred_items, actual_items, k=10))
    ndcg_list.append(ndcg_at_k(pred_items, actual_items, k=10))

avg_hr, avg_ndcg = float(np.mean(hr_list)) if hr_list else 0.0, float(np.mean(ndcg_list)) #if ndcg_list else 0.0
print(f"Average Hit Ratio@10: {avg_hr:.3f}")
print(f"Average NDCG@10: {avg_ndcg:.3f}")

