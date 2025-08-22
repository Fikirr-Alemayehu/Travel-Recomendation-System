# ===== COMPARESVDWITHOTHERALGORITHMS =====

from surprise import KNNBasic, BaselineOnly

algorithms = {
    "SVD": SVD(random_state=SEED),
    "KNNBasic": KNNBasic(),
    "BaselineOnly": BaselineOnly()
}

results = {}
for name, algo in algorithms.items():
    tr, te = train_test_split(data, test_size=0.2, random_state=SEED)
    algo.fit(tr)
    preds = algo.test(te)
    results[name] = accuracy.rmse(preds, verbose=False)

print("RMSE comparison:")
for name, score in results.items():
    print(f"{name}: {score:.4f}")

