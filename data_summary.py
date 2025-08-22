# ===== DATASUMMARY =====

n_users = trips['UserID'].nunique()
n_items = trips['DestinationID'].nunique()
n_interactions = len(trips)

sparsity = 1 - (n_interactions / (n_users * n_items))

print("Dataset Summary:")
print(f"Number of users: {n_users}")
print(f"Number of destinations: {n_items}")
print(f"Number of interactions (ratings): {n_interactions}")
print(f"Sparsity: {sparsity:.2%}")


