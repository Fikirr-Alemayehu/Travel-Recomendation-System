# ===== SETUP =====
from google.colab import files
files.upload()  # Upload kaggle.json here

!pip install kaggle scikit-surprise numpy==1.26.4 pandas matplotlib

# Configure Kaggle API
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download dataset
!kaggle datasets download -d amanmehra23/travel-recommendation-dataset
!unzip travel-recommendation-dataset.zip -d travel_data
!ls travel_data

# ===== INSTALLSURPRISE =====

!Install Surprise library for collaborative filtering
!pip install scikit-surprise --upgrade --no-cache-dir
!pip install numpy pandas matplotlib

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


# ===== COLUMN =====

# Check columns of each dataset
print("Users columns:", users.columns)
print("Destinations columns:", destinations.columns)
print("Trips columns:", trips.columns)

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

# ===== EXPLORATORYDATAANALYSIS =====

import matplotlib.pyplot as plt

# Distribution of ratings
plt.figure(figsize=(8,5))
ratings['Rating'].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()

# Most popular destinations
popular_destinations = ratings.groupby('destination_id_encoded')['Rating'].count().sort_values(ascending=False).head(10)
print("Top 10 popular destinations (encoded IDs):\n", popular_destinations)

# ===== REPRODUCIBLITYSETUP =====

from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Reader scale based on dataset
rating_min, rating_max = float(ratings['Rating'].min()), float(ratings['Rating'].max())
reader = Reader(rating_scale=(rating_min, rating_max))

data = Dataset.load_from_df(ratings[['user_id_encoded','destination_id_encoded','Rating']], reader)

# Train-test split
trainset, testset = train_test_split(data, test_size=0.2, random_state=SEED)
print(f"Trainset ratings: {trainset.n_ratings}, Testset ratings: {len(testset)}")

# ===== BUILDSVDMODEL =====

# Train the SVD model
#model = SVD()
model = SVD(random_state=SEED)
model.fit(trainset)

# Evaluate model
predictions = model.test(testset)
rmse = accuracy.rmse(predictions)
print("RMSE:", rmse)

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



