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

