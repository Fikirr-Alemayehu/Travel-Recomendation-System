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

