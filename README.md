🌍 Travel Recommendation System

This project builds a personalized travel recommendation system using Collaborative Filtering (Matrix Factorization with SVD, KNN, and Baseline models). The goal is to recommend top travel destinations for users based on their past experiences and ratings.

📊 Dataset

Source: [Kaggle Travel Recommendation Dataset](https://www.kaggle.com/datasets/amanmehra23/travel-recommendation-dataset)
Files:
Final_Updated_Expanded_Users.csv → User profiles
Expanded_Destinations.csv → Destination details
Final_Updated_Expanded_UserHistory.csv → User travel history & ratings

⚙️ Setup

Clone the repo and install dependencies:

git clone https://github.com/Fikirr-Alemayehu/Travel-Recomendation-System.git
cd Travel_Recomendation
Dependencies include:
Python 
pandas, numpy, matplotlib
scikit-surprise (for collaborative filtering)
scikit-learn

🚀 How to Run

Run the notebook in Google Colab (recommended).
Travel_Recomendation.ipynb
Or, run the pipeline using the .py scripts. Example:
python preprocessing.py
python model.py
python recommendations.py

📈 Models Implemented

SVD (Singular Value Decomposition) → Collaborative filtering (matrix factorization)
KNNBasic → User-based / Item-based collaborative filtering
BaselineOnly → Baseline predictor (global mean, user bias, item bias)
Evaluation Metrics:
RMSE (Root Mean Square Error)
Hit Ratio@K
NDCG@K (Normalized Discounted Cumulative Gain)

🎯 Results

SVD outperforms baseline methods in RMSE.
Top-N recommendations are visualized per user.
Example:
Top 5 recommended destinations for user 10: ['Goa Beaches', 'Leh Ladakh', 'Kerala Backwaters', 'Jaipur City', 'Leh Ladakh']

📊 Visualizations

Rating distribution
Most popular destinations
User-specific recommendation charts with predicted scores
