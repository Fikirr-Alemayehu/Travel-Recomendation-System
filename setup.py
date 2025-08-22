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

