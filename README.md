# App-Recommendation-System
An App Recommendation System using TF-IDF and Cosine Similarity on Google Play Store Data

## Overview
This project recommends similar apps based on their category and genre features using TF-IDF vectorization and cosine similarity.  

To improve real-world relevance, a hybrid ranking strategy is used by combining:
- Content similarity (TF-IDF + cosine similarity)
- App popularity (based on installs and ratings)

The final system allows users to select an app and instantly discover similar, high-quality alternatives with visual previews.

## Features
- **Data Preprocessing Pipeline**
  - Handles missing values, inconsistent formats, and duplicates
  - Cleans installs, price, and size columns
  - Filters out low-quality and low-popularity apps

- **Content-Based Recommendation Engine**
  - TF-IDF vectorization on app metadata (Category + Genres)
  - Cosine similarity for measuring app similarity

- **Hybrid Ranking System**
  - Combines similarity with a **popularity score**
  - Popularity score derived from installs and ratings

- **App Icon Integration**
  - Fetches app icons dynamically using Google Play Scraper
  - Enhances user experience with visual recommendations

- **Interactive Streamlit UI**
  - Dropdown-based app selection
  - Displays top 10 recommended apps
  - Clean grid layout with images and metadata 

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn (TF-IDF, Cosine Similarity)
- Streamlit
- Joblib
- TheFuzz (Fuzzy Matching)
- Google Play Scraper


## Project Structure
```
App-Recommendation-System/
│
├── data/
│ └── final_dataset.csv
│
├── model/
│ └── model.pkl
│
├── src/
│ ├── preprocessing.py
│ ├── recommendation.py # Model building
│ └── recommender.py # Recommendation logic
│
├── app.py # Streamlit UI
├── requirements.txt
└── README.md
```
## How It Works

1. **Preprocessing**
   - Clean dataset and engineer features (`text_features`)
   - Compute popularity score

2. **Model Building**
   - Apply TF-IDF on text features
   - Compute cosine similarity matrix
   - Save model using `joblib`.

3. **Recommendation**
   - User selects an app
   - System finds similar apps using cosine similarity
   - Results are re-ranked using the popularity score

4. **UI**
   - Streamlit displays recommendations with images and details


## Running the App

### 1. Clone the repository
```bash
git clone https://github.com/Snehapaul20/App-Recommendation-System.git
cd App-Recommendation-System
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

## Future Improvements
- Replace full cosine matrix with on-demand similarity computation (memory optimization)
- Build a FastAPI backend + React frontend
- Add search autocomplete and filtering
