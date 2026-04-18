import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from thefuzz import process
import joblib

df=pd.read_csv("../data/final_dataset.csv")
tfidf=TfidfVectorizer(stop_words="english")
tfidf_matrix=tfidf.fit_transform(df["text_features"])
cosine_sim=linear_kernel(tfidf_matrix,tfidf_matrix)

def recommendations(title, cosine_sim=cosine_sim):
    all_titles = df['App'].tolist()
    closest_match = process.extractOne(title, all_titles)[0]
    result=df[df["App"]==closest_match]
    if result.empty:
        return "Not found"
    else:
        idx=result.index[0]
    sim_scores=list(enumerate(cosine_sim[idx]))
    sim_scores=sorted(sim_scores, key=lambda x: (x[1],df.iloc[x[0]]["popularity_score"]),reverse=True)
    sim_scores=sim_scores[1:11]
    app_indices=[i[0] for i in sim_scores]
    return df.iloc[app_indices][["App","Rating","Category","Size","Installs","Price","Image_URL"]]

model_data={
    "tfidf":tfidf,
    "tfidf_matrix":tfidf_matrix,
    "cosine_sim":cosine_sim,
    "df":df
}
joblib.dump(model_data, "../model/model.pkl")
print("Model saved as model.pkl")