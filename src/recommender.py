import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from thefuzz import process
import pickle

df=pd.read_csv("data/preprocessed_data.csv")
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
    sim_scores=sorted(sim_scores, key=lambda x: x[1],reverse=True)
    sim_scores=sim_scores[1:11]
    app_indices=[i[0] for i in sim_scores]
    return df.iloc[app_indices][["App","Rating","Category","Size","Installs","Price"]]

model_data={
    "tfidf":tfidf,
    "tfidf_matrix":tfidf_matrix,
    "cosine_sim":cosine_sim,
    "df":df
}
with open("model/model.pkl","wb") as f:
    pickle.dump(model_data,f)
print("Model saved as model.pkl")