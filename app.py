import streamlit as st
import pandas as pd
import joblib
from src.recommender import recommendations
@st.cache_resource
def load_model():
    return joblib.load("model/model.pkl")
model=load_model()

df=model["df"]
cosine_sim=model["cosine_sim"]

st.set_page_config(page_title="App Recommender",layout="wide")
st.title("App Recommendation System")
app_list=sorted(df["App"].unique())
selected_app = st.selectbox("Choose an app", app_list)
if st.button("Recommend Apps"):
    results=recommendations(selected_app)
    if results is not None:
        st.subheader("Recommended Apps")
        cols=st.columns(5)
        for i, (_, row) in enumerate(results.iterrows()):
            with cols[i % 5]:
                st.image(row["Image_URL"], width=100)
                st.markdown(f"**{row['App']}**")
                st.caption(f"Rating: {row['Rating']}")
    else:
        st.error("No Recommendations found")

