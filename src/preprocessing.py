import pandas as pd
import numpy as np
import re
from google_play_scraper import search
import time

def clean_app_name(name):
    name=name.strip().strip('"').strip("'")
    name=re.sub(r'^[+./\-\s]+','',name)
    return name
def preprocessing(df):
    #Handling outlier of Rating
    df.loc[(df["Rating"]<0.0) | (df["Rating"]>5.0),"Rating"]=np.nan
    print(df["Rating"].max())

    #Handling missing value of Ratings
    df["Rating"]=df["Rating"].fillna(df.groupby('Category')["Rating"].transform('median'))
    df=df.dropna(subset=["Rating"])

    #Dropping irrelevant columns
    df=df.drop(columns=["Current Ver","Android Ver","Type"],errors="ignore")

    #Handling Installs column by removing + and converting it to numeric
    df["Installs"]=df["Installs"].str.replace(r"[+,]","",regex=True)
    df["Installs"]=pd.to_numeric(df["Installs"], errors="coerce")

    #Handling Price column by removing $ and converting to numeric
    df["Price"]=df["Price"].str.replace("$","",regex=False)
    df["Price"]=pd.to_numeric(df["Price"],errors="coerce")
    #Converting Reviews column to numeric
    df["Reviews"]=pd.to_numeric(df["Reviews"],errors="coerce")

    #Filling missing values of Content Rating
    df["Content Rating"]=df["Content Rating"].fillna(df["Content Rating"].mode()[0])

    #Handling Size column by converting them to numeric and filling missing values
    df.loc[(df["Size"]=="Varies with device"),"Size"]=np.nan
    is_k=df["Size"].str.contains("k", na=False)
    is_m=df["Size"].str.contains("M", na=False)
    df.loc[is_k, "Size"] = df.loc[is_k, "Size"].str.replace("k", "").astype(float) / 1024
    df.loc[is_m, "Size"] = df.loc[is_m, "Size"].str.replace("M", "").astype(float)
    df["Size"] = df["Size"].astype(float)
    df["Size"]=df["Size"].fillna(df.groupby('Category')["Size"].transform('median'))

    #Filtering out junk apps that has less than 100 installs
    df=df[df["Installs"]>=100]
    df=df[df["Rating"].notna()]

    #Cleaning App names such as .R
    df['App']=df["App"].apply(clean_app_name)

    #Removing apps that has length less than or equal to 1 and duplicate apps
    df=df[df["App"].str.len() >1]
    df=df.sort_values('Reviews',ascending=False).drop_duplicates(subset="App")
    #Feature engineering for recommendation system
    df["text_features"]=df["Category"]+ " "+ df["Genres"]

    #Calculating Popularity Score using normalized Installs count and Rating
    df["norm_installs"]=(df["Installs"]-df["Installs"].min())/(df["Installs"].max()-df["Installs"].min())
    df["norm_rating"]=(df["Rating"]-df["Rating"].min())/(df["Rating"].max()-df["Rating"].min())
    df["popularity_score"]=(0.7*df["norm_installs"])+(0.3*df["norm_rating"])
    df.drop(columns=["norm_installs","norm_rating"],inplace=True)
    return df

def add_image(df):
    image_urls=[]
    #Fetching images for top 1000 apps
    top_df = df.sort_values(by="popularity_score", ascending=False).head(1000).copy()
    for i, app in enumerate(top_df["App"]):
        try:
            result=search(app,n_hits=1)
            if result:
                image_urls.append(result[0]['icon'])
            else:
                image_urls.append(None)
        except:
            image_urls.append(None)
        #Number of app image loaded
        if i%50==0:
            print(f"{i} apps processed")
        time.sleep(0.2)
    top_df["Image_URL"]=image_urls

    #Merging back with original data
    df = df.merge(top_df[["App", "Image_URL"]], on="App", how="left")

    #Filling the apps containing no images with placeholder
    df["Image_URL"].fillna("https://via.placeholder.com/80", inplace=True)
    return df
if __name__=="__main__":
    df=pd.read_csv("../data/googleplaystore.csv")
    print(f"Shape of the dataset before preprocessing {df.shape}")
    df= preprocessing(df)
    df=add_image(df)
    df.to_csv("../data/final_dataset.csv")
    print(f"Shape of the dataset after preprocessing: {df.shape}")
