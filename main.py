import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

# Load dataset
df = pd.read_csv("spotify_millsongdata.csv")

# Data preprocessing and cleaning
df = df.sample(5000).drop("link", axis=1).reset_index(drop=True)
df["text"] = (
    df["text"]
    .str.lower()
    .replace(r"[^a-zA-Z0-9 ]", " ", regex=True)
    .replace(r"\s+", " ", regex=True)
    .str.strip()
)

# Tokenization and stemming
stemmer = PorterStemmer()


def token(txt):
    tokens = nltk.word_tokenize(txt)
    return " ".join([stemmer.stem(word) for word in tokens])


df["text"] = df["text"].apply(token)

# TF-IDF and cosine similarity computation
tfid = TfidfVectorizer(analyzer="word", stop_words="english")
matrix = tfid.fit_transform(df["text"])
sim = cosine_similarity(matrix)


# Define and test recommendation function
def recommend(song_name):
    if song_name not in df["song"].values:
        return "Song not found in the dataset."
    idx = df[df["song"] == song_name].index[0]
    similar = list(enumerate(sim[idx]))
    return [
        df.iloc[s_id[0]]["song"]
        for s_id in sorted(similar, key=lambda x: x[1], reverse=True)[1:6]
    ]


# Serialize the DataFrame and similarity matrix
output_dir = os.getcwd()  # or specify another directory
df_path = os.path.join(output_dir, "df.pkl")
sim_path = os.path.join(output_dir, "similarity.pkl")

try:
    with open(df_path, "wb") as file:
        pickle.dump(df, file)
    print(f"DataFrame serialized successfully to {df_path}")

    with open(sim_path, "wb") as file:
        pickle.dump(sim, file)
    print(f"Similarity matrix serialized successfully to {sim_path}")
except Exception as e:
    print(f"Serialization failed: {e}")

# Verify serialization by loading the files again
try:
    with open(df_path, "rb") as file:
        loaded_df = pickle.load(file)
    print("Loaded DataFrame:", loaded_df.shape)

    with open(sim_path, "rb") as file:
        loaded_sim = pickle.load(file)
    print("Loaded similarity matrix shape:", loaded_sim.shape)
except Exception as e:
    print(f"Failed to load serialized data: {e}")
