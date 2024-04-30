# Import necessary libraries
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load dataset
df = pd.read_csv("spotify_millsongdata.csv")

# Display the first and last few rows of the dataset
print(df.head(5))
print(df.tail(5))

# Display the shape and null values in the dataset
print(df.shape)
print(df.isnull().sum())

# Sample and clean the data
# df = df.drop('link', axis=1).reset_index(drop=True) # to use not sampled data
df = df.sample(5000).drop('link', axis=1).reset_index(drop=True)
print(df.head(5))
print(df['text'][0])
print(df.shape)

# Text cleaning
df['text'] = df['text'].str.lower().replace(r'[^a-zA-Z0-9 ]', ' ', regex=True).replace(r'\s+', ' ',
                                                                                       regex=True).str.strip()
print(df.tail(5))

# Define a tokenization and stemming function
stemmer = PorterStemmer()


def token(txt):
    tokens = nltk.word_tokenize(txt)
    stemmed_words = [stemmer.stem(word) for word in tokens]
    return " ".join(stemmed_words)


# Test the tokenizer and stemmer
print(token("you are beautiful"))

# Apply text processing to the dataframe
df['text'] = df['text'].apply(token)

# Setup and compute TF-IDF matrix
tfid = TfidfVectorizer(analyzer='word', stop_words='english')
matrix = tfid.fit_transform(df['text'])

# Compute cosine similarity
sim = cosine_similarity(matrix)
print(sim[0])


# Define a function to recommend songs based on similarity
def recommend(song_name):
    if song_name not in df['song'].values:
        return "Song not found in the dataset."

    idx = df[df['song'] == song_name].index[0]
    similar = list(enumerate(sim[idx]))
    sorted_similar = sorted(similar, key=lambda x: x[1], reverse=True)

    song_recommendations = []
    for s_id in sorted_similar[1:5]:  # Ensuring only existing indices are accessed
        song_recommendations.append(df.iloc[s_id[0]]['song'])

    return song_recommendations


# Test the recommendation function
print(recommend("Breakaway"))

# Serialize the similarity matrix and dataframe for later use
pickle.dump(sim, open("similarity", "wb"))
pickle.dump(df, open("df", "wb"))
