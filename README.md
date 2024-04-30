
# Spotify Song Recommender

This project contains a Python script that processes song data from Spotify, performs text cleaning, and applies natural language processing techniques to recommend songs based on textual similarity.

## Features

- Data loading and preprocessing
- Text cleaning and normalization
- Tokenization and stemming
- TF-IDF vectorization
- Cosine similarity calculation for song recommendations

## Requirements

To run this script, you will need Python installed along with the following libraries:
- Pandas
- NLTK
- Scikit-learn
- Pickle

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/turman17/music_recommender.git
   ```

2. **Navigate to the directory:**
   ```bash
   cd spotify-song-recommender
   ```

3. **Install required Python packages:**
   ```bash
   pip install pandas scikit-learn nltk
   ```

4. **Download NLTK tokenizers (if not already done):**
   ```python
   import nltk
   nltk.download('punkt')
   ```

## Usage

Run the script using the following command:

```bash
python spotify_recommender.py
```

## Example

Here is how you can get song recommendations:

```python
from spotify_recommender import recommend
print(recommend("Breakaway"))
```

## Data

The data used in this script should be a CSV file named `spotify_millsongdata.csv` with at least the following columns:
- `text`: Lyrics or description of the song.
- `song`: Name of the song.
