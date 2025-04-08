# data_loader.py
import nltk
from nltk.corpus import movie_reviews
import pandas as pd
import random

def load_movie_data():
    nltk.download('movie_reviews')

    documents = [(movie_reviews.raw(fileid), category)
                 for category in movie_reviews.categories()
                 for fileid in movie_reviews.fileids(category)]
    random.shuffle(documents)

    df = pd.DataFrame(documents, columns=['review', 'label'])
    return df