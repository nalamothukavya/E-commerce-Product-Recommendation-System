import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

products = pd.read_csv('data/products.csv')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(products['description'])

cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print('Similarity matrix shape:', cos_sim.shape)
