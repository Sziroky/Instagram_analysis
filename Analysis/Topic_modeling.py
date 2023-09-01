import pandas as pd
import spacy
from gensim import corpora, models
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim as gensimvis
import os


file_path = input('Full path of proceed file for topic modeling:')
df = pd.read_csv(file_path)
nlp = spacy.load('pl_core_news_lg')
data = df['hashtags'].values.tolist()

def preprocess_text(text):
    return [word for word in simple_preprocess(text) if word not in STOPWORDS]

df['hashtags'] = df['hashtags'].apply(preprocess_text)

# Create a Gensim dictionary
dictionary = corpora.Dictionary(df['hashtags'])

# Create a corpus (list of Bag-of-Words representation)
corpus = [dictionary.doc2bow(text) for text in df['hashtags']]

# Train an LDA model
lda_model = models.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=15)

# Interpret topics
for idx, topic in lda_model.print_topics(-1):
    print(f"Topic: {idx}\nWords: {topic}\n")

# Assign topics to documents
topics = lda_model.get_document_topics(corpus)
df['topics'] = [max(topic, key=lambda x: x[1])[0] for topic in topics]

# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))
# a measure of how good the model is. lower the better.

# # Compute Coherence Score
# coherence_model_lda = CoherenceModel(model=lda_model, texts=df['hashtags'], dictionary=dictionary, coherence='c_v')
# coherence_lda = coherence_model_lda.get_coherence()
# print('\nCoherence Score: ', coherence_lda)

file_name = 'LDA_visualization.html'
file_path = os.path.join(os.getcwd(), file_name)
vis = gensimvis.prepare(lda_model, corpus, dictionary)
pyLDAvis.save_html(vis, file_path)
print(f'LDA finished -- check {file_path} for visualization ')



