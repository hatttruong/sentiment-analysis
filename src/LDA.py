
# coding: utf-8

# In[41]:

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import string
import gensim
from pathlib import Path
import pandas as pd
import pprint

# Plotting tools
# import pyLDAvis
# import pyLDAvis.gensim  # don't skip this
# import matplotlib.pyplot as plt

# get_ipython().run_line_magic('matplotlib', 'inline')

pp = pprint.PrettyPrinter(indent=4)

# In[5]:


DATA_DIR = '../data/'
corpus_paths = [
    Path(DATA_DIR + 'Negative_train.csv.preprocessed'),
    Path(DATA_DIR + 'Neutral_train.csv.preprocessed'),
    Path(DATA_DIR + 'Positive_train.csv.preprocessed')]


# In[14]:


# load VN stopwords
stopword_path = Path(DATA_DIR + 'vietnamese-stopwords-dash.txt')
stopwords = []
with open(str(stopword_path)) as f:
    lines = f.readlines()
    stopwords = [l.strip() for l in lines]
# stopwords[:10]


# In[8]:


data = []
for corpus_path in corpus_paths:
    df = pd.read_csv(corpus_path, header=None, names=['text'])
    data.extend(df.text.tolist())


# In[27]:

special_tokens = ['email_token', 'url_token', 'number_token',
                  'phone_token', 'currency_token', 'datetime_token']
ignored_words = stopwords + list(string.punctuation) + special_tokens


def sent_to_words(sentences):
    for sentence in sentences:
        yield [w.strip() for w in sentence.split() if w not in ignored_words]


data_words = list(sent_to_words(data))

print('data_words[:1]:', data_words[:1])


# ### Creating Bigram and Trigram Models

# In[28]:


# Build the bigram and trigram models
# subset_data_words =
# higher threshold fewer phrases.
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
# trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
# trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
# print(trigram_mod[bigram_mod[data_words[0]]])


# In[29]:


def make_bigrams(texts, bigram_mod):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts, bigram_mod, trigram_mod):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


# In[30]:


data_words_bigrams = make_bigrams(data_words, bigram_mod)
print('data_words_bigrams[:1]: ', data_words_bigrams[:1])


# ### Create the Dictionary and Corpus needed for Topic Modeling

# In[31]:


# Create Dictionary
id2word = gensim.corpora.Dictionary(data_words_bigrams)

# Create Corpus
texts = data_words_bigrams

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
# print(corpus[:1])


# In[32]:


# Human readable format of corpus (term-frequency)
print('corpus[:1] (term-frequency): ',
      [[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])


# ### Building the Topic Model

# In[34]:


# Build LDA model
num_topics = 5
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=num_topics,
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            alpha='auto',
                                            per_word_topics=True)


# In[42]:


# Print the Keyword in the 10 topics
pp.pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]


# ###  Compute Model Perplexity and Coherence Score

# In[43]:


# Compute Perplexity
# a measure of how good the model is. lower the better.
print('\nPerplexity: ', lda_model.log_perplexity(corpus))

# Compute Coherence Score
coherence_model_lda = gensim.models.CoherenceModel(model=lda_model,
                                                   texts=data_words_bigrams,
                                                   dictionary=id2word,
                                                   coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


# ### How to find the optimal number of topics for LDA?

# In[52]:


def compute_coherence_values(dictionary, corpus, texts, limit,
                             start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with
    respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        print('num_topics: ', num_topics)
        # model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus,
        # num_topics=num_topics, id2word=id2word)
        model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=num_topics,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)
        model_list.append(model)
        coherencemodel = gensim.models.CoherenceModel(
            model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


# In[50]:


# Can take a long time to run.
model_list, coherence_values = compute_coherence_values(
    dictionary=id2word, corpus=corpus, texts=data_words_bigrams,
    start=2, limit=40, step=2)


# In[51]:


# Show graph
limit = 40
start = 2
step = 2
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
# plt.show()
plt.savefig('compute_coherence_values.png')


# ### Visualize the topics-keywords

# In[53]:


# # Visualize the topics
# pyLDAvis.enable_notebook()
# vis = pyLDAvis.gensim.prepare(model_list[0], corpus, id2word)
# vis
