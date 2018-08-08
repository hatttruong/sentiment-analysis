# Sentiment Analysis in Vietnamese #

## 1. Setup enrironment

```
pip3 install -r requirements.txt

```

## 2. EDA
- length of documents: number of words, number of sentences
- 
## 3. Preprocess
Steps to clean data
- remove non-character
- detect language
    + Vietnamese
    + English
- correct spelling
- accronym
- tokenize & NER
- extract: noun, adjective, verb, adverb,
- detect domain of document: "It has been shown that sentiment classifcation is highly sensitive to the domain from which the training data is extracted."

### Pyvi performs tokenizing / pos-tagging for Vietnamese in Python.
A - Adjective
C - Coordinating conjunction
E - Preposition
I - Interjection
L - Determiner
M - Numeral
N - Common noun
Nc - Noun Classifier
Ny - Noun abbreviation
Np - Proper noun
Nu - Unit noun
P - Pronoun
R - Adverb
S - Subordinating conjunction
T - Auxiliary, modal words
V - Verb
X - Unknown
F - Filtered out (punctuation)

## Feature Extractions
- **min_count = 5, not removing Np**

    ```
    tfidf:
       number of tokens before filtering: 100279
       keeping 22129 tokens which were in no less than 5 and no more than 35851 (=50.0%) documents
       number of tokens after filtering: 22129


    skipgram:
        training on a 21452335 raw words (11875788 effective words) took 111.6s, 106389 effective words/s
        storing 24005x100 projection weights into model/skipgram_min_count_5.vec

    ```


- **min_count = 5, remove Np**
    ```
    tfidf:
         keeping 14592 tokens which were in no less than 5 and no more than 35851 (=50.0%) documents
         number of tokens after filtering: 14592

          number of tokens before filtering: 58560
          keeping 14589 tokens which were in no less than 5 and no more than 35851 (=50.0%) documents
          number of tokens after filtering: 14589

    skipgram:
        training on a 20046845 raw words (9990582 effective words) took 92.3s, 108233 effective words/s
        storing 15513x100 projection weights into model/skipgram_min_count_5.vec

        training on a 20045025 raw words (9987270 effective words) took 90.5s, 110301 effective words/s
        storing 15510x100 projection weights into model/skipgram_min_count_5.vec

    ```

- **min_count = 1**
    ```
    tfidf:  keeping 16993 tokens which were in no less than 5 and no more than 36044 (=50.0%) documents
    skipgram: 
    ```

## 3. Train model & evaluation
### 3.1 Gaussian Mixture Model


- When I use w2v x tfidf, then normalize. Here the best result with Gausian Mixture Model

    ```
    2018-06-23 16:41:46,672 : INFO : Cov_type: TIED
    2018-06-23 16:41:58,076 : INFO : TRAIN RESULT:
    2018-06-23 16:41:58,344 : INFO : Accuracy: 60.3
    2018-06-23 16:41:58,403 : INFO : Confusion_matrix:
    [[18560   764  2211]
     [ 8840 12664  6205]
     [ 7568  3042 12286]]
    2018-06-23 16:41:58,404 : INFO : TEST RESULT:
    2018-06-23 16:41:58,494 : INFO : Accuracy: 60.7
    2018-06-23 16:41:58,519 : INFO : Confusion_matrix:
    [[7977  381  905]
     [3630 5346 2602]
     [3205 1256 5168]]
    ```

- Use only w2v (sum then normalize):

    ```
    2018-06-25 01:24:57,593 : INFO : Cov_type: TIED
    2018-06-25 01:25:08,743 : INFO : TRAIN RESULT:
    2018-06-25 01:25:09,052 : INFO : Accuracy: 50.0
    2018-06-25 01:25:09,108 : INFO : Confusion_matrix:
    [[19541  1972    22]
    [13731 13719   259]
    [17525  2582  2789]]
    2018-06-25 01:25:09,108 : INFO : TEST RESULT:
    2018-06-25 01:25:09,196 : INFO : Accuracy: 50.4
    2018-06-25 01:25:09,220 : INFO : Confusion_matrix:
    [[8409  852    2]
    [5772 5713   93]
    [7340 1065 1224]]
    ```

### 3.2 Logistic Regression

- Feature extraction: using **word2vec only**
    + It takes **1 hour 10mins** to conduct grid search for one type of score
    + Best resutl for "precision":
        + Patameters:
        ```
        'solver': 'newton-cg', 'multi_class': 'multinomial', 'class_weight': None, 'penalty': 'l2'
        ```

        + Mean precision:  0.740 (+/-0.005)
        + Confusion matrix (for Test set):
        ```
        [[7441  993  829]
         [1181 8326 2071]
         [1083 1788 6758]]
        ```
    + Best result for "recall":
        + Parameters: 
        ```
        'solver': 'newton-cg', 'multi_class': 'multinomial', 'class_weight': 'balanced', 'penalty': 'l2
        ```
        + Mean Recall: 0.745 (+/-0.006)
        + Confusion matrix (for Test set):
        ```
        [[7633  793  837]
         [1338 7871 2369]
         [1186 1431 7012]]
        ```

- Feature extraction: using **word2vec + tfidf**

### 3.3 SVM

**TOO MUCH TIME TO TRAIN**

### 3.4 RandomForest

- Feature extraction: using **word2vec only**
    + It takes **3 hours** to conduct grid search for one type of score
    - Best resutl for "precision":
        + Patameters:
        ```
        'max_features': 'auto', 'n_estimators': 100
        ```

        + Mean precision:  0.773 (+/-0.003)
        + Confusion matrix (for Test set):
        ```
        [[7758  897  608]
         [1098 8822 1658]
         [ 928 1814 6887]]
         
         [[7750  886  614]
         [1087 8985 1682]
         [ 863 1866 7098]]
        ```

    - Best result for "recall":
        + Parameters: 
        ```
        {'max_features': 'log2', 'n_estimators': 100}
        ```
        + Mean Recall: 0.773 (+/-0.006)
        + Confusion matrix (for Test set):
        ```
        [[7874  801  588]
         [1153 8794 1631]
         [ 933 1827 6869]]
        ```
- Feature extraction: using **word2vec + tfidf**
    
