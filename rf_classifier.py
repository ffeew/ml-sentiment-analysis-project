# from sklearn.ensemble import RandomForestClassifier
# from sklearn.pipeline import Pipeline
from sklearn import metrics
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.datasets import fetch_20newsgroups
# from sklearn_crfsuite import metrics

import numpy as np
from itertools import chain,permutations
import json
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
import match_word

words_list = match_word.get_words("EN/x_set_lower.json")

def gen_features(words,tags):

    features = []

    for i in range(len(words)):

        ft = {
            # 'bias':1.0,
            # 'word.length':len(words[i]),
            'word.lower()':words[i].lower(),#  if "[" not in words[i] and "]" not in words[i] and "<" not in words[i] else "" match_word.closest_word(words_list,words[i].lower()),
            'word.isupper()':int(words[i].isupper()),
            'word.istitle()':int(words[i].istitle()),
            'word.isdigit()':int(words[i].isdigit()),
            'word.isalnum()':int(words[i].isalnum()),
            'word.position':(i+1)/len(words),
            # 'sentence_length':len(words)
            # 'word_end':words[i].lower()[-4:]
        }

        if i>0:
            ft.update({
                '-1:word.lower()':words[i-1].lower(),#match_word.closest_word(words_list,words[i-1].lower()),
                '-1:word.isupper()':int(words[i-1].isupper()),
                '-1:word.istitle()':int(words[i-1].istitle()),
                '-1:word.isalnum()':int(words[i-1].isalnum()),
                # '-1:word.length':len(words[i-1]),
                # "-1:tag":tags[i-1],
                # '-1:tag:0':int(tags[i-1]=="O"),
                # '-1:tag:1':int(tags[i-1]=="B-INTJ"),
                # '-1:tag:2':int(tags[i-1]=="B-PP"),
                # '-1:tag:3':int(tags[i-1]=="B-NP"),
                # '-1:tag:4':int(tags[i-1]=="I-NP"),
                # '-1:tag:5':int(tags[i-1]=="B-VP"),
                # '-1:tag:6':int(tags[i-1]=="B-PRT"),
                # '-1:tag:7':int(tags[i-1]=="I-VP"),
                # '-1:tag:8':int(tags[i-1]=="B-ADJP"),
                # '-1:tag:9':int(tags[i-1]=="B-SBAR"),
                # '-1:tag:10':int(tags[i-1]=="B-ADVP"),
                # '-1:tag:11':int(tags[i-1]=="I-INTJ"),
                # '-1:tag:12':int(tags[i-1]=="B-CONJP"),
                # '-1:tag:13':int(tags[i-1]=="I-CONJP"),
                # '-1:tag:14':int(tags[i-1]=="I-ADVP"),
                # '-1:tag:15':int(tags[i-1]=="I-ADJP"),
                # '-1:tag:16':int(tags[i-1]=="I-SBAR"),
                # '-1:tag:17':int(tags[i-1]=="I-PP"),
                # 'BOS':0
            })

        else:
            ft.update({
                '-1:word.lower()':"",
                '-1:word.isupper()':0,
                '-1:word.istitle()':0,
                '-1:word.isalnum()':0,
                # '-1:tag:0':0,
                # '-1:tag:1':0,
                # '-1:tag:2':0,
                # '-1:tag:3':0,
                # '-1:tag:4':0,
                # '-1:tag:5':0,
                # '-1:tag:6':0,
                # '-1:tag:7':0,
                # '-1:tag:8':0,
                # '-1:tag:9':0,
                # '-1:tag:10':0,
                # '-1:tag:11':0,
                # '-1:tag:12':0,
                # '-1:tag:13':0,
                # '-1:tag:14':0,
                # '-1:tag:15':0,
                # '-1:tag:16':0,
                # '-1:tag:17':0,
                # "-1:tag":"",
                # 'BOS':1
            })

        if i< len(words)-1:

            ft.update({
                '+1:word.lower()':words[i+1].lower(),#match_word.closest_word(words_list,words[i+1].lower()),
                '+1:word.isupper()':int(words[i+1].isupper()),
                '+1:word.istitle()':int(words[i+1].istitle()),
                '+1:word.isalnum()':int(words[i+1].isalnum()),
                # '+1:tag':tags[i+1],
                # 'EOS':0
            })

        else:
            ft.update({
                '+1:word.lower()':"",
                '+1:word.isupper()':0,
                '+1:word.istitle()':0,
                '+1:word.isalnum()':0,
                # '+1:tag':"",
                # 'EOS':1
            })

        features.append(ft)

    return features

def words2features(words,labels):

    features = []

    for i in range(len(words)):
        features += gen_features(words[i],labels[i])

    return pd.DataFrame(features)

def read_data(filename):

    with open(filename, "r",encoding="utf8") as file:
        raw = file.read()
    
    return raw

def sentence_cutting(raw):

    s = raw.split("\n\n")

    return s

def word_cutting(sentences):
    words = []
    for sentence in sentences:
        words.append(sentence.split("\n"))
    
    return words

def process_tagging(words):
    x = []
    y = []
    for sentence in words:
        x.append([])
        y.append([])
        for word in sentence:
            x[-1].append(word.split()[0])
            y[-1].append(word.split()[1])
    
    return x,y

def get_labels(filename):
    with open(filename,"r",encoding="utf8") as file:
        y_count = json.loads(file.read())
    
    for i,label in enumerate(y_count.keys()):
        y_count[label] = i

    return y_count

def gen_xy_train(filename):

    words = get_words(filename)

    words_train, y = process_tagging(words)

    x = words2features(words_train,y)
    
    y = list(chain.from_iterable(y))

    labels = get_labels("EN/count_y.json")

    print(labels)

    for i in range(len(y)):
        y[i]=labels[y[i]]

    y = pd.DataFrame(y)

    word_columns = ["word.lower()","+1:word.lower()","-1:word.lower()"]

    x_words = x[word_columns]

    x_words = pd.get_dummies(x_words,columns = word_columns)
    
    x_process = x.drop(word_columns, axis=1)

    # x = pd.concat([x_process, x_words],axis=1)

    # y = pd.get_dummies(y,prefix="",prefix_sep="")

    enc = OneHotEncoder(handle_unknown='ignore')

    enc.fit(x_words)

    x_words = pd.DataFrame(enc.transform(x_words).toarray())

    x = pd.concat([x_process, x_words],axis=1)

    return x,y,enc

def gen_xy_test(filename,enc):

    words = get_words(filename)

    words_train, y = process_tagging(words)

    x = words2features(words_train)

    labels = get_labels("EN/count_y.json")

    for i in range(len(y)):
        y[i]=labels[y[i]]

    y = pd.DataFrame(y)

    word_columns = ["word.lower()","+1:word.lower()","-1:word.lower()"]

    x_words = x[word_columns]

    x_words = pd.get_dummies(x_words,columns = word_columns)
    
    x_process = x.drop(word_columns, axis=1)

    # x = pd.concat([x_process, x_words],axis=1)

    # y = pd.get_dummies(y,prefix="",prefix_sep="")

    x_words = pd.DataFrame(enc.transform(x_words).toarray())

    x = pd.concat([x_process, x_words],axis=1)

    return x,y

def get_words(filename):

    raw = read_data(filename)

    sentences = sentence_cutting(raw)[:-1]

    words = word_cutting(sentences)

    return words

if __name__=="__main__":
    
    #{'O': 0, 'B-INTJ': 1, 'B-PP': 2, 'B-NP': 3, 'I-NP': 4, 'B-VP': 5, 'B-PRT': 6, 'I-VP': 7, 'B-ADJP': 8, 'B-SBAR': 9, 'B-ADVP': 10, 'I-INTJ': 11, 'B-CONJP': 12, 'I-CONJP': 13, 'I-ADVP': 14, 'I-ADJP': 15, 'I-SBAR': 16, 'I-PP': 17}

    print("hello world")

    #Training

    x_train, y_train, enc = gen_xy_train("EN/train")

    # x_train["word.lower()"] = x_train["word.lower()"].astype('category')
    # x_train["-1:word.lower()"] = x_train["-1:word.lower()"].astype("category")
    # x_train["+1:word.lower()"] = x_train["+1:word.lower()"].astype("category")
    # x_train["-1:tag"] = x_train["-1:tag"].astype("category")
    # x_train["+1:tag"] = x_train["+1:tag"].astype("category")

    # print(x_train)
    print(x_train)

    clf = xgb.XGBClassifier(tree_method="gpu_hist")
    clf.fit(x_train,y_train)

    clf.save_model("EN/categorical-model3.json")

    #Testing

    x_test,y_test = gen_xy_test("EN/dev.out")

    # x_test["word.lower()"] = x_test["word.lower()"].astype('category')
    # x_test["-1:word.lower()"] = x_test["-1:word.lower()"].astype("category")
    # x_test["+1:word.lower()"] = x_test["+1:word.lower()"].astype("category")
    # x_test["-1:tag"] = x_test["-1:tag"].astype("category")
    # x_test["word_end"] = x_test["word_end"].astype("category")

    clf = xgb.XGBClassifier(tree_method="gpu_hist", enable_categorical=True)
    clf.load_model("EN/categorical-model3.json")

    # with pd.option_context("display.max_columns", None):
    #     print(x_test)
    print(x_test)

    y_pred = clf.predict(x_test)

    y_pred = pd.DataFrame(y_pred)

    print(y_pred)

    print(metrics.f1_score(y_test, y_pred,average='weighted',labels=list(range(18))))


# text_clf = Pipeline([('vect', CountVectorizer()),
#                      ('tfidf', TfidfTransformer()),
#                      ('clf', RandomForestClassifier(n_estimators=100)),
#                      ])

# text_clf.fit(X_train, y_train)


# predicted = text_clf.predict(X_test)

# print(metrics.classification_report(y_test, predicted))