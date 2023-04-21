from itertools import chain
import json
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder

def gen_features(words):

    features = []

    for i in range(len(words)):

        ft = {
            'word.length':len(words[i]),
            'word.lower()':words[i].lower(),
            'word.isupper()':int(words[i].isupper()),
            'word.istitle()':int(words[i].istitle()),
            'word.isdigit()':int(words[i].isdigit()),
            'word.isalnum()':int(words[i].isalnum()),
            'word.position':(i+1)/len(words),
        }

        if i>0:
            ft.update({
                '-1:word.lower()':words[i-1].lower(),
                '-1:word.isupper()':int(words[i-1].isupper()),
                '-1:word.istitle()':int(words[i-1].istitle()),
                '-1:word.isalnum()':int(words[i-1].isalnum()),

            })

        else:
            ft.update({
                '-1:word.lower()':"",
                '-1:word.isupper()':0,
                '-1:word.istitle()':0,
                '-1:word.isalnum()':0,
            })

        if i< len(words)-1:

            ft.update({
                '+1:word.lower()':words[i+1].lower(),
                '+1:word.isupper()':int(words[i+1].isupper()),
                '+1:word.istitle()':int(words[i+1].istitle()),
                '+1:word.isalnum()':int(words[i+1].isalnum()),

            })

        else:
            ft.update({
                '+1:word.lower()':"",
                '+1:word.isupper()':0,
                '+1:word.istitle()':0,
                '+1:word.isalnum()':0,
            })

        features.append(ft)

    return features

def words2features(words):

    features = []

    for i in range(len(words)):
        features += gen_features(words[i])

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

def get_reverse_labels(filename):
    with open(filename,"r",encoding="utf8") as file:
        y_count = json.loads(file.read())
    
    output = []
    for label in y_count.keys():
        output.append(label)

    return output

def gen_xy_train(filename,lang):

    words = get_words(filename)

    words_train, y = process_tagging(words)

    x = words2features(words_train)
    
    y = list(chain.from_iterable(y))

    labels = get_labels(lang+"/count_y.json")

    print(labels)

    for i in range(len(y)):
        y[i]=labels[y[i]]

    y = pd.DataFrame(y)

    word_columns = ["word.lower()","-1:word.lower()","+1:word.lower()"]

    x_words = x[word_columns]
    
    x_process = x.drop(word_columns, axis=1)

    enc = OneHotEncoder(handle_unknown='ignore')

    enc.fit(x_words)

    x_words = pd.DataFrame(enc.transform(x_words).toarray())

    x = pd.concat([x_process, x_words],axis=1)

    return x,y,enc

def gen_x_test(filename, enc, lang):

    words_train = get_words(filename)

    # words_train, y = process_tagging(words)

    x = words2features(words_train)

    word_columns = ["word.lower()","-1:word.lower()","+1:word.lower()"]

    x_words = x[word_columns]
    
    x_process = x.drop(word_columns, axis=1)

    x_words = pd.DataFrame(enc.transform(x_words).toarray())

    x = pd.concat([x_process, x_words],axis=1)

    return x

def get_words(filename):

    raw = read_data(filename)

    sentences = sentence_cutting(raw)[:-1]

    words = word_cutting(sentences)

    return words

def train_and_predict(lang):

    print("hello world")

    #Training

    x_train, y_train, enc = gen_xy_train(lang+"/train",lang)

    print(x_train)

    clf = xgb.XGBClassifier(tree_method="gpu_hist")
    clf.fit(x_train,y_train)

    clf.save_model(lang+"/categorical-model3.json")

    #Testing

    x_test = gen_x_test(lang+"/test.in", enc, lang)

    clf = xgb.XGBClassifier(tree_method="gpu_hist")
    clf.load_model(lang+"/categorical-model3.json")

    y_pred = clf.predict(x_test)

    # print(y_pred)

    x_words = get_words(lang+"/test.in")
    labels = get_reverse_labels(lang+"/count_y.json")
    
    counter = 0
    with open(lang+"/test.p4.out", "w",encoding="utf8") as file:
        for i in range(len(x_words)):
            for j in range(len(x_words[i])):
                file.write(x_words[i][j]+" "+labels[y_pred[counter]]+"\n")
                counter+=1
            file.write("\n")


if __name__=="__main__":
    train_and_predict("EN")
