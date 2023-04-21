# from itertools import chain

# import nltk
# import sklearn
# import scipy.stats
# from scipy.optimize import minimize
# from sklearn.metrics import make_scorer
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import RandomizedSearchCV

# import sklearn_crfsuite
# from sklearn_crfsuite import scorers
# from sklearn_crfsuite import metrics

import numpy as np
from itertools import permutations
import json

def gen_features(words):

    features = []

    for i in range(len(words)):

        ft = {
            # 'bias':1.0,
            # 'word.length':len(words[i]),
            'word.lower()':words[i].lower(),
            'word.isupper()':words[i].isupper(),
            'word.istitle()':words[i].istitle(),
            # 'word.isdigit()':words[i].isdigit(),
            'word.isalnum()':words[i].isalnum(),
            # 'word[-4:]':words[i][-4:]
        }

        if i>0:

            ft.update({
                '-1:word.lower()':words[i-1].lower(),
                '-1:word.isupper()':words[i-1].isupper(),
                '-1:word.istitle()':words[i-1].istitle(),
                '-1:word.isalnum()':words[i-1].isalnum(),
            })

        else:
            ft["BOS"] = True

        if i< len(words)-1:

            ft.update({
                '+1:word.lower()':words[i+1].lower(),
                '+1:word.isupper()':words[i+1].isupper(),
                '+1:word.istitle()':words[i+1].istitle(),
                '+1:word.isalnum()':words[i+1].isalnum(),
            })

        else:
            ft["EOS"] = True

        features.append(ft)

    return features

def gen_features2(words,tags):

    features = []

    for i in range(len(words)):

        ft = {
            'bias':1.0,
            'word.length':len(words[i]),
            'word.lower()':words[i].lower(),
            'word.isupper()':words[i].isupper(),
            'word.istitle()':words[i].istitle(),
            'word.isdigit()':words[i].isdigit(),
            'word.isalnum()':words[i].isalnum(),
            'word[-2:]':words[i][-2:]
        }

        if i>0:

            ft.update({
                '-1:word.lower()':words[i-1].lower(),
                '-1:word.isupper()':words[i-1].isupper(),
                '-1:word.istitle()':words[i-1].istitle(),
                '-1:word.isdigit()':words[i-1].isdigit(),
                '-1:tag':tags[i-1]
            })

        else:
            ft["BOS"] = True

        if i< len(words)-1:

            ft.update({
                '+1:word.lower()':words[i+1].lower(),
                '+1:word.isupper()':words[i+1].isupper(),
                '+1:word.istitle()':words[i+1].istitle(),
                '+1:word.isdigit()':words[i+1].isdigit(),
                '+1:tag':tags[i+1]
            })

        else:
            ft["EOS"] = True

        features.append(ft)

    return features

def words2features(words):

    return [gen_features(sentence) for sentence in words]

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

def gen_xy(filename):

    words = get_words(filename)

    words_train, y_train = process_tagging(words)

    x_train = words2features(words_train)

    return x_train,y_train

def get_words(filename):

    raw = read_data(filename)

    sentences = sentence_cutting(raw)[:-1]

    words = word_cutting(sentences)

    return words

# def cost(p):

#     crf = sklearn_crfsuite.CRF(
#         algorithm='lbfgs',
#         c1=p[0],
#         c2=p[1],
#         max_iterations=100,
#         all_possible_transitions=True
#     )
#     crf.fit(x_train, y_train)
    
#     labels = list(crf.classes_)

#     labels.remove('O')

#     x_test,y_test = gen_xy("EN/dev.out")

#     y_pred = crf.predict(x_test)

#     print(metrics.flat_f1_score(y_test, y_pred,average='weighted',labels=labels))

#     return 1-metrics.flat_f1_score(y_test, y_pred,average='weighted',labels=labels)

feature_functions = []
weights = np.array([])
x_train, y_train = gen_xy("EN/train")
random_labels = dict()
pos_labels = []

def load_y(filename):
    y_count = dict()
    with open(filename,"r",encoding="utf8") as file:
        y_count = json.loads(file.read())
    return list(y_count.keys())

def generate_random_labels(length):

    labels = []
    for i in range(100):
        labels.append(np.random.choice(pos_labels,length))
    random_labels[length]=labels

def gen_feature_f(x_train,y_train):
    features2get = [
        ["BOS","tag"],
        ["EOS","tag"],
        ["-1:tag","tag","+1:tag"],
        ["word.lower()","word.isalnum()","word.istitle()","word.isupper()","tag"],
        ["-1:word.lower()","+1:word.lower()","word.lower()","tag"],
        ["-1:word.isalnum()","word.isalnum()","tag","+1:word.isalnum()"],
        ["-1:word.istitle()","word.istitle()","tag","+1:word.istitle()"],
        ["-1:word.isupper()","word.isupper()","tag","+1:word.isupper()"]
    ]

    for i in range(len(x_train)):
        for k in range(len(x_train[i])):
            for j in range(len(features2get)):
                dictionary = dict()
                for feature in features2get[j]:
                    if "tag" not in feature:
                        dictionary[feature] = x_train[i][k].get(feature)
                    else:
                        if feature=="tag":
                            dictionary[feature]=y_train[i][k]
                        elif feature=="-1:tag":
                            dictionary[feature]=y_train[i][k-1] if k>0 else None
                        elif feature=="+1;tag":
                            dictionary[feature]=y_train[i][k+1] if k<len(x_train[i])-1 else None 
                feature_functions.append(dictionary)

    # with open("EN/feature_functions.json","w",encoding="utf8") as y_file:
    #     for function in feature_functions:
    #         json.dump(function,y_file, indent = 4)
    with open("EN/features_functions.npy","wb") as file:
        np.save(file,feature_functions)

def feature_f(labels, sentence_features, j, i):

    #Accepts sentence features as a list of dictionary of word features and j as the index of the feature function

    # for feature in feature_functions[j].keys():
    #     if not sentence_features[i].get(feature) == feature_functions[j].get(feature):
    #         return 0
        # if "tag" not in feature:
            
        # else:
        #     if feature == "tag" and not labels[i]==feature_functions[j][feature]:
        #         return 0
        #     elif feature=="-1:tag" and (i==0 or not labels[i-1]==feature_functions[j][feature]):
        #         return 0
        #     elif feature=="+1:tag" and (i==len(labels) or not labels[i+1]==feature_functions[j][feature]):
        #         return 0
    if all(sentence_features[i].get(feature) == feature_functions[j].get(feature) for feature in feature_functions[j].keys()):
        print(sentence_features[i])

    return int(all(sentence_features[i].get(feature) == feature_functions[j].get(feature) for feature in feature_functions[j].keys()))

def p_sentence(sentence_features, weights, labels):

    m = len(feature_functions)
    n_score=0
    normalize = 0

    #Combining tags with x features
    for i in range(len(sentence_features)):
        sentence_features[i]["tag"] = labels[i]
        if i>0:
            sentence_features[i]["-1:tag"] = labels[i]
        if i<len(sentence_features[i])-1:
            sentence_features[i]["+1:tag"] = labels[i]

    for j in range(m):
        print(j,end="\r")
        for i in range(len(sentence_features)):
           n_score+=weights[j]*feature_f(labels, sentence_features, j, i)
    n_score = np.exp(n_score)

    if random_labels.get(len(labels))==None:
        generate_random_labels(len(labels))

    for any_labels in random_labels[len(labels)]+[labels]:

        #Combining tags with x features

        for i in range(len(sentence_features)):
            sentence_features[i]["tag"] = any_labels[i]
            if i>0:
                sentence_features[i]["-1:tag"] = any_labels[i]
            if i<len(sentence_features[i])-1:
                sentence_features[i]["+1:tag"] = any_labels[i]

        score=0
        for j in range(m):
            for i in range(len(sentence_features)):
                score+=weights[j]*feature_f(any_labels, sentence_features, j, i)
        normalize += np.exp(score)

    return score/normalize
    
def gradient_descent(sentence_features, alpha, epochs, labels):
    # print(labels)
    for n in range(epochs):

        if random_labels.get(len(labels))==None:
            generate_random_labels(len(labels))

        for i in range(len(feature_functions)):
            term1 = sum([feature_f(labels,sentence_features,j,i) for j in range(len(sentence_features))])
            term2 = sum([p_sentence(sentence_features, weights, any_label)*
                        sum([feature_f(any_label,sentence_features,j,i) for j in range(len(sentence_features))])
                        for any_label in random_labels[len(labels)]+[labels]])
            weights[i] += alpha*(term1-term2)

def training(x_train,y_train):

    #Combining tags with x features

    for i in range(len(x_train)):
        # gradient_descent(x_train[i],0.1,100,y_train[i])
        # p_sentence(x_train[i], weights, y_train[i])
        # for j in range(len(feature_functions)):
        #     feature_f(y_train[0],x_train[0],j,0)
        print(i,"/",len(x_train),"sentences completed", end="\r")

    with open("EN/features_weights.npy","wb") as file:
        np.save(file,weights)
    
    with open("EN/random_labels.npy","wb") as file:
        np.save(file,random_labels)


if __name__ == "__main__":
    print("hello world!")

    #feature function -> a dictionary of 

    #feature function -> whether a set of features and tags that matches what is observed.

    # print(words_train)
    # print(y_train)

    #print(x_train[:10])

    pos_labels = load_y("EN/count_y.json")
    
    try:

        with open("EN/features_functions.npy","rb") as file:
            feature_functions = np.load(file)
        print("Features functions loading completed")

    except:

        print("Error loading features functions")
        gen_feature_f(x_train,y_train)

    try:
        
        with open("EN/features_weights.npy","rb") as file:
            weights = np.load(file)

        print("Weights loading completed")

    except:
        print("Error loading weights")
        weights = np.zeros(len(feature_functions)) 

    training(x_train,y_train)

    ###Training###

    #Best c1 and c2 for FR: [0.09875066 0.09940136]
    #Best c1 and c2 for EN: [0.1075     0.09000002]

    #1e-8 error acceptable in convergence
    # res = minimize(cost, [0.1,0.1], method='nelder-mead', options={'xatol':1e-8,'disp':True})

    # print(res.x)

    # trained_features = gen_xy("EN/train")

    # crf = sklearn_crfsuite.CRF(
    #     algorithm='lbfgs',
    #     c1=res.x[0],
    #     c2=res.x[1],
    #     max_iterations=100,
    #     all_possible_transitions=True
    # )
    # crf.fit(x_train, y_train)

    # x_test,y_test = gen_xy("EN/dev.out")

    # y_pred = crf.predict(x_test)
    
    # words = get_words("EN/dev.in")

    # fileout = "EN/dev.crf.out"

    # with open(fileout, "w",encoding="utf8") as file:
    #     for i in range(len(words)):
    #         for j in range(len(words[i])):
    #             file.write(words[i][j]+" "+y_pred[i][j]+"\n")
    #         file.write("\n")