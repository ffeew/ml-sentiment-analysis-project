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

x_train, y_train = gen_xy("EN/train")

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

feature_functions = [
    {}
]
weights = [] 

def gen_feature_f(feature_functions):
    pass #TODO

def feature_f(label, word_f,k):

    #Accepts word as a dicitonary and k as the index of the feature function

    if label!=feature_f[k]["tag"]:
        return 0

    for condition in feature_functions[k].keys():
        if not word_f[condition]==feature_functions[k][condition]:
            return 0

    return 1


def p_sentence(words_features, weights, labels):
    m = len(feature_functions)
    l_score=0
    for j in range(m):
        for i in range(len(words_features)):
            l_score+=weights[j]*feature_f(labels[i], words_features[i], j)
    l_score = np.exp(l_score)

    for any_labels in permutations(labels):
        score=0
        for j in range(m):
            for i in range(len(words_features)):
                score+=weights[j]*feature_f(any_labels[i], words_features[i], j)
        normalize += np.exp(score)

        
        return score/normalize
    
def gradient_descent(words_features, alpha, epochs,label):

    for i in range(epochs):
        for i in range(len(feature_functions)):
            term1 = sum([feature_f(label,words_features[j],i) for j in range(len(words_features))])
            term2 = sum([p_sentence(words_features, weights, any_label)*
                        sum([feature_f(any_label[j],words_features[j],i)]) for any_label in permutations(label)])
            weights[i] += alpha*(term1-term2)

if __name__ == "__main__":
    print("hello world!")

    #feature function -> a dictionary of 

    #feature function -> whether a set of features and tags that matches what is observed.

    # print(words_train)
    # print(y_train)

    #print(x_train[:10])

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