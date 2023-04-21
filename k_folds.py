import numpy as np
from sklearn.model_selection import KFold
import Part1

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

def get_words(filename):

    raw = read_data(filename)

    sentences = sentence_cutting(raw)[:-1]

    words = word_cutting(sentences)

    return words

# def get_labels(filename):
#     with open(filename,"r",encoding="utf8") as file:
#         y_count = json.loads(file.read())
    
#     for i,label in enumerate(y_count.keys()):
#         y_count[label] = i

#     return y_count

def split_data(filename):

    words = get_words(filename)

    words_train, y = process_tagging(words)

    kf = KFold(n_splits=5)
    kf.get_n_splits(words_train)

    return words_train, y, kf

def k_folds_part_1():
    Part1.

if __name__ == "__main__":

    words_train,y, kf = split_data("EN/train")

    for i, (train_index,test_index) in enumerate(kf.split(words_train)):
        
        if i == 0:
            print(train_index)
            print(test_index)