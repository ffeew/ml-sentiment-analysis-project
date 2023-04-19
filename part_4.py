from tqdm import tqdm
from collections import deque
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn

import nltk
import numpy as np
import pandas as pd

from Part1 import gen_e

def nltk_setup():
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')


def thede_smoothing(tags, lang):
    """First pass counts each occurence as defined in the README,
    second pass calculates each q given the formula."""
    out = {}
    n1 = {}
    n2 = {}
    n3 = {}
    c0 = len(tags)
    c1 = {}
    c2 = {}

    y = deque()
    y.append("START")
    y.append("START")
    for tag in tags:
        if tag is None:
            tag = "STOP"
        y.append(tag)
        if len(y) > 3:
            y.popleft()
        if len(y) == 3:
            if y[2] in n1.keys():
                n1[y[2]] += 1
            else:
                n1[y[2]] = 1
            if (y[1], y[2]) in n2.keys():
                n2[(y[1], y[2])] += 1
            else:
                n2[(y[1], y[2])] = 1
            if (y[0], y[1], y[2]) in n3.keys():
                n3[(y[0], y[1], y[2])] += 1
            else:
                n3[(y[0], y[1], y[2])] = 1
            if y[1] in c1.keys():
                c1[y[1]] += 1
            else:
                c1[y[1]] = 1
            if (y[0], y[1]) in c2.keys():
                c2[(y[0], y[1])] += 1
            else:
                c2[(y[0], y[1])] = 1
            if tag == "STOP":
                y = deque()
                y.append("START")
                y.append("START")
    
    ls = {0: 0, 1: 0, 2: 0, 3: 0}
    t = deque()
    # print(set(tags))
    for tag in tags:
        if tag != "O" and tag is not None:
            if tag.startswith("B"):
                if len(t) >= 1:
                    if len(t) <= 3:
                        ls[len(t)] += 1
                    # ls[min(3, len(t))] += 1
                    t = deque()
            t.append(tag)
        elif tag == "O":
            ls[0] += 1
            if len(t) >= 1:
                if len(t) <= 3:
                    ls[len(t)] += 1
            t = deque()
        else:
            if len(t) >= 1:
                if len(t) <= 3:
                    ls[len(t)] += 1
            t = deque()
    s = ls[0] + ls[1] + ls[2] + ls[3]
    a0 = ls[0] / s
    a1 = ls[1] / s
    a2 = ls[2] / s
    a3 = ls[3] / s
    ss = np.exp(a0) + np.exp(a1) + np.exp(a2) + np.exp(a2)
    b0 = np.exp(a0) / ss
    b1 = np.exp(a1) / ss
    b2 = np.exp(a2) / ss
    b3 = np.exp(a3) / ss
    print(a0, a1, a2, a3)
    print(b0, b1, b2, b3)

    y = deque()
    y.append("START")
    y.append("START")
    for tag in tags:
        if tag is None:
            tag = "STOP"
        y.append(tag)
        if len(y) > 3:
            y.popleft()
        if len(y) == 3:
            if not (y[0], y[1], y[2]) in out.keys():
                n_1 = n1[y[2]]
                n_2 = n2[(y[1], y[2])]
                n_3 = n3[(y[0], y[1], y[2])]
                c_1 = c1[y[1]]
                c_2 = c2[(y[0], y[1])]
                k_2 = (np.log(n_2 + 1) + 1) / (np.log(n_2 + 1) + 2)
                k_3 = (np.log(n_3 + 1) + 1) / (np.log(n_3 + 1) + 2)
                # out[(y[0], y[1], y[2])] = (k_3 * (n_3 / c_2)) + ((1 - k_3) * k_2 * (n_2 / c_1)) + ((1 - k_3) * (1 - k_2) * (n_1 / c0))
                # out[(y[0], y[1], y[2])] = (a3 * k_3 * (n_3 / c_2)) + (a2 * (1 - k_3) * k_2 * (n_2 / c_1)) + (a1 * (1 - k_3) * (1 - k_2) * (n_1 / c0))
                # out[(y[0], y[1], y[2])] = n_1 / c0
                # out[(y[0], y[1], y[2])] = (n_3 / c_2) + (n_2 / c_1) + (n_1 / c0)
                if lang == "english":
                    out[(y[0], y[1], y[2])] = b3 * (n_3 / c_2) + b2 * (n_2 / c_1) + (b1 + (b0 / 3)) * (n_1 / c0)
                elif lang == "french":
                    out[(y[0], y[1], y[2])] = (b3 / 3) * (n_3 / c_2) + (b2 / 2) * (n_2 / c_1) + (b1 + b0) * (n_1 / c0)
            if tag == "STOP":
                y = deque()
                y.append("START")
                y.append("START")

    dict_out = {}
    for (p2, p1, p0), value in out.items():
        if p2 in dict_out:
            dict_out[p2][(p1, p0)] = value
        else:
            dict_out[p2] = {(p1, p0): value}
    return dict_out, b0


def lexical_probability(words, tags):
    """First pass counts each occurence as defined in the README,
    second pass calculates each q given the formula."""
    out = {}
    n2 = {}
    n3 = {}
    c1 = {}
    c2 = {}

    y = deque()
    y.append((None, "START"))
    y.append((None, "START"))
    for (word, tag) in zip(words, tags):
        if tag is None:
            word = None
            tag = "STOP"
        y.append((word, tag))
        if len(y) > 3:
            y.popleft()
        if len(y) == 3:
            if (y[1][1], y[2][0]) in n2.keys():
                n2[(y[1][1], y[2][0])] += 1
            else:
                n2[(y[1][1], y[2][0])] = 1
            if (y[0][1], y[1][1], y[2][0]) in n3.keys():
                n3[(y[0][1], y[1][1], y[2][0])] += 1
            else:
                n3[(y[0][1], y[1][1], y[2][0])] = 1
            if y[1][1] in c1.keys():
                c1[y[1][1]] += 1
            else:
                c1[y[1][1]] = 1
            if (y[0][1], y[1][1]) in c2.keys():
                c2[(y[0][1], y[1][1])] += 1
            else:
                c2[(y[0][1], y[1][1])] = 1
            if tag == "STOP":
                y = deque()
                y.append((None, "START"))
                y.append((None, "START"))
    
    # ls = {1: 0, 2: 0, 3: 0}
    # t = deque()
    # # print(set(tags))
    # for tag in tags:
    #     if tag != "O" and tag is not None:
    #         if tag.startswith("B"):
    #             if len(t) >= 1:
    #                 if len(t) <= 3:
    #                     ls[len(t)] += 1
    #                 # ls[min(3, len(t))] += 1
    #                 t = deque()
    #         t.append(tag)
    # s = ls[1] + ls[2] + ls[3]
    # a1 = 1 / (ls[1] / s)
    # a2 = 1 / (ls[2] / s)
    # a3 = 1 / (ls[3] / s)

    y = deque()
    y.append((None, "START"))
    y.append((None, "START"))
    for (word, tag) in zip(words, tags):
        if tag is None:
            word = None
            tag = "STOP"
        y.append((word, tag))
        if len(y) > 3:
            y.popleft()
        if len(y) == 3:
            if not (y[0][1], y[1][1], y[2][1]) in out.keys():
                n_2 = n2[(y[1][1], y[2][0])]
                n_3 = n3[(y[0][1], y[1][1], y[2][0])]
                c_1 = c1[y[1][1]]
                c_2 = c2[(y[0][1], y[1][1])]
                k_2 = (1 / (np.log(n_3 + 1) + 2))
                k_3 = (np.log(n_3 + 1) + 1) / (np.log(n_3 + 1) + 2)
                out[(y[0][1], y[1][1], y[2][0])] = (k_3 * (n_3 / c_2)) + (k_2 * (n_2 / c_1))
            if tag == "STOP":
                y = deque()
                y.append((None, "START"))
                y.append((None, "START"))

    dict_out = {}
    for (p2, p1, p0), value in out.items():
        if p2 in dict_out:
            dict_out[p2][(p1, p0)] = value
        else:
            dict_out[p2] = {(p1, p0): value}
    return dict_out


def estimate_new_transition_parameters(path: str, lang: str) -> dict[dict]:
    """Estimate the transition parameters from the training data.
    :param path: The path to the training data.
    :return: A dictionary of dictionaries of transition probabilities.
    """
    # parse the data
    data = []
    with open(path, 'r') as f:
        for line in f.readlines():
            # remove the newline character
            data.append(line[:-1].split(" "))

    # convert to dataframe
    df = pd.DataFrame(data, columns=["word", "tag"])
    transition, b0 = thede_smoothing(df["tag"], lang)
    lexicon = lexical_probability(df["word"], df["tag"])

    return transition, lexicon, b0


def get_transition_probabilities(transition: dict, previous_1: str, previous_2:str, current: str, b0:float, lang: str) -> float:
    """
    Helper function to get the transition probability when given the previous and current state
    """
    x = b0 / len(transition.items())
    if lang == "french":
        x = 0
    return transition[previous_2].get((previous_1, current), x)


def get_lexical_probabilities(lexicon: dict, previous_1: str, previous_2:str, current_word: str, lang: str) -> float:
    """
    Helper function to get the transition probability when given the previous and current state
    """
    x = 0
    if lang == "english":
        x = min(lexicon[previous_2].values())
    return lexicon[previous_2].get((previous_1, current_word), x)


def viterbi(sentence: str, transition: dict, lexicon: dict, emission: gen_e, b0: float, lang: str) -> tuple:
    """Estimate the tag for each word from the sentence.
    :param sentence: A string of n words.
    :param transition: the transition matrix.
    :param emission: The emission matrix.
    :return: A tuple of tags.
    """
    # split the sentence into words
    words = sentence.split("\n")

    states = [key for key in transition.keys() if key != "START"]

    # initialize the viterbi matrix
    viterbi = np.zeros((len(words), len(states)))
    backpointer = np.zeros((len(words) - 1, len(states)))

    # fill up the start probabilities
    for i, key in enumerate(states):
        viterbi[0, i] = np.log(transition["START"].get(
            ("START", key), 0) * emission.get_p(key, words[0]))
    # fill in the rest of the matrix
    for i in range(1, len(words)):
        for j, tag in enumerate(states):
            # get probabilities for each state

            # prob = [viterbi[i-1, k] + np.log(get_transition_probabilities(transition, states[k], states[k], tag, lang) * 
            #                                  get_lexical_probabilities(lexicon, states[k], states[k], words[i], lang)) + 
            #                                  np.log(emission.get_p(tag, words[i])) for k in range(len(states))]
            prob = [viterbi[i-1, k] + np.log(get_transition_probabilities(transition, states[k], states[k], tag, b0, lang)) + 
                    np.log(emission.get_p(tag, words[i])) for k in range(len(states))]
            backpointer[i - 1, j] = np.argmax(prob)
            viterbi[i, j] = np.max(prob)

    S = np.zeros(len(words))
    last_state = np.argmax(viterbi[len(words) - 1, :])
    S[0] = last_state

    backtrack_index = 1
    for i in range(len(words) - 2, -1, -1):
        S[backtrack_index] = backpointer[i, int(last_state)]
        last_state = backpointer[i, int(last_state)]
        backtrack_index += 1
    S = np.flip(S, axis=0)

    result = []
    for s in S:
        result.append(states[int(s)])
    return result


def word_preprocess(sentences: list, lang:str, mode:str, sentence_tags: list=None) -> list:
    if mode == "train":
        text = pd.DataFrame({"original": sentences, "tags": sentence_tags})
    else:
        text = pd.DataFrame([" ".join(x.split("\n")) for x in sentences]).rename(columns={0: "original"})
    # Step - a : Remove blank rows if any.
    text['original'].dropna(inplace=True)
    # Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
    text['original'] = [entry.lower() for entry in text['original']]
    # Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
    text['original'] = [word_tokenize(entry, lang) for entry in text['original']]
    # Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
    # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    for index,entry in tqdm(enumerate(text['original'])):
        # Declaring Empty List to store the words that follow the rules for this step
        Final_words = []
        # Initializing WordNetLemmatizer()
        word_Lemmatized = WordNetLemmatizer()
        # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
        for word, tag in pos_tag(entry):
            # Below condition is to check for Stop words and consider only alphabets
            if word not in stopwords.words(lang) and word.isalpha():
            # if word.isalpha():
                word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                Final_words.append(word_Final)
            else:
                Final_words.append(word)
        # The final processed set of words for each iteration will be stored in 'text_final'
        text.loc[index,'text_final'] = str(Final_words)
    # out = ["\n".join(x) for x in text.loc[:, 'text_final'].values.tolist()]
    if mode == "train":
        out = [x[1:-1].split(", ") for x in text.loc[:, 'text_final'].values.tolist()]
        out = ["".join([x[1:-1] for x in y]) for y in out]
    else:
        out = [x[1:-1].split(", ") for x in text.loc[:, 'text_final'].values.tolist()]
        out = ["\n".join([x[1:-1] for x in y]) for y in out]
    return out


def train_preprocess(path_in, path_out, lang):
    data = []
    with open(path_in, 'r') as f:
        for line in f.readlines():
            # remove the newline character
            data.append(line[:-1].split(" "))
    text = [x[0] for x in data]
    tags = [x[1] if len(x) == 2 else '' for x in data]
    sentence = []
    sentence_tags = []
    data_out = []
    for word, tag in zip(text, tags):
        if word == '' and len(sentence) > 0:
            cleaned = word_preprocess(sentence, lang, "train", sentence_tags)
            cleaned_data = list(filter(None, [[x, y] if x != "" else None for x, y in zip(cleaned, sentence_tags)]))
            for x in cleaned_data:
                data_out.append(x)
            data_out.append("")
            sentence = []
            sentence_tags = []
        # elif tag != "O":
        else:
            sentence.append(word)
            sentence_tags.append(tag)
    with open(path_out, "w", encoding="utf-8") as f_out:
        for x in data_out:
            if x == "":
                f_out.write("\n")
            else:
                
                if "\n" in x[0]:
                    x[0] = x[0].encode("unicode_escape").decode("utf-8")
                f_out.write(x[0] + " " + x[1] + "\n")

def main():  
    # nltk_setup()
    # train_preprocess("EN/train", "EN/train_new", "english")
    # train_preprocess("FR/train", "FR/train_new", "french")
    # return

    count = gen_e("FR")
    # count.count_e("FR/train")
    # trans = estimate_new_transition_parameters("FR/train")
    trans, lex, b0 = estimate_new_transition_parameters("FR/train", "french")

    path_in = "FR/dev.in"
    path_out = "FR/dev.p4test.out"
    with open(path_in, 'r') as f:
        data = f.read()
    sentences = data.split("\n\n")[:-1]
    # sentences = word_preprocess(data.split("\n\n")[:-1], "french", "test")
    # sentences = word_preprocess(sentences, "french")
    # sentences = [word_preprocess(" ".join(sentence.split("\n"))) for sentence in sentences]
    # print(sentences)
    # return
    # tags = [viterbi(sentence, trans, count) for sentence in sentences]
    tags = [viterbi(sentence, trans, lex, count, b0, "french") for sentence in sentences]

    output = []
    for i, sentence in enumerate(sentences):
        words = sentence.split("\n")
        for j, word in enumerate(words):
            output.append(word + " " + tags[i][j])
        output.append("")
    final = "\n".join(output)
    final = final + "\n"
    with open(path_out, 'w') as f:
        f.write(final)

    # generate the tags for EN/dev.in
    count = gen_e("EN")
    # count.count_e("EN/train")
    # trans = estimate_new_transition_parameters("EN/train")
    trans, lex, b0 = estimate_new_transition_parameters("EN/train", "english")

    path_in = "EN/dev.in"
    path_out = "EN/dev.p4test.out"
    with open(path_in, 'r') as f:
        data = f.read()
    sentences = data.split("\n\n")[:-1]
    # sentences = word_preprocess(data.split("\n\n")[:-1], "english", "test")

    tags = [viterbi(sentence, trans, lex, count, b0, "english") for sentence in sentences]

    output = []
    for i, sentence in enumerate(sentences):
        words = sentence.split("\n")
        for j, word in enumerate(words):
            output.append(word + " " + tags[i][j])
        output.append("")
    final = "\n".join(output)
    final = final + "\n"
    with open(path_out, 'w') as f:
        f.write(final)
    print("done")


if __name__ == "__main__":
    main()
