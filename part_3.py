from copy import deepcopy
from collections import deque

import numpy as np
import pandas as pd
import json

from Part1 import gen_e

FILE_TRAIN_EN = "./EN/train"
Q_OUT_EN = "./EN/q"

def q_2(tags):
    """First pass counts each occurence as defined in the README,
    second pass calculates each q given the formula."""
    out = {}
    n1 = {}
    n2 = {}
    n3 = {}
    c0 = len(set(tags))
    c1 = {}
    c2 = {}

    y = deque()
    y.append(tags[0])
    y.append(tags[1])
    for tag in tags[2:]:
        y.append(tag)
        if len(y) > 3:
            y.popleft()
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
        if (y[0], y[1]) in c2. keys():
            c2[(y[0], y[1])] += 1
        else:
            c2[(y[0], y[1])] = 1

    print("n1:", n1)
    print("n2:", n2)
    print("n3:", n3)
    print("c0:", c0)
    print("c1:", c1)
    print("c2:", c2)

    y = deque()
    y.append(tags[0])
    y.append(tags[1])
    for tag in tags[2:]:
        y.append(tag)
        if len(y) > 3:
            y.popleft()
        if not (y[0], y[1], y[2]) in out.keys():
            n_1 = n1[y[2]]
            n_2 = n2[(y[1], y[2])]
            n_3 = n3[(y[0], y[1], y[2])]
            c_1 = c1[y[1]]
            c_2 = c2[(y[0], y[1])]
            k_2 = (np.log(n_2 + 1) + 1) / (np.log(n_2 + 1) + 2)
            k_3 = (np.log(n_3 + 1) + 1) / (np.log(n_3 + 1) + 2)
            out[(y[0], y[1], y[2])] = (k_3 * (n_3 / c_2)) + ((1 - k_3) * k_2 * (n_2 / c_1)) + ((1 - k_3) * (1 - k_2) * (n_1 / c0))
    print("out:", out)
    return out


def gen_2nd_ord_trans_probs(file_in):
    out = {}
    len_2_seqs = {}
    len_3_seqs = {}

    with open(file_in) as f_in:
        y = deque()
        for line in f_in:
            if line == "\n":
                y.clear()
                continue
            if len(y) == 3:
                if (y[0], y[1]) in len_2_seqs:
                    len_2_seqs[(y[0], y[1])] += 1
                else:
                    len_2_seqs[(y[0], y[1])] = 1
                if (y[0], y[1], y[2]) in len_3_seqs:
                    len_3_seqs[(y[0], y[1], y[2])] += 1
                else:
                    len_3_seqs[(y[0], y[1], y[2])] = 1
            y.append(line.split()[1])
            if len(y) > 3:
                y.popleft()

    with open(file_in) as f_in:
        y = deque()
        for line in f_in:
            if line == "\n":
                y.clear()
                continue
            if len(y) == 3:
                if not (y[0], y[1], y[2]) in out:
                    out[(y[0], y[1], y[2])] = len_3_seqs[(y[0], y[1], y[2])] / len_2_seqs[(y[0], y[1])]
            y.append(line.split()[1])
            if len(y) > 3:
                y.popleft()
    with open(Q_OUT_EN, "w") as f_out:        
        f_out.write(json.dumps({json.dumps(k): v for k, v in out.items()}))


def estimate_order_2_transition_parameters(path: str) -> dict[dict]:
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
    df2 = deepcopy(df.iloc[:,-1:])
    df2.loc[-1] = "START"
    df2.index = df2.index + 1
    df2 = df2.sort_index()
    df2 = df2[:-1]
    df.insert(loc=1, column="previous", value=df2)

    transition = {}
    previous_1 = None
    previous_2 = None
    for tag in df["tag"]:
        # check for start probability
        if previous_2 is None:
            inner = transition.get("START", {})
            inner[("START", tag)] = inner.get(("START", tag), 0) + 1
            transition["START"] = inner
            previous_1 = tag
            previous_2 = "START"
        elif tag == None:
            inner = transition.get(previous_2, {})
            inner[(previous_1, "STOP")] = inner.get((previous_2, "STOP"), 0) + 1
            transition[previous_2] = inner
            previous_1 = tag
            previous_2 = tag
        else:
            inner = transition.get(previous_2, {})
            inner[(previous_1, tag)] = inner.get((previous_1, tag), 0) + 1
            transition[previous_2] = inner
            previous_2 = deepcopy(previous_1)
            previous_1 = tag

    state_count = {}
    for p2, value in transition.items():
        for (p1, _), c in value.items():
            if (p2, p1) in state_count:
                state_count[(p2, p1)] += c
            else:
                state_count[(p2, p1)] = c

    # divide by the number of times the previous 2 tags occurs
    for p2, value in transition.items():
        for (p1, t), v in value.items():
            value[(p1, t)] = v/state_count[(p2, p1)]

    return transition


def get_transition_probabilities(transition: dict, previous_1: str, previous_2:str, current: str) -> float:
    """
    Helper function to get the transition probability when given the previous and current state
    """
    return transition[previous_2].get((previous_1, current), 0)


def viterbi(sentence: str, transition: dict, emission: gen_e) -> tuple:
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
    backpointer = np.zeros((len(words)-1, len(states)))

    # fill up the start probabilities
    for i, key in enumerate(states):
        viterbi[0, i] = np.log(transition["START"].get(
            ("START", key), 0) * emission.get_e(key, words[0]))
    # fill in the rest of the matrix
    for i in range(1, len(words)):
        for j, tag in enumerate(states):
            # get probabilities for each state
            prob = [viterbi[i-1, k] + np.log(get_transition_probabilities(transition, states[k], states[k], tag)) + np.log(
                emission.get_e(tag, words[i])) for k in range(len(states))]
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


def main():
    print(estimate_order_2_transition_parameters("EN/train"))
    
    # gen_2nd_ord_trans_probs(FILE_TRAIN_EN)
    # with open(Q_OUT_EN, "r") as f:
    #     d = json.loads(f.read())
    #     q = {tuple(json.loads(k)): v for k, v in d.items()}
    # print(q)

    count = gen_e()
    count.count_e("FR/train")
    trans = estimate_order_2_transition_parameters("FR/train")

    path_in = "FR/dev.in"
    path_out = "FR/dev.p3.out"
    with open(path_in, 'r') as f:
        data = f.read()
    sentences = data.split("\n\n")[:-1]

    tags = [viterbi(sentence, trans, count) for sentence in sentences]

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
    count = gen_e()
    count.count_e("EN/train")
    trans = estimate_order_2_transition_parameters("EN/train")

    path_in = "EN/dev.in"
    path_out = "EN/dev.p3.out"
    with open(path_in, 'r') as f:
        data = f.read()
    sentences = data.split("\n\n")[:-1]

    tags = [viterbi(sentence, trans, count) for sentence in sentences]

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
