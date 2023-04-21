from copy import deepcopy
from collections import deque

import numpy as np
import pandas as pd
import json

from Part1 import gen_e


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
    viterbi = np.zeros((len(words), len(states) ** 2))
    backpointer = np.zeros((len(words)-1, len(states) ** 2))

    # fill up the start probabilities
    for i, key in enumerate(states):
        viterbi[0, i] = np.log(transition["START"].get(
            ("START", key), 0) * emission.get_e(key, words[0]))
    # fill in the rest of the matrix
    for i in range(1, len(words)):
        for j, tag in enumerate(states):
            # get probabilities for each state
            prob = [viterbi[i-1, k] + np.log(get_transition_probabilities(transition, states[k], states[l], tag)) + np.log(
                emission.get_e(tag, words[i])) for k in range(len(states)) for l in range(len(states))]
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
        result.append(states[int(s) % len(states)])
    return result


def main():
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
