from Part1 import gen_e
import pandas as pd
import numpy as np


def estimate_transition_parameters(path: str) -> dict[dict]:
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

    transition = {}
    previous = None
    for tag in df["tag"]:
        # check for start probability
        if previous is None:
            inner = transition.get("START", {})
            inner[tag] = inner.get(tag, 0) + 1
            transition["START"] = inner
            previous = tag
        elif tag == None:
            inner = transition.get(previous, {})
            inner["STOP"] = inner.get("STOP", 0) + 1
            transition[previous] = inner
            previous = tag
        else:
            inner = transition.get(previous, {})
            inner[tag] = inner.get(tag, 0) + 1
            transition[previous] = inner
            previous = tag

    # divide by the number of times the previous tag occurs
    state_count = df["tag"].value_counts()
    for key, value in transition.items():
        if key == "START":
            continue
        for k, v in value.items():
            value[k] = v/state_count[key]

    # handle the start probabilities
    start_count = 0
    for value in transition["START"].values():
        start_count += value
    for key, value in transition["START"].items():
        transition["START"][key] = value/start_count

    return transition


def get_transition_probabilities(transition: dict, previous: str, current: str) -> float:
    """
    Helper function to get the transition probability when given the previous and current state
    """
    return transition[previous].get(current, 0)


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
    backpointer = np.zeros((len(words), len(states)))

    # fill up the start probabilities
    for i, key in enumerate(states):
        viterbi[0, i] = np.log(transition["START"].get(
            key, 0) * emission.get_e(key, words[0]))
        backpointer[0, i] = 0
    # fill in the rest of the matrix
    for i in range(1, len(words)):
        for j, tag in enumerate(states):
            # get probabilities for each state
            prob = [viterbi[i-1, k] + np.log(get_transition_probabilities(transition, states[k], tag)) + np.log(
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


if __name__ == "__main__":
    # generate the tags for FR/dev.in
    count = gen_e()
    count.count_e("FR/train")
    trans = estimate_transition_parameters("FR/train")

    path_in = "FR/dev.in"
    path_out = "FR/dev.p2.out"
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
    trans = estimate_transition_parameters("EN/train")

    path_in = "EN/dev.in"
    path_out = "EN/dev.p2.out"
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
