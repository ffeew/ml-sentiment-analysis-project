from collections import deque
import numpy as np
import json

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

# tags = [1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3]
# q_2(tags)

gen_2nd_ord_trans_probs(FILE_TRAIN_EN)
with open(Q_OUT_EN, "r") as f:
    d = json.loads(f.read())
    q = {tuple(json.loads(k)): v for k, v in d.items()}
print(q)
