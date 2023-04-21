import sys
import json

from nltk.stem import WordNetLemmatizer
from collections import Counter

FILE_IN = ""
PREFIX_OUT = "prefix_tagged"
SUFFIX_OUT = "suffix_tagged"
FILE_TEMP = ""
MODE = "train"


def nested_get(dic, keys):    
    for key in keys:
        try:
            dic = dic[key]
        except KeyError:
            return None
    return dic

def nested_set(dic, keys, value):
    # define nested dict
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})

    # iterate
    if len(keys) > 1:
        # if branch exists
        if dic:
            # if key is branch's key
            if keys[-1] in dic.keys():
                # end of branch
                if type(dic[keys[-1]]) is str:
                    dic[keys[-1]] = [dic[keys[-1]], value]
                # list at end exists
                elif type(dic[keys[-1]]) is list:
                    dic[keys[-1]].append(value)
                # assume dic, maj assignment exists
                elif "tag" in dic[keys[-1]].keys():
                    # append to maj list, resolve later
                    if type(dic[keys[-1]]["tag"]) is list:
                        dic[keys[-1]]["tag"].append(value)
                    # create new list of majs
                    else:
                        dic[keys[-1]]["tag"] = [dic[keys[-1]]["tag"], value]
                # new maj assignment
                else:
                    dic[keys[-1]].update(tag=value)
            # new sub-branch
            else:
                temp_dict = {keys[-1]: value}
                dic.update(temp_dict)
        # new branch
        else:
            dic[keys[-1]] = value

    # handle single len chars
    else:
        key = keys[0]
        if key in dic.keys():
            if dic[key].values():
                dic[key].update(tag=value)
        else:
            dic[key] = {"tag": value}


def tag_tree(tree):
    if "tag" in tree.keys():
        if type(tree["tag"]) is list:
            # tree["tag"] = Counter(tree["tag"]).most_common(1)[0][0]
            counts = Counter(tree["tag"]).most_common(len(tree["tag"]))
            total_count = sum([count for (_, count) in counts])
            tree["tag"] = tuple([(tag, count / total_count) for (tag, count) in counts])
        elif type(tree["tag"]) is str:
            tree["tag"] = (tree["tag"], 1.0)
        upstream_tag = tree["tag"]
    for key, branch in tree.items():
        if key != "tag":
            if type(branch) is dict:
                if not "tag" in branch.keys():
                    tree[key]["tag"] = upstream_tag
                tree[key] = tag_tree(branch)
            elif type(branch) is list:
                counts = Counter(tree[key]).most_common(len(tree[key]))
                total_count = sum([count for (_, count) in counts])
                tree[key] = {"tag": tuple([(tag, count / total_count) for (tag, count) in counts])}
                # tree[key] = {"tag": Counter(branch).most_common(1)[0][0]}
            elif type(branch) is str:
                tree[key] = {"tag": tuple((branch, 1.0))}
    return tree


def get_prefix_estimation(words, word):
    letters = [letter for letter in word]
    while len(letters) > 0:
        if nested_get(words, letters) is not None:
            dict_out = {}
            list_of_probs = nested_get(words, letters + ["tag"])
            for item in list_of_probs:
                if type(item) is list:
                    dict_out[item[0]] = float(item[1])
            if dict_out:
                return dict_out
            else:
                # assume flat list
                return {list_of_probs[0]: float(list_of_probs[1])}
        else:
            return get_prefix_estimation(words, word[:-1])
    return {}


def get_suffix_estimation(words, word):
    letters = [letter for letter in word[::-1]]
    while len(letters) > 0:
        if nested_get(words, letters) is not None:
            dict_out = {}
            list_of_probs = nested_get(words, letters + ["tag"])
            for item in list_of_probs:
                if type(item) is list:
                    dict_out[item[0]] = float(item[1])
            if dict_out:
                return dict_out
            else:
                # assume flat list
                return {list_of_probs[0]: float(list_of_probs[1])}
        else:
            return get_prefix_estimation(words, word[:-1])
    return None


def gen_affix_dictionaries(f_in, pf_out, sf_out, f_temp):
    words = {}
    word_Lemmatized = WordNetLemmatizer()

    # sort data by word len, write to temp
    with open(f_in, "r", encoding="utf-8") as file_in:
        sentences = file_in.read().split("\n")
        words_tags = [tuple(sentence.split(" ")) for sentence in sentences if sentence != ""]
        words_tags = [(word_Lemmatized.lemmatize(word.lower()), tag) for (word, tag) in words_tags]
        lines = [word + " " + tag + "\n" for (word, tag) in sorted(words_tags, key=lambda x: len(x[0]), reverse=True)]

    with open(f_temp, "w", encoding="utf-8") as file_temp:
        file_temp.writelines(lines)

    # construct nested dict
    with open(f_temp, "r", encoding="utf-8") as file_temp:
        for line in file_temp:
            if line == "\n":
                continue
            word, tag = line.split(" ")
            tag = tag[:-1]
            letters = [x for x in word]
            nested_set(words, letters, tag)

    # augment dict section
    stl_words = [(word, tag) for (word, tag) in sorted(words_tags, key=lambda x: len(x[0]))]

    roots = {root: [] for root in words.keys()}

    # assign pseudoroot tags as majority of tags in roots
    for root in words.keys():
        shortest_len = 0
        for word, tag in stl_words:
            if shortest_len == 0:
                if word.startswith(root):
                    shortest_len = len(word)
                    roots[root].append((word, tag))
            else:
                if len(word) > shortest_len:
                    break
                if word.startswith(root):
                    roots[root].append((word,tag))
    for root, pseudoroots_tags in roots.items():
        pseudoroots_tags.sort(key=lambda x: x[0])
        pseudoroots, tags  = list(zip(*pseudoroots_tags))
        pseudoroots_counts = Counter(pseudoroots)
        majority_pseudoroots_tags = {}
        start_index = 0
        unique_pseudoroot = ""
        for pseudoroot in pseudoroots:
            if pseudoroot != unique_pseudoroot:
                # majority_pseudoroots_tags[pseudoroot] = Counter(tags[start_index:start_index + pseudoroots_counts[pseudoroot]]).most_common(1)[0][0]
                majority_pseudoroots_tags[pseudoroot] = Counter(tags[start_index:start_index + pseudoroots_counts[pseudoroot]]).most_common(pseudoroots_counts[pseudoroot])
                total_count = sum([count for _, count in majority_pseudoroots_tags[pseudoroot]])
                majority_pseudoroots_tags[pseudoroot] = tuple([(tag, count / total_count) for (tag, count) in majority_pseudoroots_tags[pseudoroot]])
                start_index = pseudoroots_counts[pseudoroot]
                unique_pseudoroot = pseudoroot
        roots[root] = majority_pseudoroots_tags
    # print(roots["."])

    # propagate psuedoroots upstream; assign root tags
    for root, pseudoroots_tags in roots.items():
        # print(roots[root])
        if len(pseudoroots_tags.keys()) == 1:
            roots[root] = {"tag": list(pseudoroots_tags.values())[0]}
            # print(roots[root])
        # assume pseudoroots greater than length 2 affect root regardless of distance from root 
        else:
            # let Counter decide between ties of "O" and other tags which is more common
            # roots[root]["tag"] = Counter(list(roots[root].values())).most_common(1)[0][0]
            roots[root]["tag"] = max(roots[root].values(), key=len)
    # print(roots)

    # print(words)
    # propagate tags downstream in words
    for root in words.keys():
        # print(roots[root]["tag"])
        words[root]["tag"] = roots[root]["tag"]
        words[root] = tag_tree(words[root])
    
    # print(words["."])
    # write dict out
    with open(pf_out, "w", encoding="utf-8") as file_out:
        file_out.write(json.dumps(words))

    # TODO suffix

    words = {}
    word_Lemmatized = WordNetLemmatizer()

    # sort data by word len, write to temp
    with open(f_in, "r", encoding="utf-8") as file_in:
        sentences = file_in.read().split("\n")
        words_tags = [tuple(sentence.split(" ")) for sentence in sentences if sentence != ""]
        words_tags = [(word_Lemmatized.lemmatize(word.lower())[::-1], tag) for (word, tag) in words_tags]
        lines = [word + " " + tag + "\n" for (word, tag) in sorted(words_tags, key=lambda x: len(x[0]), reverse=True)]

    with open(f_temp, "w", encoding="utf-8") as file_temp:
        file_temp.writelines(lines)

    # construct nested dict
    with open(f_temp, "r", encoding="utf-8") as file_temp:
        for line in file_temp:
            if line == "\n":
                continue
            word, tag = line.split(" ")
            tag = tag[:-1]
            letters = [x for x in word]
            nested_set(words, letters, tag)

    # TODO for suffix estimation
    stl_words_reversed = [(word[::-1], tag) for (word, tag) in stl_words]
    print(stl_words_reversed)

    roots = {root: [] for root in words.keys()}

    # assign pseudoroot tags as majority of tags in roots
    for root in words.keys():
        shortest_len = 0
        for word, tag in stl_words_reversed:
            if shortest_len == 0:
                if word.startswith(root):
                    shortest_len = len(word)
                    roots[root].append((word, tag))
            else:
                if len(word) > shortest_len:
                    break
                if word.startswith(root):
                    roots[root].append((word,tag))
    for root, pseudoroots_tags in roots.items():
        pseudoroots_tags.sort(key=lambda x: x[0])
        pseudoroots, tags  = list(zip(*pseudoroots_tags))
        pseudoroots_counts = Counter(pseudoroots)
        majority_pseudoroots_tags = {}
        start_index = 0
        unique_pseudoroot = ""
        for pseudoroot in pseudoroots:
            if pseudoroot != unique_pseudoroot:
                # majority_pseudoroots_tags[pseudoroot] = Counter(tags[start_index:start_index + pseudoroots_counts[pseudoroot]]).most_common(1)[0][0]
                majority_pseudoroots_tags[pseudoroot] = Counter(tags[start_index:start_index + pseudoroots_counts[pseudoroot]]).most_common(pseudoroots_counts[pseudoroot])
                total_count = sum([count for _, count in majority_pseudoroots_tags[pseudoroot]])
                majority_pseudoroots_tags[pseudoroot] = tuple([(tag, count / total_count) for (tag, count) in majority_pseudoroots_tags[pseudoroot]])
                start_index = pseudoroots_counts[pseudoroot]
                unique_pseudoroot = pseudoroot
        roots[root] = majority_pseudoroots_tags

    # propagate psuedoroots upstream; assign root tags
    for root, pseudoroots_tags in roots.items():
        # print(roots[root])
        if len(pseudoroots_tags.keys()) == 1:
            roots[root] = {"tag": list(pseudoroots_tags.values())[0]}
            # print(roots[root])
        # assume pseudoroots greater than length 2 affect root regardless of distance from root 
        else:
            # let Counter decide between ties of "O" and other tags which is more common
            # roots[root]["tag"] = Counter(list(roots[root].values())).most_common(1)[0][0]
            roots[root]["tag"] = max(roots[root].values(), key=len)
    # print(roots)

    # propagate tags downstream in words
    for root in words.keys():
        # print(roots[root]["tag"])
        words[root]["tag"] = roots[root]["tag"]
        words[root] = tag_tree(words[root])
    print(words)

    # write dict out
    with open(sf_out, "w", encoding="utf-8") as file_out:
        file_out.write(json.dumps(words))


def load_prefixes(lang):
    with open("./" + lang + "/" + PREFIX_OUT, "r", encoding="utf-8") as file_out:
        words = json.loads(file_out.read())
    return words


def main():
    if len(sys.argv) < 3:
        print ('Please make sure you have installed Python 3.4 or above!')
        print ("Usage on Windows:  python visualise_data.py <file_in> <file_out> <file_temp>")
        print ("Usage on Linux/Mac:  python visualise_data.py <file_in> <file_out> <file_temp>")
        print ("Example: python affix_estimation.py FR train affix_tagged traintemp")
        sys.exit()

    LANG = sys.argv[1]
    FILE_IN = "./" + LANG + "/" + sys.argv[2]
    PREFIX_OUT = "./" + LANG + "/" + sys.argv[3]
    SUFFIX_OUT = "./" + LANG + "/" + sys.argv[4]
    FILE_TEMP = "./" + LANG + "/" + sys.argv[5]
    gen_affix_dictionaries(FILE_IN, PREFIX_OUT, SUFFIX_OUT, FILE_TEMP)

    # # example call dict
    # with open(PREFIX_OUT, "r", encoding="utf-8") as file_out:
    #     words = json.loads(file_out.read())

    # # example of using get_prefix_estimation
    # if LANG == "FR":
    #     # print(words)
    #     print(get_prefix_estimation(words, "brasseri"))  # tag O
    #     print(get_prefix_estimation(words, "purée"))  # tag B-positive based on majority
    #     print(get_prefix_estimation(words, "pur"))  # tag O based on upstream tag i.e. suffix pur with tag O
    #     print(get_prefix_estimation(words, "puréet"))  # estimated to be B-positive based on upstream tag i.e. purée with tag B-positive
    # elif LANG == "EN":
    #     print(get_prefix_estimation(words, "no"))  # tag B-NP by majority
    #     print(get_prefix_estimation(words, "helpful"))  # tag I-ADJP
    #     print(get_prefix_estimation(words, "helpfully"))  # tag I-ADJP based on upstream tag i.e. suffix helpful with tag I-ADJP
    #     print(get_prefix_estimation(words, "hel"))  # tag B-NP based on upstream tag i.e. suffix hel with tag B-NP
    #     print(get_prefix_estimation(words, "hell"))  # tag B-NP

    with open(SUFFIX_OUT, "r", encoding="utf-8") as file_out:
        words = json.loads(file_out.read())

    # example of using get_suffix_estimation
    if LANG == "FR":
        # print(words)
        print(get_suffix_estimation(words, "rts")) 
        print(get_suffix_estimation(words, "boulangerie"))
        print(get_suffix_estimation(words, "eil")) 
        print(get_suffix_estimation(words, "produitse")) 
    elif LANG == "EN":
        print(get_suffix_estimation(words, "no"))
        print(get_suffix_estimation(words, "helpful")) 
        print(get_suffix_estimation(words, "helpfully"))
        print(get_suffix_estimation(words, "hel"))
        print(get_suffix_estimation(words, "hell"))

if __name__ == "__main__":
    main()
