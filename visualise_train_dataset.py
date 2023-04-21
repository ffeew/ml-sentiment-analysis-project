import sys

FILE_IN = ""
FILE_OUT = ""


def get_tags_by_length(file_in, lang):
    with open(file_in, "r") as f_in:
        tags = {0: 0}
        tag_len = 0
        if lang == "french":
            sentences = f_in.read().split("\n\n")
            for sentence in sentences:
                if "\n" in sentence:
                    words = sentence.split("\n")
                else:
                    words = [sentence]
                for word in words:
                    if word:
                        tag = word.split(" ")[1]
                        if tag.startswith("O"):
                            if tag_len > 0:
                                if tag_len in tags.keys():
                                    tags[tag_len] += 1
                                else:
                                    tags[tag_len] = 1
                            tags[0] += 1
                            tag_len = 0
                        elif tag.startswith("B"):
                            if tag_len > 0:
                                if tag_len in tags.keys():
                                    tags[tag_len] += 1
                                else:
                                    tags[tag_len] = 1
                                tag_len = 1
                            else:
                                tag_len += 1
                        else:
                            tag_len += 1
            return tags
        for line in f_in:
            if line != "\n":
                tag = line.split(" ")[1]
                if tag.startswith("O"):
                    tags[0] += 1
                    tag_len = 0
                elif tag.startswith("B"):
                    if tag_len > 0:
                        if tag_len in tags.keys():
                            tags[tag_len] += 1
                        else:
                            tags[tag_len] = 1
                        tag_len = 1
                    else:
                        tag_len += 1
                else:
                    tag_len += 1
            else:
                if tag_len in tags.keys():
                    tags[tag_len] += 1
                else:
                    tags[tag_len] = 1
                tag_len = 0
    return tags


def tag_count_by_length(entities, file_out):
    print("Count")
    with open(file_out, "w") as f_out:
        f_out.write("Count\n") 
        for key, value in sorted(list(entities.items()), key=lambda x:x[0]):
            if key == 0:
                out = f"The 'O' tag appeared: {value} times"
                print(out)
                f_out.write(out + "\n")
            else:
                out = f"The {key}-length tag sequence had count:{value}"
                print(out)
                f_out.write(out + "\n")
        print("\n")
        f_out.write("\n")


def relative_tag_frequency_by_length(entities, file_out):
    print("Relative Frequency")
    with open(file_out, "a") as f_out:
        f_out.write("Relative Frequency\n")
        entities_list = [(key, value) for key, value in entities.items()]
        total_count = sum([value for _, value in entities_list])
        for key, value in sorted(list(entities.items()), key=lambda x:x[0]):
            if key == 0:
                out = f"The 'O' tag appeared: {value / total_count} times"
                print(out)
                f_out.write(out + "\n")
            else:
                out = f"The {key}-length tag sequence had count:{value / total_count}"
                print(out)
                f_out.write(out + "\n")
        print("\n")
        f_out.write("\n")


def main(file_in, file_out, lang):
    tags = get_tags_by_length(file_in, lang)
    tag_count_by_length(tags, file_out)
    relative_tag_frequency_by_length(tags, file_out)
    del tags[0]
    relative_tag_frequency_by_length(tags, file_out)
    

if __name__ == "__main__":
    FILE_IN = sys.argv[1]
    FILE_OUT = sys.argv[2]
    LANG = sys.argv[3]
    main(FILE_IN, FILE_OUT, LANG)
