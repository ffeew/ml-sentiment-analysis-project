from thefuzz import fuzz

def get_words(filename):
    with open(filename, "r") as file:
        words = file.read().split("\n")
    
    return words

def closest_word(words,target_word):
    best_word = ""
    max_score = 0
    for word in words:
        if fuzz.ratio(word,target_word)>max_score:
            best_word = word
            max_score = fuzz.ratio(word,target_word)
    return best_word

if __name__ == "__main__":
    words = get_words("EN/x_set.json")

    print(closest_word(words,"apple"))
