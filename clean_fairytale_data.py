# Import the libraries. #
import numpy as np
import pandas as pd
import pickle as pkl

from collections import Counter
from nltk import wordpunct_tokenize

# Load the data. #
with open("../Data/merged_clean.txt", "rb") as tmp_file:
    raw_data = tmp_file.read()

story_data = [str(x) for x in raw_data.decode(
    "utf-8").split("\n\n\n") if x != ""]
story_data = ["\n".join([y.replace(
    "\n", " ").strip() for y in str(
        x).split("\n\n")]) for x in story_data]
print("Total of", len(story_data), "fairytales.")

# Extract the vocabulary. #
w_counter = Counter()
for tmp_story in story_data:
    tmp_tokens = [
        x for x in wordpunct_tokenize(
            str(tmp_story).lower().strip())]
    w_counter.update(tmp_tokens)

# Only use words which occur more than once. #
min_count  = 1
word_vocab = ["[SOS]", "[EOS]", "[UNK]", "[PAD]"]
word_vocab += list(sorted([
    x for x, y in w_counter.most_common() if y > min_count]))

word_2_idx = dict([
    (word_vocab[x], x) for x in range(len(word_vocab))])
idx_2_word = dict([
    (x, word_vocab[x]) for x in range(len(word_vocab))])
print("Vocabulary Size:", len(word_vocab), "words.")

# Generate the training data. #
l_window  = 50
UNK_token = word_2_idx["[UNK]"]
n_verbose = int(len(story_data) / 10)

input_data = []
label_data = []
for m in range(len(story_data)):
    tmp_story  = story_data[m]
    tmp_tokens = ["[SOS]"]
    tmp_tokens += [
        x for x in wordpunct_tokenize(
            str(tmp_story).lower().strip())]
    tmp_tokens += ["[EOS]"]
    num_tokens = len(tmp_tokens)
    
    tmp_tok_id = [word_2_idx.get(
        x, UNK_token) for x in tmp_tokens]
    for n_st in range(0, num_tokens-l_window-1):
        n_en = n_st + l_window + 1
        tmp_seq_text = np.array(
            tmp_tok_id[n_st:n_en], 
            dtype=np.int32).reshape((1, l_window+1))
        
        label_data.append(tmp_seq_text[:, 1:])
        input_data.append(tmp_seq_text[:, :-1])

# Concatenate the arrays. #
input_data = np.concatenate(input_data, axis=0)
label_data = np.concatenate(label_data, axis=0)
print("Input data array:", input_data.shape)
print("Label data array:", label_data.shape)

# Save the data. #
fairytale_data = {
    "word_vocab": word_vocab, 
    "word_2_idx": word_2_idx, 
    "idx_2_word": idx_2_word, 
    "input_array": input_data, 
    "label_array": label_data
}

tmp_pkl_file = "../Data/fairytale_words_data_window_50.pkl"
with open(tmp_pkl_file, "wb") as tmp_save:
    pkl.dump(fairytale_data, tmp_save)
