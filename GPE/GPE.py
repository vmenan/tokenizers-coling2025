import json
import regex as re
import grapheme
import os
import pickle
import unicodedata
import time
from tqdm.auto import tqdm

DUMMY_PREFIX = " "

def calculate_elapsed_time(start_time):
    # Get the current time
    end_time = time.time()

    # Calculate the time difference
    time_difference = end_time - start_time

    # Convert seconds to days, hours, minutes, and seconds
    days, remainder = divmod(time_difference, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    return int(days), int(hours), int(minutes), int(seconds)


def load_txt_files_from_folder(folder_path):
    txt_files = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                lines = [line.strip() for line in lines if len(line.strip()) > 0]
                txt_files.extend(lines)
    return txt_files



# -----------------------------------------------------------------------------
# a few helper functions useful for both BasicTokenizer and RegexTokenizer

def get_stats(ids, counts=None):
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally allows to update an existing dictionary of counts
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]): # iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):
        # if not at the very last position AND the pair matches, replace it
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids



# Function to save a dictionary as a pickle file
def save_dict_to_pickle(dictionary, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(dictionary, file)


with open("./train/samanantar/train/samanantar_train_150K.txt", "r", encoding='utf8') as f:
    lines = f.readlines()
    if DUMMY_PREFIX is not None:
        lines = [DUMMY_PREFIX + re.sub(r'\s+', ' ', line.strip()) for line in lines]
    else:
        lines = [re.sub(r'\s+', ' ', line.strip()) for line in lines]



range_ = len(lines)
lines_limited = lines[:range_]


whitespace_pattern = r"""\s[\p\u0B80-\u0BFF]+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""





intial_gh = []
progress_bar = tqdm(range(len(lines_limited)))
for text in lines_limited:
    flat_list = []
    text_chunks = re.findall(whitespace_pattern, text)
    # Replace leading whitespace with U+2581 and collect the tokens
    text_chunks = [token.replace(' ','\u2581') for  token in text_chunks]
#     print(text_chunks)
    text_chunks = [t  for t in text_chunks if t!=' ' ]
#     print(text_chunks)
    graphemed_ls = [list(grapheme.graphemes(t)) for t in text_chunks]
    flat_list = [
            x
            for ls in graphemed_ls
            for x in ls
    ]
    intial_gh.extend(list(set(flat_list)))
    progress_bar.update()




intial_gh = list(set(intial_gh))


vocab = {idx:intial_gh[idx]  for idx in range(len(intial_gh))}
vocab_re = {intial_gh[idx] :idx for idx in range(len(intial_gh))}

vocab_size = 6_000 - len(vocab)
num_merges = vocab_size



def covert_to_ids_train(texts,pattern):
    ids = []
    progress_bar = tqdm(range(len(texts)))
    for text in texts:
        text_chunks = re.findall(pattern, text)
        text_chunks = [token.replace(' ','\u2581') for  token in text_chunks]
        text_chunks = [t  for t in text_chunks if t!=' ' ]
        graphemed_ls = [list(grapheme.graphemes(t)) for t in text_chunks]
        ids_temp = [list(map(lambda x:vocab_re[x],ls)) for ls in graphemed_ls ]
        ids.extend(ids_temp)
        progress_bar.update()
    return ids

def covert_to_ids(text_chunk):
    graphemed_ls = list(grapheme.graphemes(text_chunk))
    ids = list(map(lambda x:vocab_re[x],graphemed_ls))
    return ids



merges = {}
ids = covert_to_ids_train(lines_limited,whitespace_pattern)
startTime=time.time()
print()
for i in range(num_merges):
    # count the number of times every consecutive pair appears
    stats = {}
    for chunk_ids in ids:
        # passing in stats will update it in place, adding up counts
        get_stats(chunk_ids, stats)
    # find the pair with the highest count
    pair = max(stats, key=stats.get)
    # mint a new token: assign it the next available id
    idx = len(vocab) + i
    # replace all occurrences of pair in ids with idx
    ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]

    # save the merge
    merges[pair] = idx
    vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
    # prints
    if True:
        print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")


days, hours, minutes, _ = calculate_elapsed_time(startTime)
print('Time taken for training : {} days {} hrs {} mints'.format(days, hours, minutes))
print('training finished')

save_dict_to_pickle(merges, r"D:\My Studies\Research\Tokenization\LAT\train\indic_merges\merges.pkl")
save_dict_to_pickle(vocab, r"D:\My Studies\Research\Tokenization\LAT\train\indic_merges\vocab.pkl")
save_dict_to_pickle(vocab_re, r"D:\My Studies\Research\Tokenization\LAT\train\indic_merges\vocab_re.pkl")

