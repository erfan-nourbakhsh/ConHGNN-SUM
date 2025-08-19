import os
import json
import nltk
import random
import argparse


def catDoc(textlist):
    """
    Flatten a list of lists of sentences into a single list of sentences.

    :param textlist: list of lists of sentences, e.g., [[sent1, sent2], [sent3, sent4]]
    :return: list of all sentences concatenated
    """
    res = []
    for tlist in textlist:  # iterate over each sublist of sentences
        res.extend(tlist)    # extend the result list with the current sublist
    return res


def PrintInformation(keys, allcnt):
    """
    Prints statistics about word frequencies in a vocabulary.

    :param keys: list of tuples, [(word, frequency), ...] sorted by frequency descending
    :param allcnt: total number of words (sum of all frequencies)
    """
    # Count words with frequency > 10 and their total proportion
    cnt = 0
    first = 0.0
    for key, val in keys:
        if val >= 10:  # only consider words appearing 10 times or more
            cnt += 1
            first += val
    print("appearance > 10 cnt: %d, percent: %f" % (cnt, first / allcnt))

    # Statistics for the first 30,000 most frequent words
    if len(keys) > 30000:
        first = 0.0
        for k, v in keys[:30000]:
            first += v
        print("First 30,000 percent: %f, last freq %d" % (first / allcnt, keys[30000][1]))

    # Statistics for the first 50,000 most frequent words
    if len(keys) > 50000:
        first = 0.0
        for k, v in keys[:50000]:
            first += v
        print("First 50,000 percent: %f, last freq %d" % (first / allcnt, keys[50000][1]))

    # Statistics for the first 100,000 most frequent words
    if len(keys) > 100000:
        first = 0.0
        for k, v in keys[:100000]:
            first += v
        print("First 100,000 percent: %f, last freq %d" % (first / allcnt, keys[100000][1]))


if __name__ == '__main__':
    """
    Main script:
    - Parse arguments
    - Read dataset
    - Extract all words from text and summaries
    - Compute frequency distribution
    - Save vocabulary and print stats
    """
    parser = argparse.ArgumentParser()

    # Input dataset JSONL file
    parser.add_argument('--data_path', type=str, default='data/CNNDM/train.label.jsonl',
                        help='File to deal with')
    # Dataset name
    parser.add_argument('--dataset', type=str, default='CNNDM', help='dataset name')

    args = parser.parse_args()  # parse command-line arguments

    # Directory to save vocabulary file
    save_dir = os.path.join("cache", args.dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    saveFile = os.path.join(save_dir, "vocab")  # path to save vocab
    print("Save vocab of dataset %s to %s" % (args.dataset, saveFile))

    text = []      # will hold text sentences
    summary = []   # will hold summary sentences
    allword = []   # all words from text and summaries
    cnt = 0        # counter for number of examples

    # Read dataset line by line
    with open(args.data_path, encoding='utf8') as f:
        for line in f:
            e = json.loads(line)  # parse JSON line
            # Flatten text if nested
            if isinstance(e["text"], list) and isinstance(e["text"][0], list):
                sents = catDoc(e["text"])
            else:
                sents = e["text"]
            text = " ".join(sents)  # join sentences into a single string

            allword.extend(text.split())  # add words to allword list

            # Process summary if exists
            try:
                summary = " ".join(e["summary"])
                allword.extend(summary.split())
            except:
                pass

            cnt += 1  # increment example count

    print("Training set of dataset has %d example" % cnt)

    # Compute frequency distribution of all words
    fdist1 = nltk.FreqDist(allword)

    # Save vocabulary to file
    fout = open(saveFile, "w")
    keys = fdist1.most_common()  # list of (word, frequency) sorted descending
    for key, val in keys:
        try:
            fout.write("%s\t%d\n" % (key, val))  # save word and count
        except UnicodeEncodeError as e:
            # Skip words that cannot be encoded
            continue

    fout.close()

    allcnt = fdist1.N()  # total number of word occurrences
    allset = fdist1.B()  # total number of unique words
    print("All appearance %d, unique word %d" % (allcnt, allset))

    # Print detailed statistics
    PrintInformation(keys, allcnt)
