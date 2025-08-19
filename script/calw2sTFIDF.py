import os
import argparse
import json

from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def GetType(path):
    """
    Extracts the base name (without extension) from a file path.
    :param path: str, path to a file
    :return: str, file name without extension
    """
    filename = path.split("/")[-1]  # take the last part of the path (the file name)
    return filename.split(".")[0]    # remove file extension and return base name


def catDoc(textlist):
    """
    Concatenates a list of lists of sentences into a single list of sentences.
    :param textlist: list of lists, e.g., [[sent1, sent2], [sent3, sent4]]
    :return: list, flattened list of sentences
    """
    res = []
    for tlist in textlist:  # iterate over each sublist
        res.extend(tlist)    # add all elements of sublist to result
    return res


def get_tfidf_embedding(text):
    """
    Computes the TF-IDF embeddings for a list of sentences.
    
    :param text: list of str, sentences/documents
    :return: 
        vectorizer: CountVectorizer object, contains vocabulary mapping word->id
        tfidf_weight: np.array, dense TF-IDF matrix [num_sentences, num_words]
    """
    vectorizer = CountVectorizer(lowercase=True)  # initialize word count vectorizer
    word_count = vectorizer.fit_transform(text)   # convert text to sparse word count matrix
    tfidf_transformer = TfidfTransformer()        # initialize TF-IDF transformer
    tfidf = tfidf_transformer.fit_transform(word_count)  # convert counts to TF-IDF
    tfidf_weight = tfidf.toarray()  # convert sparse matrix to dense numpy array
    return vectorizer, tfidf_weight


def compress_array(a, id2word):
    """
    Converts a dense TF-IDF matrix to a dictionary format, keeping only non-zero values.
    
    :param a: 2D array [num_docs, num_words], TF-IDF matrix
    :param id2word: dict, maps word index to word string
    :return: dict, {doc_id: {word: tfidf_value, ...}, ...}
    """
    d = {}
    for i in range(len(a)):  # iterate over each document
        d[i] = {}
        for j in range(len(a[i])):  # iterate over each word in document
            if a[i][j] != 0:  # store only non-zero TF-IDF values
                d[i][id2word[j]] = a[i][j]
    return d


def main():
    """
    Main script function:
    - Parses input arguments
    - Reads JSONL dataset
    - Computes TF-IDF embeddings per document
    - Saves compressed TF-IDF vectors in JSONL format
    """
    parser = argparse.ArgumentParser()

    # Input file path
    parser.add_argument('--data_path', type=str, default='data/CNNDM/train.label.jsonl', 
                        help='File to deal with')
    # Dataset name
    parser.add_argument('--dataset', type=str, default='CNNDM', help='dataset name')

    args = parser.parse_args()  # parse command-line arguments

    # Directory to save TF-IDF features
    save_dir = os.path.join("cache", args.dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Output file name
    fname = GetType(args.data_path) + ".w2s.tfidf.jsonl"
    saveFile = os.path.join(save_dir, fname)
    print("Save word2sent features of dataset %s to %s" % (args.dataset, saveFile))

    fout = open(saveFile, "w")  # open output file

    # Process dataset line by line
    with open(args.data_path) as f:
        for line in f:
            e = json.loads(line)  # parse JSON line
            # Flatten list of sentences if nested
            if isinstance(e["text"], list) and isinstance(e["text"][0], list):
                sents = catDoc(e["text"])
            else:
                sents = e["text"]  # single-level list

            # Compute TF-IDF embeddings
            cntvector, tfidf_weight = get_tfidf_embedding(sents)

            # Build id->word mapping
            id2word = {}
            for w, tfidf_id in cntvector.vocabulary_.items():  # word -> column index
                id2word[tfidf_id] = w

            # Compress TF-IDF array into dictionary
            tfidfvector = compress_array(tfidf_weight, id2word)

            # Write JSON line
            fout.write(json.dumps(tfidfvector) + "\n")


if __name__ == '__main__':
    main()  # execute main function
