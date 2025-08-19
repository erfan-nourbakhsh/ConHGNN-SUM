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
    filename = path.split("/")[-1]  # get the last component of path (file name)
    return filename.split(".")[0]    # remove extension and return base name


def get_tfidf_embedding(text):
    """
    Converts a list of documents into TF-IDF embeddings.
    
    :param text: list of strings, where each string is a document (sentences concatenated)
    :return: 
        vectorizer: CountVectorizer object with vocabulary mapping
        tfidf_weight: numpy array [num_documents, num_features], TF-IDF weights
    """
    vectorizer = CountVectorizer(lowercase=True)  # initialize vectorizer
    word_count = vectorizer.fit_transform(text)   # convert text to word count matrix

    tfidf_transformer = TfidfTransformer()        # initialize TF-IDF transformer
    tfidf = tfidf_transformer.fit_transform(word_count)  # compute TF-IDF matrix

    tfidf_weight = tfidf.toarray()  # convert sparse matrix to dense numpy array
    return vectorizer, tfidf_weight


def compress_array(a, id2word):
    """
    Compresses a dense TF-IDF array to a dictionary format, ignoring zero values.
    
    :param a: 2D array [num_documents, num_words] of TF-IDF weights
    :param id2word: dict mapping column index to word string
    :return: dict {doc_id: {word: tfidf_value, ...}, ...}
    """
    d = {}
    for i in range(len(a)):  # iterate over documents
        d[i] = {}
        for j in range(len(a[i])):  # iterate over words in document
            if a[i][j] != 0:  # only keep non-zero TF-IDF values
                d[i][id2word[j]] = a[i][j]
    return d


def main():
    """
    Main function for generating TF-IDF word-to-document features for a dataset.
    - Reads JSONL file
    - Computes TF-IDF for each document
    - Saves compressed TF-IDF vectors as JSONL
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/CNNDM/train.label.jsonl', 
                        help='File to deal with')
    parser.add_argument('--dataset', type=str, default='CNNDM', help='Dataset name')
    args = parser.parse_args()  # parse command-line arguments

    # Directory to save preprocessed TF-IDF features
    save_dir = os.path.join("cache", args.dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fname = GetType(args.data_path) + ".w2d.tfidf.jsonl"  # output file name
    saveFile = os.path.join(save_dir, fname)
    print("Save word2sent features of dataset %s to %s" % (args.dataset, saveFile))

    fout = open(saveFile, "w")  # open output file for writing

    # Read dataset line by line
    with open(args.data_path) as f:
        for line in f:
            e = json.loads(line)  # parse JSON line
            # If text is a list of lists (multi-sentence documents), join sentences
            if isinstance(e["text"], list) and isinstance(e["text"][0], list):
                docs = [" ".join(doc) for doc in e["text"]]
            else:
                docs = [e["text"]]  # single-document case

            # Compute TF-IDF embeddings
            cntvector, tfidf_weight = get_tfidf_embedding(docs)

            # Build id2word mapping from CountVectorizer
            id2word = {tfidf_id: w for w, tfidf_id in cntvector.vocabulary_.items()}

            # Compress dense TF-IDF array to dictionary
            tfidfvector = compress_array(tfidf_weight, id2word)

            # Write each document's TF-IDF dictionary as a JSON line
            fout.write(json.dumps(tfidfvector) + "\n")


if __name__ == '__main__':
    main()  # run main function when script is executed
