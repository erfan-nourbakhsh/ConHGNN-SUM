import os
import json
import argparse
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def catDoc(textlist):
    """
    Flatten a list of lists of sentences into a single list of sentences.
    
    :param textlist: list of lists of sentences, e.g., [[sent1, sent2], [sent3, sent4]]
    :return: list of all sentences concatenated
    """
    res = []
    for tlist in textlist:  # iterate over each sublist of sentences
        res.extend(tlist)    # add each sentence from the sublist to the result
    return res


def calTFidf(text):
    """
    Compute TF-IDF matrix for a list of text documents.
    
    :param text: list of strings, each string is a document
    :return: 
        vectorizer: CountVectorizer object containing vocabulary
        tfidf_matrix: TF-IDF sparse matrix of shape [num_docs, num_words]
    """
    vectorizer = CountVectorizer(lowercase=True)        # Convert documents to word count matrix
    wordcount = vectorizer.fit_transform(text)          # Fit and transform the text into counts
    tf_idf_transformer = TfidfTransformer()             # Initialize TF-IDF transformer
    tfidf_matrix = tf_idf_transformer.fit_transform(wordcount)  # Transform counts to TF-IDF
    return vectorizer, tfidf_matrix


def main():
    """
    Main script:
    - Parse command-line arguments
    - Read dataset JSONL file
    - Flatten and join text
    - Compute TF-IDF
    - Sort words by average TF-IDF
    - Save low TF-IDF words to a file
    """
    parser = argparse.ArgumentParser()

    # Path to dataset file
    parser.add_argument('--data_path', type=str, default='data/CNNDM/train.label.jsonl', 
                        help='File to deal with')
    # Name of dataset
    parser.add_argument('--dataset', type=str, default='CNNDM', help='dataset name')

    args = parser.parse_args()  # parse command-line arguments

    # Directory to save output
    save_dir = os.path.join("cache", args.dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    saveFile = os.path.join(save_dir, "filter_word.txt")  # file to save low TF-IDF words
    print("Save low tfidf words in dataset %s to %s" % (args.dataset, saveFile))

    documents = []  # list to hold processed text documents

    # Read dataset and flatten nested text if needed
    with open(args.data_path, "r", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)  # parse JSON line
            if isinstance(e["text"], list) and isinstance(e["text"][0], list):
                text = catDoc(e["text"])  # flatten nested lists of sentences
            else:
                text = e["text"]
            documents.append(" ".join(text))  # join sentences into a single string

    # Compute TF-IDF
    vectorizer, tfidf_matrix = calTFidf(documents)
    print("The number of example is %d, and the TFIDF vocabulary size is %d" % 
          (len(documents), len(vectorizer.vocabulary_)))

    # Compute average TF-IDF score for each word across all documents
    word_tfidf = np.array(tfidf_matrix.mean(0))  
    del tfidf_matrix  # free memory

    # Get indices of words sorted by average TF-IDF (ascending)
    word_order = np.argsort(word_tfidf[0])  

    # Map indices back to words
    id2word = vectorizer.get_feature_names()  

    # Save words sorted by TF-IDF to file
    with open(saveFile, "w") as fout:
        for idx in word_order:
            w = id2word[idx]             # get word by index
            string = w + "\n"            # prepare string to write
            try:
                fout.write(string)       # write word to file
            except:
                pass                    # skip any encoding errors


if __name__ == '__main__':
    main()
