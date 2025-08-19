from tools.logger import *

# Special tokens used in sequence-to-sequence tasks
PAD_TOKEN = '[PAD]'       # Padding token; used to pad encoder/decoder inputs and target sequences
UNKNOWN_TOKEN = '[UNK]'   # Unknown token; represents out-of-vocabulary (OOV) words
START_DECODING = '[START]'# Start-of-sequence token; added at the beginning of decoder input
STOP_DECODING = '[STOP]'  # Stop-of-sequence token; added at the end of target sequences

# Note: None of [PAD], [UNK], [START], [STOP] should appear in the vocab file.

class Vocab(object):
    """Vocabulary class for mapping between words and their integer ids."""

    def __init__(self, vocab_file, max_size):
        """
        Initialize the vocabulary by reading words from a file and assigning ids.
        
        :param vocab_file: Path to the vocab file, expected format "<word> <frequency>" per line.
        :param max_size: Maximum number of words in the vocab. If 0, read all words.
        """
        self._word_to_id = {}  # Maps word (str) -> id (int)
        self._id_to_word = {}  # Maps id (int) -> word (str)
        self._count = 0        # Counter for total words added

        # Add special tokens first with fixed ids
        for w in [PAD_TOKEN, UNKNOWN_TOKEN, START_DECODING, STOP_DECODING]:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1

        # Read vocab file and add words up to max_size
        with open(vocab_file, 'r', encoding='utf8') as vocab_f:  # ensure UTF-8 encoding
            cnt = 0  # line counter
            for line in vocab_f:
                cnt += 1
                pieces = line.split("\t")  # split by tab (or use space if needed)
                w = pieces[0]  # the word itself

                # Raise error if special tokens appear in vocab file
                if w in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
                    raise Exception(
                        '[UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w
                    )

                # Skip duplicated words and log error
                if w in self._word_to_id:
                    logger.error('Duplicated word in vocabulary file Line %d : %s' % (cnt, w))
                    continue

                # Add word to vocab
                self._word_to_id[w] = self._count
                self._id_to_word[self._count] = w
                self._count += 1

                # Stop if max_size reached
                if max_size != 0 and self._count >= max_size:
                    logger.info(
                        "[INFO] max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (
                            max_size, self._count))
                    break

        logger.info("[INFO] Finished constructing vocabulary of %i total words. Last word added: %s",
                    self._count, self._id_to_word[self._count - 1])

    def word2id(self, word):
        """
        Get the id of a word.
        Returns the id of UNKNOWN_TOKEN if the word is not in vocab.
        """
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id2word(self, word_id):
        """
        Get the word corresponding to an id.
        Raises ValueError if the id does not exist.
        """
        if word_id not in self._id_to_word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id_to_word[word_id]

    def size(self):
        """Return the total number of words in the vocabulary."""
        return self._count

    def word_list(self):
        """Return a list of all words in the vocabulary."""
        return self._word_to_id.keys()

    def get_subscription_words(self, words):
        """
        Return words and their ids that are not in the given list 'words'.
        
        :param words: list of words to exclude
        :return: tuple (list of words, list of ids) not in 'words'
        """
        result_words = []
        result_ids = []
        for item, id_ in self._word_to_id.items():
            if item not in words:
                result_words.append(item)
                result_ids.append(id_)
        return result_words, result_ids
