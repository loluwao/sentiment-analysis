# STEP 1: rename this file to textclassify_model.py

# feel free to include more imports as needed here
# these are the ones that we used for the base model
import numpy as np
import sys
from collections import Counter

"""
Your name and file comment here:
Temi Akinyoade, textclassify_model.py
"""

"""
Cite your sources here:
stopwords: https://gist.github.com/sebleier/554280
positive and negative sentiment words: https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html
"""

"""
Implement your functions that are not methods of the TextClassify class here
"""

negatives = ["didn't", "not", "no", "none"]


def generate_tuples_from_file(training_file_path):
    """
    Generates tuples from file formated like:
    id\text\tlabel
    Parameters:
      training_file_path - str path to file to read in
    Return:
      a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
    """
    f = open(training_file_path, "r", encoding="utf8")
    listOfExamples = []
    for review in f:
        if len(review.strip()) == 0:
            continue
        dataInReview = review.split("\t")
        for i in range(len(dataInReview)):
            # remove any extraneous whitespace
            dataInReview[i] = dataInReview[i].strip()
        t = tuple(dataInReview)
        listOfExamples.append(t)
    f.close()
    return listOfExamples


def precision(gold_labels, predicted_labels):
    """
    Calculates the precision for a set of predicted labels give the gold (ground truth) labels.
    Parameters:
        gold_labels (list): a list of labels assigned by hand ("truth")
        predicted_labels (list): a corresponding list of labels predicted by the system
    Returns: double precision (a number from 0 to 1)
    """
    true_pos = 0
    false_pos = 0
    for i in range(len(gold_labels)):
        # add up true positives (predicted and gold are True)
        if gold_labels[i] == '1' and predicted_labels[i] == '1':
            true_pos += 1
        # add up false positives (system wrongly predicted True)
        elif gold_labels[i] == '0' and predicted_labels[i] == '1':
            false_pos += 1

    return true_pos / (true_pos + false_pos)


def recall(gold_labels, predicted_labels):
    """
    Calculates the recall for a set of predicted labels give the gold (ground truth) labels.
    Parameters:
        gold_labels (list): a list of labels assigned by hand ("truth")
        predicted_labels (list): a corresponding list of labels predicted by the system
    Returns: double recall (a number from 0 to 1)
    """
    true_pos = 0
    false_neg = 0
    for i in range(len(gold_labels)):
        # add up true positives (predicted and gold are True)
        if gold_labels[i] == '1' and predicted_labels[i] == '1':
            true_pos += 1
        # add up false negatives (system wrongly predicted True)
        elif gold_labels[i] == '1' and predicted_labels[i] == '0':
            false_neg += 1

    return true_pos / (false_neg + true_pos)


def f1(gold_labels, predicted_labels):
    # calc precision and recall
    p = precision(gold_labels, predicted_labels)
    r = recall(gold_labels, predicted_labels)

    if p + r == 0:
        return 0

    return 2 * p * r / (p + r)


class TextClassify:

    def __init__(self):
        # do whatever you need to do to set up your class here
        self.data = None  # list of tuples

        self.positive_toks = {}
        self.negative_toks = {}
        self.pos_ids = []
        self.neg_ids = []

        self.pos_prior, self.neg_prior = None, None

        self.vocab = []  # unique tokens

        self.pos_probs = {}
        self.neg_probs = {}

        self.positive_word_count, self.negative_word_count = 0, 0

        # gold_labels: correct labels, predicted labels: labels calculated by my model
        self.gold_labels = []
        self.predicted_labels = []

    def train(self, examples):
        positive_txt = []
        negative_txt = []

        self.data = examples
        for example in examples:
            if example[2] == '1':  # positive review
                positive_txt += example[1].split()
                self.pos_ids.append(example[0])
            elif example[2] == '0':
                negative_txt += example[1].split()
                self.neg_ids.append(example[0])

        # count pos and neg vocabs
        self.positive_word_count = len(positive_txt)
        self.negative_word_count = len(negative_txt)
        self.vocab = list(dict(Counter(negative_txt + positive_txt)))

        # token counts for negative and positive reviews
        self.positive_toks = dict(Counter(positive_txt))
        self.negative_toks = dict(Counter(negative_txt))

        # calculate prior probabilities
        self.pos_prior = len(self.pos_ids) / len(examples)
        self.neg_prior = len(self.neg_ids) / len(examples)

    def score(self, data):
        scores = {'0': 1.0, '1': 1.0}

        data_ls = data.split()

        # calculations for each token in the data
        pos_calc, neg_calc = [], []
        numerator = 0

        # find positive and negative probabilities for each token in data/"line"
        for tok in data_ls:
            if tok in self.vocab:
                if self.positive_toks.get(tok) is None:
                    numerator = 0
                else:
                    numerator = self.positive_toks[tok]
                pos_calc.append((numerator + 1) / (self.positive_word_count + len(self.vocab)))

                if self.negative_toks.get(tok) is None:
                    numerator = 0
                else:
                    numerator = self.negative_toks[tok]
                neg_calc.append((numerator + 1) / (self.negative_word_count + len(self.vocab)))

        for x in pos_calc:
            scores['1'] *= x
        for x in neg_calc:
            scores['0'] *= x

        # multiply total P(w | c) probabilities by priors
        scores['1'] *= self.pos_prior
        scores['0'] *= self.neg_prior

        return scores

    def classify(self, data):
        scores = self.score(data)

        # return key with maximum score
        return max(scores, key=scores.get)

    def classify_all(self):
        classifications = {}
        # go thru every example
        # create dictionary of {key: data, value: class}
        for tup in self.data:
            # fill gold labels
            self.gold_labels += tup[2]
            pred_score = self.classify(tup[1])
            self.predicted_labels += pred_score
            classifications[tup[1]] = pred_score

        return classifications

    def featurize(self, data):
        return [(word, True) for word in data]

    def __str__(self):
        return "Naive Bayes - bag-of-words baseline"


class TextClassifyImproved:
    def __init__(self):
        self.data = []  # list of tuples

        # all tokens seen in positive and negative reviews
        self.positive_toks = {}
        self.negative_toks = {}

        self.pos_ids = []
        self.neg_ids = []

        self.pos_prior, self.neg_prior = None, None

        self.vocab = []  # unique tokens

        self.pos_probs = {}
        self.neg_probs = {}

        self.positive_word_count, self.negative_word_count = 0, 0

        self.gold_labels = []
        self.predicted_labels = []

        self.pos_lex = []
        self.neg_lex = []

        self.stopwords = []

    def load_neg_lex(self):
        """
        Reads in all negative words from word list into an array.
        """
        f = open("negative-words.txt", "r")
        words = f.read().splitlines()
        f.close()
        self.neg_lex = [word for word in words]

    def load_pos_lex(self):
        """
        Reads in all positive words from word list into an array.
        """
        f = open("positive-words.txt", "r")
        words = f.read().splitlines()
        f.close()
        self.pos_lex = [word for word in words]

    def load_stopwords(self):
        """
        Read in all stop words from word list into an array.
        """
        f = open("stopwords.txt", "r")
        self.stopwords = f.read().splitlines()
        f.close()

    def not_helper(self, text):
        """
        Modifies any given text in cases of negation.
        ex: "I'm not happy" -> "I'm NOT_happy", so "not" and "happy" become one token.
        ... "I didn't enjoy this" -> "I NOT_enjoy this"
        """

        # split text into a list and have new list to return
        text_ls = text.split()
        modified_ls = []

        # find all negation words in string
        for i in range(len(text_ls)):
            if text_ls[i].lower() in negatives:
                if i < len(text_ls) - 1:
                    modified_ls.append("NOT_" + text_ls[i + 1])
                else:
                    modified_ls.append(text_ls[i])  # add not
            else:
                if i > 0:
                    if text_ls[i - 1].lower() not in negatives:
                        modified_ls.append(text_ls[i])  # add normal word
                else:  # only in the case of the first word
                    modified_ls.append(text_ls[i])

        return modified_ls

    def normalize_helper(self, text):
        '''
        Extra text normalization.
        - removes punctuation I deemed as unimportant or not indicative
        - modifies awkward or first-word capitalization
        "(2007)" -> "2007"
        ""wow!"" -> "wow"
        "Hello" -> "hello"
        "BAD" -> "BAD" (doesn't change because capitalization may be a sentiment indicator)
        '''

        text_ls = text.split()
        mod_word = ""
        new_text = []
        irrelevant_characters = ['.', ',', '!', '?', '\"', ')', '(']
        for word in text_ls:
            # remove punctuation?
            mod_word = word
            for char in irrelevant_characters:
                mod_word = mod_word.replace(char, '')

            # normalize capitalization
            if not mod_word.isupper():
                mod_word = mod_word.lower()
            new_text.append(mod_word)

        return " ".join(new_text)

    def train(self, examples):
        positive_txt = []
        negative_txt = []

        self.load_pos_lex()
        self.load_neg_lex()
        self.load_stopwords()
        print("done loading words")

        # iterate thru all examples and sort normalized text into object's data
        for example in examples:
            if example[2] == '1':  # positive review
                normalized_txt = self.not_helper(self.normalize_helper(example[1]))
                positive_txt += normalized_txt
                self.pos_ids.append(example[0])
                self.data.append((example[0], " ".join(normalized_txt), example[2]))
            elif example[2] == '0':
                normalized_txt = self.not_helper(self.normalize_helper(example[1]))
                positive_txt += normalized_txt
                self.neg_ids.append(example[0])
                self.data.append((example[0], " ".join(normalized_txt), example[2]))

        # count pos and neg vocabs
        self.positive_word_count = len(positive_txt)
        self.negative_word_count = len(negative_txt)
        self.vocab = list(dict(Counter(negative_txt + positive_txt)))

        # remove stop words from vocab
        for word in self.stopwords:
            if word in self.vocab:
                self.vocab.remove(word)

        # token counts for negative and positive reviews
        self.positive_toks = dict(Counter(positive_txt))
        self.negative_toks = dict(Counter(negative_txt))

        # calculate prior probabilities
        self.pos_prior = len(self.pos_ids) / len(examples)
        self.neg_prior = len(self.neg_ids) / len(examples)

    def score(self, data):
        """
        Score a given piece of text
        you’ll compute e ^ (log(p(c)) + sum(log(p(w_i | c))) here

        Parameters:
          data - str like "I loved the hotel"
        Return: dict of class: score mappings
        return a dictionary of the values of P(data | c)  for each class,
        as in section 4.3 of the textbook e.g. {"0": 0.000061, "1": 0.000032}
        """
        scores = {'0': 1.0, '1': 1.0}

        # split tokens of data item into a list of Strings
        data_ls = data.split()

        # calculations for each token in the data
        pos_calc, neg_calc = [], []

        # iterate thru each token
        for tok in data_ls:
            if tok in self.vocab:
                if self.positive_toks.get(tok) is None:
                    numerator = 0
                else:
                    numerator = self.positive_toks[tok]
                pos_calc.append((numerator + 1) / (self.positive_word_count + len(self.vocab)))

                if self.negative_toks.get(tok) is None:
                    numerator = 0
                else:
                    numerator = self.negative_toks[tok]
                neg_calc.append((numerator + 1) / (self.negative_word_count + len(self.vocab)))

        # calc BoW probabilities
        bow_pos = 1
        bow_neg = 1
        for x in pos_calc:
            bow_pos *= x
        for x in neg_calc:
            bow_neg *= x

        # count positive and negative words and map in dictionaries
        data_pos_lex = []
        data_neg_lex = []

        for word in data_ls:
            if word in self.pos_lex:
                data_pos_lex.append(word)

        for word in data_ls:
            if word in self.neg_lex:
                data_neg_lex.append(word)

        num_pos_words = len(data_pos_lex)
        num_neg_words = len(data_neg_lex)

        # calculate total probability
        total_pos_prob, total_neg_prob = 0, 0

        if num_pos_words == 0:
            total_pos_prob = np.log(bow_pos) + np.log(self.pos_prior)
        else:
            total_pos_prob = np.log(bow_pos) + np.log(num_pos_words / (num_pos_words + num_neg_words)) + np.log(
                self.pos_prior)

        if num_neg_words == 0:
            total_neg_prob = np.log(bow_neg) + np.log(self.neg_prior)
        else:
            total_neg_prob = np.log(bow_neg) + np.log(num_neg_words / (num_pos_words + num_neg_words)) + np.log(
                self.neg_prior)

        scores['1'] = np.exp(total_pos_prob)
        scores['0'] = np.exp(total_neg_prob)

        return scores

    def classify(self, data):
        """
        Label a given piece of text
        Parameters:
          data - str like "I loved the hotel"
        Return: string class label
        """
        scores = self.score(data)
        return max(scores, key=scores.get)

    def classify_all(self):
        classifications = {}
        # go thru every example
        for tup in self.data:
            # fill gold labels
            self.gold_labels += tup[2]
            pred_score = self.classify(tup[1])
            self.predicted_labels += pred_score
            classifications[tup[1]] = pred_score

        return classifications

    def featurize(self, data):
        """
        we use this format to make implementation of this class more straightforward and to be
        consistent with what you see in nltk
        Parameters:
          data - str like "I loved the hotel"
        Return: a list of tuples linking features to values
        for BoW, a list of tuples linking every word to True [("I", True), ("loved", True), ("it", True)]
        """
        pass

    def __str__(self):
        return "improved text classifier!"

    def describe_experiments(self):
        s = """
            The first thing I did was extra text normalization. This had a very minimal but observable
            affect on the F1 score, as it increased by around 0.02 from the baseline model.
            Then, I implemented the counts of positive or negative words. This had a very big affect on
            the F1 score. This time it increased by at least 0.2.
            The last thing I did was remove all stop words. This also had a minimal but positive affect
            on my F1 score.
            """
        return s


def main():
    training = sys.argv[1]
    testing = sys.argv[2]

    classifier = TextClassify()
    print("training")
    classifier.train(generate_tuples_from_file(training))
    print(classifier)
    # report precision, recall, f1

    print("classifying each sample")
    weak_class = classifier.classify_all()
    print("calculating precision")
    p1 = precision(classifier.gold_labels, classifier.predicted_labels)
    print("calculating recall")
    r1 = recall(classifier.gold_labels, classifier.predicted_labels)
    print("calculating F1 score")
    f11 = f1(classifier.gold_labels, classifier.predicted_labels)

    print("precision: " + str(p1))
    print("recall: " + str(r1))
    print("f1: " + str(f11) + "\n")

    # better classifier
    better_classifier = TextClassifyImproved()
    print(better_classifier)
    print("training...")
    better_classifier.train(generate_tuples_from_file(training))

    # report precision, recall, f1
    print("classifying all...")
    goat_class = better_classifier.classify_all()
    p1 = precision(better_classifier.gold_labels, better_classifier.predicted_labels)
    r1 = recall(better_classifier.gold_labels, better_classifier.predicted_labels)
    f11 = f1(better_classifier.gold_labels, better_classifier.predicted_labels)

    print("\nprecision: " + str(p1))
    print("recall: " + str(r1))
    print("f1: " + str(f11))

    # report a summary of your experiments/features here
    print(better_classifier.describe_experiments())


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:", "python textclassify_model.py training-file.txt testing-file.txt")
        sys.exit(1)

    main()
