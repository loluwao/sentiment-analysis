# STEP 1: rename this file to textclassify_model.py

# feel free to include more imports as needed here
# these are the ones that we used for the base model
import numpy as np
import sys
from collections import Counter
import math

"""
Your name and file comment here:
"""

"""
Cite your sources here:
"""

"""
Implement your functions that are not methods of the TextClassify class here
"""


def generate_tuples_from_file(training_file_path):
    """
    Generates tuples from file formated like:
    id\ttext\tlabel
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
    pass


def recall(gold_labels, predicted_labels):
    """
    Calculates the recall for a set of predicted labels give the gold (ground truth) labels.
    Parameters:
        gold_labels (list): a list of labels assigned by hand ("truth")
        predicted_labels (list): a corresponding list of labels predicted by the system
    Returns: double recall (a number from 0 to 1)
    """
    pass


def f1(gold_labels, predicted_labels):
    """
    Calculates the f1 for a set of predicted labels give the gold (ground truth) labels.
    Parameters:
        gold_labels (list): a list of labels assigned by hand ("truth")
        predicted_labels (list): a corresponding list of labels predicted by the system
    Returns: double f1 (a number from 0 to 1)
    """
    pass


"""
Implement any other non-required functions here
"""

"""
implement your TextClassify class here
"""


class TextClassify:

    def __init__(self):
        # do whatever you need to do to set up your class here
        self.data = None  # list of tuples

        self.positive_toks = {}
        self.negative_toks = {}
        self.pos_ids = []
        self.neg_ids = []

        self.pos_prior, self.neg_prior = None, None

        self.vocab = [] # unique tokens

        self.pos_probs = {}
        self.neg_probs = {}

        self.positive_word_count, self.negative_word_count = 0, 0

    def train(self, examples):
        """
        Trains the classifier based on the given examples
        Parameters:
          examples - a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
        Return: None
        """
        positive_txt = []
        negative_txt = []

        for example in examples:
            if example[2] == '1':  # positive review
                positive_txt += example[1].split()
                self.pos_ids.append(example[0])
            elif example[2] == '0':
                negative_txt += example[1].split()
                self.neg_ids.append(example[0])

        self.positive_word_count = len(positive_txt)
        self.negative_word_count = len(negative_txt)
        self.vocab = list(dict(Counter(negative_txt + positive_txt)))
        '''
        print(positive_txt)
        print(negative_txt)
        '''
        # token counts for negative and positive reviews
        self.positive_toks = dict(Counter(positive_txt))
        self.negative_toks = dict(Counter(negative_txt))

        # created vocab list from sorted tokens
        #self.vocab = [*set(list(self.positive_toks.keys()) + list(self.negative_toks.keys()))]
        print("Vocab: " + str(self.vocab))

        # calculate prior probabilities
        self.pos_prior = len(self.pos_ids) / len(examples)
        self.neg_prior = len(self.neg_ids) / len(examples)

        print("0 Prior: " + str(self.pos_prior))
        print("1 Prior: " + str(self.neg_prior))

        # calculate each token's probability
        for token in self.vocab:
            # set unseen words for each category to 0
            if token not in positive_txt:
                self.positive_toks[token] = 0
            elif token not in negative_txt:
                self.negative_toks[token] = 0
            '''
            self.pos_probs[token] = math.log10(self.positive_toks[token]
                                               / (self.positive_word_count + len(self.vocab)))
            '''


    def score(self, data):
        """
        Score a given piece of text
        Parameters:
          data - str like "I loved the hotel"
        Return: dict of class: score mappings
        """
        scores = {'0': 1.0, '1': 1.0}

        data_ls = data.split()

        # calculations for each token in the data
        pos_calc, neg_calc = [], []

        for tok in data_ls:
          if tok in self.vocab:
            pos_calc.append((self.positive_toks[tok] + 1) / (self.positive_word_count + len(self.vocab)))
            neg_calc.append((self.negative_toks[tok] + 1) / (self.negative_word_count + len(self.vocab)))

        for x in pos_calc:
          scores['1'] *= x
        for x in neg_calc:
          scores['0'] *= x

        scores['1'] *= self.pos_prior
        scores['0'] *= self.neg_prior

        print(scores)

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

    def featurize(self, data):
        """
        we use this format to make implementation of your TextClassifyImproved model more straightforward and to be
        consistent with what you see in nltk
        Parameters:
          data - str like "I loved the hotel"
        Return: a list of tuples linking features to values
        for BoW, a list of tuples linking every word to True [("I", True), ("loved", True), ("it", True)]
        """
        pass

    def __str__(self):
        return "Naive Bayes - bag-of-words baseline"


class TextClassifyImproved:
    '''
    Normalize text in 2 ways:
      change "not good" to not_good in txt
      make words w one capital letter just lowercase
    '''
    def __init__(self):
        pass

    def train(self, examples):
        """
        Trains the classifier based on the given examples
        Parameters:
          examples - a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
        Return: None
        """
        pass

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
        pass

    def classify(self, data):
        """
        Label a given piece of text
        Parameters:
          data - str like "I loved the hotel"
        Return: string class label
        """
        pass

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
        return "NAME OF YOUR CLASSIFIER HERE"

    def describe_experiments(self):
        s = """
    Description of your experiments and their outcomes here.
    """
        return s


def main():
    training = sys.argv[1]
    testing = sys.argv[2]

    # print(generate_tuples_from_file(training))

    classifier = TextClassify()
    classifier.train(generate_tuples_from_file(training))
    #classifier.score("I hated the hotel")
    print(classifier.classify("I loved the hotel"))
    print(classifier)
    # do the things that you need to with your base class

    # report precision, recall, f1

    improved = TextClassifyImproved()
    print(improved)
    # do the things that you need to with your improved class

    # report final precision, recall, f1 (for your best model)

    # report a summary of your experiments/features here
    print(improved.describe_experiments())


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:", "python textclassify_model.py training-file.txt testing-file.txt")
        sys.exit(1)

    main()
