# STEP 1: rename this file to textclassify_model.py

# feel free to include more imports as needed here
# these are the ones that we used for the base model
import numpy as np
import sys
from collections import Counter
import math
import nltk

"""
Your name and file comment here:
"""

"""
Cite your sources here:
"""

"""
Implement your functions that are not methods of the TextClassify class here
"""

negatives = ["didn't", "not", "no", "none"]


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
    """
    Calculates the f1 for a set of predicted labels give the gold (ground truth) labels.
    Parameters:
        gold_labels (list): a list of labels assigned by hand ("truth")
        predicted_labels (list): a corresponding list of labels predicted by the system
    Returns: double f1 (a number from 0 to 1)
    """
    # calc precision and recall
    p = precision(gold_labels, predicted_labels)
    r = recall(gold_labels, predicted_labels)

    if p + r == 0:
        return 0

    return 2 * p * r / (p + r)


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

        self.vocab = []  # unique tokens

        self.pos_probs = {}
        self.neg_probs = {}

        self.positive_word_count, self.negative_word_count = 0, 0

        self.gold_labels = []
        self.predicted_labels = []

    def train(self, examples):
        """
        Trains the classifier based on the given examples
        Parameters:
          examples - a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
        Return: None
        """
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

        # created vocab list from sorted tokens
        # self.vocab = [*set(list(self.positive_toks.keys()) + list(self.negative_toks.keys()))]
        # print("Vocab: " + str(self.vocab))

        # calculate prior probabilities
        self.pos_prior = len(self.pos_ids) / len(examples)
        self.neg_prior = len(self.neg_ids) / len(examples)

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
        numerator = 0

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

        scores['1'] *= self.pos_prior
        scores['0'] *= self.neg_prior

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


    # THIS ONE IS MULTINOMIAL
    def featurize(self, data):
        """
        we use this format to make implementation of your TextClassifyImproved model more straightforward and to be
        consistent with what you see in nltk
        Parameters:
          data - str like "I loved the hotel"
        Return: a list of tuples linking features to values
        for BoW, a list of tuples linking every word to True [("I", True), ("loved", True), ("it", True)]
        """
        return [(word, True) for word in data]


    def __str__(self):
        return "Naive Bayes - bag-of-words baseline"


class TextClassifyImproved:
    '''
    Normalize text in 2 ways:
      change "not good" to not_good in txt
      make words w one capital letter just lowercase
    '''

    def __init__(self):
        self.data = []  # list of tuples

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

    def not_helper(self, text):
        '''
        Modifies strings that contain "not" so that "not good" becomes "NOT_good"
        :param text:
        :return: String with not-normalization
        '''
        # split text into a list
        text_ls = text.split()
        modified_ls = []
        # find "not" or other negation if in string
        not_on = False
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
        Normalizes the text by modifying capital letters, removing periods
        remove commas
        shave spaces
        :param text:
        :return:
        '''
        print(text)

        text_ls = text.split()
        mod_word = ""
        new_text = []
        # text_ls = nltk.word_tokenize(text)
        irrelevant_characters = ['.', ',', '!', '?', '\"', ')', '(']
        for word in text_ls:
            # remove punctuation?
            mod_word = word
            for char in irrelevant_characters:
                mod_word = mod_word.replace(char, '')
            '''
            mod_word = word.replace('.', '')
            mod_word = mod_word.replace(',', '')
            mod_word = mod_word.replace('!', '')
            mod_word = mod_word.replace('?', '')
            mod_word = mod_word.replace('\"', '')
            '''

            # normalize capitalization
            if not mod_word.isupper():
                mod_word = mod_word.lower()
            new_text.append(mod_word)

        return " ".join(new_text)

    def train(self, examples):
        positive_txt = []
        negative_txt = []

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

        # token counts for negative and positive reviews
        self.positive_toks = dict(Counter(positive_txt))
        self.negative_toks = dict(Counter(negative_txt))

        # created vocab list from sorted tokens
        # self.vocab = [*set(list(self.positive_toks.keys()) + list(self.negative_toks.keys()))]
        # print("Vocab: " + str(self.vocab))

        # calculate prior probabilities
        self.pos_prior = len(self.pos_ids) / len(examples)
        self.neg_prior = len(self.neg_ids) / len(examples)
        # make this "into the appropriate features"

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

        data_ls = data.split()

        # calculations for each token in the data
        pos_calc, neg_calc = [], []

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

        scores['1'] *= self.pos_prior
        scores['0'] *= self.neg_prior

        # print(scores)

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
            print("DONE - " + tup[1])

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
    Description of your experiments and their outcomes here.
    """
        return s


def main():
    training = sys.argv[1]
    testing = sys.argv[2]

    # print(generate_tuples_from_file(training))
    '''
    classifier = TextClassify()
    print("training")
    classifier.train(generate_tuples_from_file(training))
    print(classifier)
    # report precision, recall, f1

    print("classifying each sample")
    weak_class = classifier.classify_all()
    print("Calculating precision.")
    p1 = precision(classifier.gold_labels, classifier.predicted_labels)
    print("Calculating recall.")
    r1 = recall(classifier.gold_labels, classifier.predicted_labels)
    print("Calculating F1.")
    f11 = f1(classifier.gold_labels, classifier.predicted_labels)
    
    #for text, num in weak_class.items():
        #print(str(num) + " - " + text)
    

    print("\nprecision: " + str(p1))
    print("recall: " + str(r1))
    print("f1: " + str(f11))
    '''

    improved = TextClassifyImproved()
    print(improved)


    # better classifier
    better_classifier = TextClassifyImproved()
    print(better_classifier.normalize_helper("This is WITHOUT a doubt one of the best movies I have ever seen. The first time I saw it I was about 9 or 10 years old. I began looking sometime before the rape scene. And when I saw it I was really shocked thinking \"What kinda sick movie is this?\". Today I've seen it from the beginning and really understood how great this movie really is. It's exciting, frightening, shocking and in it's own unique way disturbing. But the best thing about it is the ending where the audience is shown that this experience will haunt the characters for the rest of their lifes. It'll torture their conscience and they will worry for the rest of their lifes about the bodies being found in that river. And there is nothing they can do about it, it's something they have to live with. This ending is one of the most unhappy endings in movie history and very smart, brilliant and horrifyingAnd the acting is also great, especially Jon Voight and Burt Reynolds. Magnificent acting in this movie. All in all, John Boorman has created one of the best movies throughout movie history based on Dick Chaney's novel. A must see for all the movie lovers"))


    print("training...")
    better_classifier.train(generate_tuples_from_file(training))
    print(better_classifier)
    # report precision, recall, f1
    print("classifying all...")
    goat_class = better_classifier.classify_all()
    p1 = precision(better_classifier.gold_labels, better_classifier.predicted_labels)
    r1 = recall(better_classifier.gold_labels, better_classifier.predicted_labels)
    f11 = f1(better_classifier.gold_labels, better_classifier.predicted_labels)
    #for text, num in weak_class.items():
        #print(str(bin) + " - " + text)

    print("\nprecision: " + str(p1))
    print("recall: " + str(r1))
    print("f1: " + str(f11))
    improved = TextClassifyImproved()
    print(improved)
    
    # report a summary of your experiments/features here
    print(improved.describe_experiments())



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:", "python textclassify_model.py training-file.txt testing-file.txt")
        sys.exit(1)

    main()
