import collections

import numpy as np

import csv
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from matplotlib.ticker import PercentFormatter

LABELS = []
IMPORTANT_COLS = []

# Model
#MODEL_CHOICE = "LOGREG"
MODEL_CHOICE = "NAIVE"
#MODEL_CHOICE = "OFFSHELF"

# Mode (either print out train/test data, or get accuracy distribution)
MODE = "TRAIN_TEST"
#MODE = "GET_DISTRIBUTION"
#MODE = "MODE_SENTIMENT"
DISTRIBUTION_DISCOUNT = 0.75

TRIALS = 10000
BINS = 15
VAL_SPLIT = 10

MAX_MODE = 200


# OFFSHELF MODEL PARAMETERS
mixed_threshold = 0.2

# category 1 corresponds to affect, category 2 corresponds to subject category
CATEGORY = 1

# Option to create confusion matrix
CONFUSION_MATRIX_OPTION = False

# Off shelf only detects sentiment
if CATEGORY == 2 and MODEL_CHOICE == "OFFSHELF":
  throw("ERROR: Off the shelf model cannot predict category!")



# Extract Data from CSV
if CATEGORY == 1:
  LABELS = [1, 2, 3, 4]
  LABEL_NAMES = ["Negative", "Neutral", "Mixed", "Positive"]
  IMPORTANT_COLS = [1]

elif CATEGORY == 2:
  LABELS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
  LABEL_NAMES = ["Aesthetics", "Cost", "Safety", "Noise", "Location", "Construction", "Environmental", "Security", "Power", "Durability", "Information", "Disruption", "Government"]
  IMPORTANT_COLS = [2, 3, 4, 5, 6]

QUALTRICS_COMMENT_COLUMN = 62

def extract_csv(filename, labels, col_to_read):
  '''
  Extract messages and corresponding label from CSV file.
  First column must be the message, second column must be the corresponding integer label
  '''

  all_data = list()
  all_labels = list()
  with open(filename, mode='r') as infile:
    reader = csv.reader(infile)
    next(reader, None)
    for rows in reader:
      if rows[col_to_read] is not '':
        label = int(rows[col_to_read])
        if label in labels:
          all_data.append(re.sub(r"[^A-Za-z ]+", '', rows[0]).lower())
          all_labels.append(int(rows[col_to_read]))
  return all_data, all_labels

def mimic_qualtrics(filename, qualtrics_file, labelled_file, all_data, all_labels, all_predictions):
  '''
  Mimics format of qualtricsdata.xlsx, listing the comment, the computer label, and then
  the human label(s). Stores in filename
  '''
  inverse_mapping = dict()
  for index, data in enumerate(all_data):
    inverse_mapping[data.decode('utf-8').strip()] = index

  corrected_all_labels = [[] for _ in range(len(all_data))]
  if len(IMPORTANT_COLS) > 1:
    with open(labelled_file, mode='r') as infile:
      reader = csv.reader(infile)
      next(reader, None)
      count = 0
      for rows in reader:
        normalized = re.sub(r"[^A-Za-z ]+", '', rows[0]).lower().strip()
        if normalized in inverse_mapping:
          example_index = inverse_mapping[normalized]
          if len(corrected_all_labels[example_index]) == 0:
            for col in IMPORTANT_COLS:
              if col < len(rows):
                corrected_all_labels[example_index].append(rows[col])
  else:
    for i in range(len(all_labels)):
      corrected_all_labels[i].append(str(all_labels[i]))

  try:
    os.remove(filename)
  except OSError:
    pass

  with open(filename, 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerow(['comment', 'computer label', 'human label'])
    writer.writerow([])
    with open(qualtrics_file, mode='r') as infile:
      reader = csv.reader(infile)
      next(reader, None)
      next(reader, None)
      for rows in reader:
        #normalized = " " + re.sub(r"[^A-Za-z ]+", '', rows[QUALTRICS_COMMENT_COLUMN]).lower()
        normalized = re.sub(r"[^A-Za-z ]+", '', rows[QUALTRICS_COMMENT_COLUMN]).lower().strip()
        if normalized in inverse_mapping:
          mapped = inverse_mapping[normalized]
          new_row = [rows[QUALTRICS_COMMENT_COLUMN]]+[str(all_predictions[mapped])]+corrected_all_labels[mapped]
          writer.writerow(new_row)
        else:
          writer.writerow([])

def split_folds(X_train, y_train, num_folds):
    """Split up the training data into `num_folds` folds.

    The goal of the functions is to return training sets (features and labels) along with
    corresponding validation sets. In each fold, the validation set will represent (1/num_folds)
    of the data while the training set represent (num_folds-1)/num_folds.
    If num_folds=5, this corresponds to a 80% / 20% split.

    For instance, if X_train = [0, 1, 2, 3, 4, 5], and we want three folds, the output will be:
        X_trains = [[2, 3, 4, 5],
                    [0, 1, 4, 5],
                    [0, 1, 2, 3]]
        X_vals = [[0, 1],
                  [2, 3],
                  [4, 5]]

    Args:
        X_train: numpy array of shape (N, D) containing N examples with D features each
        y_train: numpy array of shape (N,) containing the label of each example
        num_folds: number of folds to split the data into

    jeturns:
        X_trains: numpy array of shape (num_folds, train_size * (num_folds-1) / num_folds, D)
        y_trains: numpy array of shape (num_folds, train_size * (num_folds-1) / num_folds)
        X_vals: numpy array of shape (num_folds, train_size / num_folds, D)
        y_vals: numpy array of shape (num_folds, train_size / num_folds)

    """
    assert X_train.shape[0] == y_train.shape[0]

    validation_size = X_train.shape[0] // num_folds
    training_size = X_train.shape[0] - validation_size
    '''
    X_trains = np.zeros((num_folds, training_size, X_train.shape[1]))
    y_trains = np.zeros((num_folds, training_size), dtype=np.int)
    X_vals = np.zeros((num_folds, validation_size, X_train.shape[1]))
    y_vals = np.zeros((num_folds, validation_size), dtype=np.int)
    '''
    X_trains = np.empty([num_folds, training_size, X_train.shape[1]], dtype="S1000")
    y_trains = np.zeros((num_folds, training_size), dtype=np.int)
    X_vals = np.empty([num_folds, validation_size, X_train.shape[1]], dtype="S1000")
    y_vals = np.zeros((num_folds, validation_size), dtype=np.int)

    for i in range(num_folds):
        X_trains[i] = np.concatenate((X_train[:i*validation_size], X_train[(i+1)*validation_size:]))
        X_vals[i] = X_train[i*validation_size:(i+1)*validation_size]
        y_trains[i] = np.concatenate((y_train[:i*validation_size], y_train[(i+1)*validation_size:]))
        y_vals[i] = y_train[i*validation_size:(i+1)*validation_size]

    return X_trains, y_trains, X_vals, y_vals

def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """
    #lemmatizer = WordNetLemmatizer()
    #stop_words = set([lemmatizer.lemmatize(word) for word in stopwords.words('english')])

    if isinstance(message, (np.ndarray)):
      return list(filter(None, message[0].decode("utf-8").lower().split(' ')))
    result = list(filter(None, message.lower().split(' ')))
    #result = [lemmatizer.lemmatize(word) for word in result]
    return result

def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    metadata = dict()
    index = 0
    indices = dict()
    seen_words = set()
    for message in messages:
      if isinstance(message, (np.ndarray)):
        all_words = get_words(message[0].decode("utf-8"))
      else:
        all_words = get_words(message.decode("utf-8"))
      for word in set(all_words):
        if word in seen_words:
          metadata[word] += 1
          if metadata[word] == 5:
            indices[word] = index
            index += 1
        else:
          seen_words.add(word)
          metadata[word] = 1
    return indices

def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    appears in each message. Each row in the resulting array should correspond to each
    message and each column should correspond to a word.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
    """
    A = np.zeros((len(messages), len(word_dictionary)))
    for i in range(len(messages)):
      all_words = get_words(messages[i])
      for word in all_words:
        index = word_dictionary.get(word, -1)
        if index != -1:
          A[i][index] += 1
    return A

def fit_logistic_regresion_model(matrix, labels, possibles):
    """Fit a logistic regresion model.

    This function should fit a logistic regresion model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    ylabel_words = []
    ylabel_p = []
    for possible in possibles:
      the_prob = (np.dot((labels == possible).T, matrix) + 1)/(np.sum((labels == possible) * np.sum(matrix, axis = 1)) + matrix.shape[1])
      ylabel_words.append(the_prob)
      ylabel_p.append(np.sum(labels == possible)/len(labels))
    return (ylabel_words, ylabel_p, possibles)

def predict_from_logistic_regresion_model(model, matrix, verbose = False, messages = None):
    """Use a logistic regresion model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_logistic_regresion_model
    outputs.

    Args:
        model: A trained model from fit_logistic_regresion_model
        matrix: A numpy array containing word counts

    Returns: The trained model
    """

    ylabel_words, ylabel_p, labels = model
    predictions = np.zeros(len(matrix))
    for i in range(len(matrix)):
      message = matrix[i]
      best_index = -1
      best = np.NINF
      if verbose == True:
        print("Message:", messages[i])
      for j in range(len(ylabel_p)):
        prob = sum(np.multiply(message,np.log(ylabel_words[j]))) + np.log(ylabel_p[j])
        if verbose == True:
          print("Label:",labels[j],"Confidence:",prob)
        if prob > best:
          best = prob
          best_index = j
      predictions[i] = labels[best_index]
    return predictions

def get_top_five_logistic_regresion_words(model, dictionary, aggregator, verbose = False):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Use the metric given in 6c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The logistic regresion model returned from fit_logistic_regresion_model
        dictionary: A mapping of word to integer ids

    Returns: The top five most indicative words in sorted order with the most indicative first
    """

    inv_map = {v: k for k, v in dictionary.items()}
    ylabel_words, ylabel_p, labels = model

    for i in range(len(labels)):
      ranking = np.log(ylabel_words[i])
      for j in range(len(labels)):
        if j != i:
          ranking -= np.log(ylabel_words[j])
      k = 10
      top_five = np.argpartition(ranking, -1*k)[-1*k:]
      top = []
      for j in sorted(top_five, key=lambda x: ranking[x], reverse=True):
        top.append(inv_map[j])
        aggregator[i][inv_map[j]] = aggregator[i].get(inv_map[j], 0) + 1
      if verbose:
        print("Top words for label", labels[i], ":")
        print(top)

def write_csv_results(val_messages, val_labels, naive_bayes_predictions, trial, confusion_matrix = False):
    """Write CSV file containing message, true label, and predicted label
    """
    filename = 'results'+str(trial)+'.csv'
    try:
      os.remove(filename)
    except OSError:
      pass
    if confusion_matrix:
      num_true_pos = 0
      num_true_neg = 0
      num_false_pos = 0
      num_false_neg = 0
      try:
        os.remove('true_pos'+str(trial)+'.csv')
        os.remove('true_neg'+str(trial)+'.csv')
        os.remove('false_pos'+str(trial)+'.csv')
        os.remove('false_neg'+str(trial)+'.csv')
      except OSError:
        pass
      with open('true_pos'+str(trial)+'.csv', 'w') as writeFile_true_pos:
        writer_true_pos = csv.writer(writeFile_true_pos)
        writer_true_pos.writerow(['comment', 'human label', 'computer label'])
        with open('true_neg'+str(trial)+'.csv', 'w') as writeFile_true_neg:
          writer_true_neg = csv.writer(writeFile_true_neg)
          writer_true_neg.writerow(['comment', 'human label', 'computer label'])
          with open('false_pos'+str(trial)+'.csv', 'w') as writeFile_false_pos:
            writer_false_pos = csv.writer(writeFile_false_pos)
            writer_false_pos.writerow(['comment', 'human label', 'computer label'])
            with open('false_neg'+str(trial)+'.csv', 'w') as writeFile_false_neg:
              writer_false_neg = csv.writer(writeFile_false_neg)
              writer_false_neg.writerow(['comment', 'human label', 'computer label'])
              for i in range(len(val_messages)):
                if (val_labels[i] == 4 and naive_bayes_predictions[i] == 4):
                  num_true_pos += 1
                  writer_true_pos.writerow([val_messages[i], val_labels[i], naive_bayes_predictions[i]])
                elif (val_labels[i] == 1 and naive_bayes_predictions[i] == 1):
                  num_true_neg += 1
                  writer_true_neg.writerow([val_messages[i], val_labels[i], naive_bayes_predictions[i]])
                elif (val_labels[i] == 1 and naive_bayes_predictions[i] == 4):
                  num_false_pos += 1
                  writer_false_pos.writerow([val_messages[i], val_labels[i], naive_bayes_predictions[i]])
                elif (val_labels[i] == 4 and naive_bayes_predictions[i] == 1):
                  num_false_neg += 1
                  writer_false_neg.writerow([val_messages[i], val_labels[i], naive_bayes_predictions[i]])
      print("Number of True Positives:", num_true_pos)
      print("Number of True Negatives:", num_true_neg)
      print("Number of False Positives:", num_false_pos)
      print("Number of False Negatives:", num_false_neg)
    else:
      filename = 'results'+str(trial)+'.csv'
      with open(filename, 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow(['comment', 'human label', 'computer label'])
        for i in range(len(val_messages)):
          writer.writerow([val_messages[i], val_labels[i], naive_bayes_predictions[i]])

def write_csv_mixed(val_messages, val_labels, naive_bayes_predictions, trial, confusion_matrix = False):
    """Write CSV file containing message, true label, and predicted label
    """
    filename = 'results'+str(trial)+'.csv'
    try:
      os.remove(filename)
    except OSError:
      pass
    if confusion_matrix:
      num_true_pos = 0
      num_true_neg = 0
      num_true_other = 0
      num_false_pos = 0
      num_false_neg = 0
      num_false_other = 0
      num_pos_pos = 0
      num_pos_other = 0
      num_pos_neg = 0
      num_other_pos = 0
      num_other_other = 0
      num_other_neg = 0
      num_neg_pos = 0
      num_neg_other = 0
      num_neg_neg = 0
      try:
        os.remove('true_pos'+str(trial)+'.csv')
        os.remove('true_neg'+str(trial)+'.csv')
        os.remove('wrong_pos'+str(trial)+'.csv')
        os.remove('wrong_neg'+str(trial)+'.csv')
        os.remove('true_other'+str(trial)+'.csv')
        os.remove('wrong_other'+str(trial)+'.csv')
      except OSError:
        pass
      with open('true_pos'+str(trial)+'.csv', 'w') as writeFile_true_pos:
        writer_true_pos = csv.writer(writeFile_true_pos)
        writer_true_pos.writerow(['comment', 'human label', 'computer label'])
        with open('true_neg'+str(trial)+'.csv', 'w') as writeFile_true_neg:
          writer_true_neg = csv.writer(writeFile_true_neg)
          writer_true_neg.writerow(['comment', 'human label', 'computer label'])
          with open('wrong_pos'+str(trial)+'.csv', 'w') as writeFile_false_pos:
            writer_false_pos = csv.writer(writeFile_false_pos)
            writer_false_pos.writerow(['comment', 'human label', 'computer label'])
            with open('wrong_neg'+str(trial)+'.csv', 'w') as writeFile_false_neg:
              writer_false_neg = csv.writer(writeFile_false_neg)
              writer_false_neg.writerow(['comment', 'human label', 'computer label'])
              with open('true_other'+str(trial)+'.csv', 'w') as writeFile_true_other:
                writer_true_other = csv.writer(writeFile_true_other)
                writer_true_other.writerow(['comment', 'human label', 'computer label'])
                with open('wrong_other'+str(trial)+'.csv', 'w') as writeFile_false_other:
                  writer_false_other = csv.writer(writeFile_false_other)
                  writer_false_other.writerow(['comment', 'human label', 'computer label'])
                  for i in range(len(val_messages)):
                    if (val_labels[i] == 4 and naive_bayes_predictions[i] == 4):
                      num_true_pos += 1
                      num_pos_pos += 1
                      writer_true_pos.writerow([val_messages[i], val_labels[i], naive_bayes_predictions[i]])
                    elif (val_labels[i] == 1 and naive_bayes_predictions[i] == 1):
                      num_true_neg += 1
                      num_neg_neg += 1
                      writer_true_neg.writerow([val_messages[i], val_labels[i], naive_bayes_predictions[i]])
                    elif (val_labels[i] == 1 and naive_bayes_predictions[i] != 1):
                      num_false_neg += 1
                      writer_false_neg.writerow([val_messages[i], val_labels[i], naive_bayes_predictions[i]])
                      if (naive_bayes_predictions[i] == 4):
                        num_neg_pos += 1
                      else:
                        num_neg_other += 1
                    elif (val_labels[i] == 4 and naive_bayes_predictions[i] != 4):
                      num_false_pos += 1
                      writer_false_pos.writerow([val_messages[i], val_labels[i], naive_bayes_predictions[i]])
                      if (naive_bayes_predictions[i] == 1):
                        num_pos_neg += 1
                      else:
                        num_pos_other += 1
                    elif ((val_labels[i] == 2 or val_labels[i] == 3) and (naive_bayes_predictions[i] == 2 or naive_bayes_predictions[i] == 3)):
                      num_true_other += 1
                      num_other_other += 1
                      writer_true_other.writerow([val_messages[i], val_labels[i], naive_bayes_predictions[i]])
                    else:
                      num_false_other += 1
                      writer_false_other.writerow([val_messages[i], val_labels[i], naive_bayes_predictions[i]])
                      if (naive_bayes_predictions[i] == 1):
                        num_other_neg += 1
                      else:
                        num_other_pos += 1

      print("Number of True Positives:", num_true_pos)
      print("Number of True Negatives:", num_true_neg)
      print("Number of True Other:", num_true_other)
      print("Number of Incorrect Positives:", num_false_pos)
      print("Number of Incorrect Negatives:", num_false_neg)
      print("Number of Incorrect Other:", num_false_other)
      with open('confusion_matrix'+str(trial)+'.csv', 'w') as writeFile_confusion:
        writer_confusion = csv.writer(writeFile_confusion)
        writer_confusion.writerow(['True labels', 'Positive', 'Negative', 'Other'])
        total_pos = num_pos_pos+num_pos_neg+num_pos_other
        writer_confusion.writerow(['Positive', num_pos_pos/total_pos, num_pos_neg/total_pos, num_pos_other/total_pos])
        total_neg = num_neg_pos+num_neg_neg+num_neg_other
        writer_confusion.writerow(['Negative', num_neg_pos/total_neg, num_neg_neg/total_neg, num_neg_other/total_neg])
        total_other = num_other_pos+num_other_neg+num_other_other
        writer_confusion.writerow(['Other', num_other_pos/total_other, num_other_neg/total_other, num_other_other/total_other])
      '''
      print('')
      print("True labels\tPositive\tNegative\tOther")
      print("Positive\t", num_pos_pos/(num_pos_pos+num_pos_neg+num_pos_other), '\t', num_pos_neg/(num_pos_pos+num_pos_neg+num_pos_other), '\t', num_pos_other/(num_pos_pos+num_pos_neg+num_pos_other))
      print("Negative\t", num_neg_pos/(num_neg_pos+num_neg_neg+num_neg_other), '\t', num_neg_neg/(num_neg_pos+num_neg_neg+num_neg_other), '\t', num_neg_other/(num_neg_pos+num_neg_neg+num_neg_other))
      print("Other\t", num_other_pos/(num_other_pos+num_other_neg+num_other_other), '\t', num_other_neg/(num_other_pos+num_other_neg+num_other_other), '\t', num_other_other/(num_other_pos+num_other_neg+num_other_other))
      '''
    else:
      filename = 'results'+str(trial)+'.csv'
      with open(filename, 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow(['comment', 'human label', 'computer label'])
        for i in range(len(val_messages)):
          writer.writerow([val_messages[i], val_labels[i], naive_bayes_predictions[i]])

def plot_frequent_words(labels, aggregator):
    """ Uses matplotlib to plot frequent words from dictionary
    """
    for i in range(len(labels)):
      # Sort aggregator in descending order
      sorted_d = sorted([(value, key) for (key,value) in aggregator[i].items()])
      zipped = list(zip(*sorted_d))
      values = list(zipped[0])
      keys = list(zipped[1])

      plt.figure(figsize=(10, 6))
      plt.barh(range(len(aggregator[i])), values)#, align='center')
      plt.yticks(range(len(aggregator[i])), keys)
      title_string = LABEL_NAMES[i]
      plt.ylabel("Word")
      plt.xlabel("Top Word Count")
      plt.title(title_string+' Label Top Words')
      plt.savefig(title_string+'_label_frequent_words.png')

def fit_naive_bayes_model(matrix, labels, possibles):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    ylabel_words = []
    ylabel_p = []
    for possible in possibles:
      the_prob = (np.dot((labels == possible).T, matrix) + 1)/(np.sum((labels == possible) * np.sum(matrix, axis = 1)) + matrix.shape[1])
      ylabel_words.append(the_prob)
      ylabel_p.append(np.sum(labels == possible)/len(labels))
    return (ylabel_words, ylabel_p, possibles)

def predict_from_naive_bayes_model(model, matrix, verbose = False, messages = None):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: The trained model
    """

    ylabel_words, ylabel_p, labels = model
    predictions = np.zeros(len(matrix))
    for i in range(len(matrix)):
      message = matrix[i]
      best_index = -1
      best = np.NINF
      if verbose == True:
        print("Message:", messages[i])
      for j in range(len(ylabel_p)):
        prob = sum(np.multiply(message,np.log(ylabel_words[j]))) + np.log(ylabel_p[j])
        if verbose == True:
          print("Label:",labels[j],"Confidence:",prob)
        if prob > best:
          best = prob
          best_index = j
      predictions[i] = labels[best_index]
    return predictions

def predict_from_off_shelf_model(model, messages):
    """Use a NLTK Sentiment Intensity Analyzer to compute predictions for a list of messages.

    This function should be able to predict on the models that SentimentIntensityAnalyzer()
    outputs.

    Args:
        model: A trained model from SentimentIntensityAnalyzer()
        messages: A numpy array where each index is a list containing exactly one message in bytes form.

    Returns: The trained model
    """
    sid_predictions = []
    correct_predictions = 0
    for index, message in enumerate(messages):
      final_pred = None

      ss = model.polarity_scores(message)
      if ss['neg'] > mixed_threshold and ss['pos'] > mixed_threshold:
        final_pred = 3
      else:
        top_score = ss['neg']
        final_pred = 1
        if ss['neu'] > top_score:
          top_score = ss['neu']
          final_pred = 2
        if ss['pos'] > top_score:
          top_score = ss['pos']
          final_pred = 4
      sid_predictions.append(final_pred)
    return np.asarray(sid_predictions)

def train_test():

  all_data, all_labels = extract_csv('comments.csv', LABELS, CATEGORY)
  all_data = np.asarray(all_data)
  all_labels = np.asarray(all_labels)

  # dictionary to count top instances of words
  aggregator = []
  for i in range(len(LABELS)):
    aggregator.append(dict())

  randomization_scheme = np.arange(len(all_data))
  np.random.shuffle(randomization_scheme)
  randomized_data = all_data[randomization_scheme]
  randomized_labels = all_labels[randomization_scheme]


  num_trials = 10


  X_trains, y_trains, X_vals, y_vals = split_folds(randomized_data.reshape((len(all_data), 1)), randomized_labels, num_trials)

  all_predictions = None

  for trial in range(num_trials):

    train_messages = X_trains[trial]
    train_labels = y_trains[trial]
    val_messages = X_vals[trial]
    val_labels = y_vals[trial]


    dictionary = create_dictionary(train_messages)
    train_matrix = transform_text(train_messages, dictionary)
    val_matrix = transform_text(val_messages, dictionary)

    if MODEL_CHOICE is "LOGREG":
      
      logreg = LogisticRegression()
      logreg.fit(train_matrix, train_labels)

      #logistic_regresion_model = fit_logistic_regresion_model(train_matrix, train_labels, labels)
      #logistic_regresion_predictions = predict_from_logistic_regresion_model(logistic_regresion_model, val_matrix, True, val_messages)
      #logistic_regresion_predictions = predict_from_logistic_regresion_model(logistic_regresion_model, val_matrix)
      logistic_regresion_predictions = logreg.predict(val_matrix)
      logistic_regresion_accuracy = np.mean(logistic_regresion_predictions == val_labels)
      print('Logistic Regression had an accuracy of {} on the testing set'.format(logistic_regresion_accuracy))
      #get_top_five_logistic_regresion_words(logreg, dictionary, aggregator)
      write_csv_mixed(val_messages, val_labels, logistic_regresion_predictions, trial, CONFUSION_MATRIX_OPTION)



    elif MODEL_CHOICE is "NAIVE":
      naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels, LABELS)
      naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, val_matrix)
      naive_bayes_accuracy = np.mean(naive_bayes_predictions == val_labels)
      print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))
      get_top_five_logistic_regresion_words(naive_bayes_model, dictionary, aggregator)
      write_csv_mixed(val_messages, val_labels, naive_bayes_predictions, trial, CONFUSION_MATRIX_OPTION)

    elif MODEL_CHOICE is "OFFSHELF":

      sid = SentimentIntensityAnalyzer()
      converted = [x[0].decode('utf-8') for x in val_messages]

      sid_predictions = predict_from_off_shelf_model(sid, converted)
      sid_accuracy = np.mean(sid_predictions == val_labels)
      print('Off The Shelf had an accuracy of {} on the testing set'.format(sid_accuracy))
      write_csv_mixed(val_messages, val_labels, sid_predictions, trial, CONFUSION_MATRIX_OPTION)


    # Just do this once
    if trial == 0:
      all_matrix = transform_text(all_data, dictionary)
      if MODEL_CHOICE is "LOGREG":
        all_predictions = logreg.predict(all_matrix)
      elif MODEL_CHOICE is "NAIVE":
        all_predictions = predict_from_naive_bayes_model(naive_bayes_model, all_matrix)
      elif MODEL_CHOICE is "OFFSHELF":
        all_predictions = predict_from_off_shelf_model(sid, randomized_data)


  plot_frequent_words(LABELS, aggregator)
  #mimic_qualtrics("mimic_results.csv", "all_survey_responses.csv", "comments.csv", all_data, all_labels, all_predictions)

def get_distribution():

  all_data, all_labels = extract_csv('comments.csv', LABELS, CATEGORY)
  all_data = np.asarray([[x] for x in all_data], dtype="S1000")
  all_labels = np.asarray(all_labels)

  num_trials = TRIALS
  accuracies = []

  for trial in range(num_trials):
    if (trial % 10) == 0:
      print("Trial:",trial)

    randomization_scheme = np.arange(len(all_data))
    np.random.shuffle(randomization_scheme)
    randomized_data = all_data[randomization_scheme]
    randomized_labels = all_labels[randomization_scheme]

    train_messages = randomized_data[len(all_data)//VAL_SPLIT:]
    train_labels = randomized_labels[len(all_data)//VAL_SPLIT:]
    val_messages = randomized_data[:len(all_data)//VAL_SPLIT]
    val_labels = randomized_labels[:len(all_data)//VAL_SPLIT]


    dictionary = create_dictionary(train_messages)
    train_matrix = transform_text(train_messages, dictionary)
    val_matrix = transform_text(val_messages, dictionary)

    if MODEL_CHOICE is "LOGREG":

      logreg = LogisticRegression()
      logreg.fit(train_matrix, train_labels)

      logistic_regresion_predictions = logreg.predict(val_matrix)
      logistic_regresion_accuracy = np.mean(logistic_regresion_predictions == val_labels)
      accuracies.append(logistic_regresion_accuracy)



    elif MODEL_CHOICE is "NAIVE":
      naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels, LABELS)
      naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, val_matrix)
      naive_bayes_accuracy = np.mean(naive_bayes_predictions == val_labels)
      accuracies.append(naive_bayes_accuracy)

    elif MODEL_CHOICE is "OFFSHELF":

      sid = SentimentIntensityAnalyzer()
      converted = [x[0].decode('utf-8') for x in val_messages]

      sid_predictions = predict_from_off_shelf_model(sid, converted)
      sid_accuracy = np.mean(sid_predictions == val_labels)
      accuracies.append(sid_accuracy)

  plt.figure()
  plt.hist(accuracies, bins=BINS, label='data', weights=np.ones(num_trials)/num_trials)
  plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
  plt.title("Accuracy Distribution for "+MODEL_CHOICE[0]+MODEL_CHOICE[1:].lower()+" Model")
  plt.xlabel("Accuracy")
  plt.ylabel("Percentage")
  accuracies = np.asarray(accuracies)
  plt.axvline(x=np.mean(accuracies), color='red', linestyle='--', label='mean')
  plt.savefig(MODEL_CHOICE.lower()+"_acc_dist.png")

def get_mode_sentiment():
  all_data, all_labels = extract_csv('comments.csv', LABELS, CATEGORY)
  all_data = np.asarray([[x] for x in all_data], dtype="S1000")
  all_labels = np.asarray(all_labels)

  predictions = []
  for _ in range(len(all_labels)):
    predictions.append([])

  trial = 0

  not_done = True

  while not_done:

    randomization_scheme = np.arange(len(all_data))
    np.random.shuffle(randomization_scheme)
    useful = False
    for i in range(len(all_data)//VAL_SPLIT):
      if len(predictions[randomization_scheme[i]]) < MAX_MODE:
        useful = True
        break
    if useful:
      trial += 1
      if (trial % 10) == 0:
        print("Trial:",trial)

      randomized_data = all_data[randomization_scheme]
      randomized_labels = all_labels[randomization_scheme]

      train_messages = randomized_data[len(all_data)//VAL_SPLIT:]
      train_labels = randomized_labels[len(all_data)//VAL_SPLIT:]
      val_messages = randomized_data[:len(all_data)//VAL_SPLIT]
      val_labels = randomized_labels[:len(all_data)//VAL_SPLIT]


      dictionary = create_dictionary(train_messages)
      train_matrix = transform_text(train_messages, dictionary)
      val_matrix = transform_text(val_messages, dictionary)

      guesses = None

      if MODEL_CHOICE is "LOGREG":

        logreg = LogisticRegression()
        logreg.fit(train_matrix, train_labels)

        guesses = logreg.predict(val_matrix)



      elif MODEL_CHOICE is "NAIVE":
        naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels, LABELS)
        guesses = predict_from_naive_bayes_model(naive_bayes_model, val_matrix)

      elif MODEL_CHOICE is "OFFSHELF":

        sid = SentimentIntensityAnalyzer()
        converted = [x[0].decode('utf-8') for x in val_messages]

        guesses = predict_from_off_shelf_model(sid, converted)

      for i in range(len(guesses)):
        if len(predictions[randomization_scheme[i]]) < MAX_MODE:
          predictions[randomization_scheme[i]].append(guesses[i])

      total = 0
      for i in range(len(predictions)):
        total += len(predictions[i])
      if total == len(all_data)*MAX_MODE:
        not_done = False

  total_all_sentiment = np.zeros(len(LABELS))
  total_mode_sentiment = np.zeros(len(LABELS))
  per_comment_proportion = np.zeros((len(predictions), len(LABELS)))
  mode_sentiment = np.zeros(len(predictions))
  for i in range(len(predictions)):
    curr_sentiment = np.zeros(len(LABELS))
    for label in predictions[i]:
      curr_sentiment[label - 1] += 1
      total_all_sentiment[label-1] += 1
    mode_sentiment[i] = np.argmax(curr_sentiment) + 1
    per_comment_proportion[i] = curr_sentiment / sum(curr_sentiment)
    total_mode_sentiment[int(mode_sentiment[i]) - 1] += 1
  total_all_sentiment = total_all_sentiment/sum(total_all_sentiment)
  total_mode_sentiment = total_mode_sentiment/sum(total_mode_sentiment)
  print("Total proportion of all sentiments:",total_all_sentiment)
  print("Total proportion of sentiment by mode:",total_mode_sentiment)
  print("Total Distance:",np.sum(np.abs(total_all_sentiment - total_mode_sentiment)))

  my_accuracy = np.mean(mode_sentiment == all_labels)
  print("This is the accuracy:",my_accuracy)
  mimic_qualtrics("mimic_results2.csv", "all_survey_responses.csv", "comments.csv", all_data.flatten(), all_labels, mode_sentiment.astype(int))




def main():
  if MODE == "TRAIN_TEST":
    train_test()
  elif MODE == "GET_DISTRIBUTION":
    get_distribution()
  elif MODE == "MODE_SENTIMENT":
    get_mode_sentiment()
  else:
    print("ERROR: Unrecognized mode:", MODE)


if __name__ == "__main__":
    main()
