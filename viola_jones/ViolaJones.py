import numpy as np
from math import log
import pickle
from datetime import datetime
import operator
import concurrent.futures
import multiprocessing as mp
from sklearn.feature_selection import SelectPercentile, f_classif
import inspect
from WeakClassifier import WeakClassifier
from helpers import POSITIVE_CLASSIFICATION, \
  NEGATIVE_CLASSIFICATION, \
  NUM_THREADS, \
  get_integral_image, \
  create_rectangular_region, \
  coumpute_feature, \
  get_applied_feature



class ViolaJones():
  def __init__(self, T = 15):
    self.T = T
    self.classifiers = []
    self.alphas = []


  @staticmethod
  def INPUT_DIMENSIONS():
    return 19,19  
  """
    The Following Methods are not needed during runtime
  """

  def train(self, training_data, features, applied_features ,num_pos, num_neg):
    weights = np.zeros(len(training_data))
    print("Calculating Weights")
    for i in range(len(training_data)):
      image, classification = training_data[i]

      if classification == POSITIVE_CLASSIFICATION:
        weights[i] = 1 / (2 * num_pos) # multiply by 2 to normalize values to sum to 1
      else:
        weights[i] = 1 / (2 * num_neg)
    print("Finished Calculating Weights")

    X,y = applied_features

    for i in range(self.T):
      # Normalize weights...
      weights = weights / np.linalg.norm(weights)
      weak_classifiers = self.train_weak_classifiers_mt(X, y, features, weights)
      classifier, e, accuracy = self.select_best_classifier_mt(weak_classifiers, weights, training_data)
      ## Updating weights according to: https://miro.medium.com/max/657/1*YL7Km5a5NpQ-wJujo0rWzw.png
      beta = e / (1 - e)
      for j in range(len(accuracy)):
        weights[j] = weights[j] * beta ** (1 - accuracy[j])
      ## Alphas are computed according to: https://miro.medium.com/max/700/1*cIfKETbjBCjOVFOYYJjxrg.png
      alpha = log(1/beta)
      self.alphas.append(alpha)
      self.classifiers.append(classifier)
      print("Chose classifier: %s with accuracy: %f and alpha: %f" % (str(classifier), len(accuracy) - sum(accuracy), alpha))

  def __count_pos_neg(self, y, weights):
    weight_pos = 0
    weight_neg = 0

    for weight, label in zip(weights, y):
      if label == POSITIVE_CLASSIFICATION:
        weight_pos = weight_pos + weight
      else:
        weight_neg = weight_neg + weight

    return (weight_pos, weight_neg)

  def __get_sorting_key(self, x):
    return x[1]


  def train_weak_classifiers_mt(self, X, y, features, weights):
    print("Started training weak classifiers")
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    chunks = np.array_split(X, NUM_THREADS)
    pool = mp.Pool(mp.cpu_count())
    results = [pool.apply_async(self.train_weak_classifiers, args=(chunk, y, features, weights)) for chunk in chunks]
    pool.close()
    classifiers = []
    for result in results:
      _classifiers = result.get()
      classifiers = np.concatenate((classifiers, _classifiers))
    print("Finished training weak classifiers")
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    return classifiers


  def train_weak_classifiers(self, X, y, features, weights):

    weight_pos, weight_neg = self.__count_pos_neg(y, weights)

    weak_classifiers = []
    num_features, __discarded = X.shape
    for i, feature in enumerate(X):
      applied_features = sorted(
          zip(weights, feature, y),
          key=self.__get_sorting_key
      )

      pos_seen = 0
      neg_seen = 0

      pos_weights = 0
      neg_weights = 0

      min_error = float('inf')
      best_feature = None
      best_theta = None
      best_p = None

      for weight, theta, label in applied_features:
        error_neg = neg_weights + weight_pos - pos_weights
        error_pos = pos_weights + weight_neg - neg_weights
        error = min(error_neg, error_pos)

        if error < min_error:
          min_error = error
          best_feature = features[i]
          best_theta = theta
          best_p = 1 if pos_seen > neg_seen else -1

        if label == POSITIVE_CLASSIFICATION:
          pos_seen = pos_seen + 1
          pos_weights = pos_weights + weight
        else:
          neg_seen = neg_seen + 1
          neg_weights = neg_weights + weight

      weak_classifier = WeakClassifier(best_feature, best_theta, best_p)
      weak_classifiers.append(weak_classifier)

    return weak_classifiers

  def select_best_classifier_mt(self, classifiers, weights, training_data_with_integral_images):
    print("Started counting errors")
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    chunks = np.array_split(classifiers, NUM_THREADS)
    pool = mp.Pool(mp.cpu_count())
    futures = []
    j = 0
    results = [pool.apply_async(self.select_best_classifier, args=(chunk, weights, training_data_with_integral_images)) for chunk in chunks]
    pool.close()
    errors = []
    for result in results:
      error = result.get()
      errors.append((error[0], j*len(chunks[j]) + error[1], error[2]))
      j = j + 1
    classifier, e, acc = min(errors, key=operator.itemgetter(1))
    print("Finished counting errors")
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    return classifier, e, acc

  def select_best_classifier(self, classifiers, weights, training_data_with_integral_images):

    best_classifier = None
    best_error = float('inf')
    best_accuracy = None
    errors = []
    divisor = len(training_data_with_integral_images);
    zipped = zip(training_data_with_integral_images, weights)
    num_classifiers = len(classifiers)
    for i in range(num_classifiers):
      error = 0
      classifier = classifiers[i]
      accuracy = []
      for j in range(divisor):
        data = training_data_with_integral_images[j]
        weight = weights[j]
        integral_image, classification = data
        prediction = classifier.classify(integral_image)
        is_incorrect = abs(prediction - classification)
        accuracy.append(is_incorrect)
        error = error + weight * is_incorrect

      errors.append((error / divisor, i, accuracy));

    e, idx, acc = min(errors, key=operator.itemgetter(0))
    return classifiers[idx], e, acc




  def classify(self, image):
    ## Final classification is computed according to: https://miro.medium.com/max/700/1*cIfKETbjBCjOVFOYYJjxrg.png
    classification_sum = 0
    alpha_sum = 0
    integral_image = get_integral_image(image)
    h,w = integral_image.shape
    for alpha, classifier in zip(self.alphas, self.classifiers):
      classification_sum = classification_sum + (alpha * classifier.classify(integral_image))
      alpha_sum = alpha_sum + alpha
    return classification_sum / (alpha_sum) 

  def save(self, filename):
    with open(filename, "wb") as file:
      pickle.dump(self, file)

  @staticmethod
  def load(filename):
    print(__name__)
    with open(filename, "rb") as file:
      print("file")
      return pickle.load(file)

