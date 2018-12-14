from __future__ import division
from __future__ import print_function
import os
import sys
import collections
import re
import copy
import ast


class Document:
	text = ""
	true_class = ""
	learned_class = ""

	word_freqs = {}

	def __init__(self, text, counter, true_class):
		self.text = text
		self.word_freqs = counter
		self.true_class = true_class

	def getText(self):
		return self.text

	def getWordFreqs(self):
		return self.word_freqs

	def getTrueClass(self):
		return self.true_class

	def getLearnedClass(self):
		return self.learned_class

	def setLearnedClass(self, prediction):
		self.learned_class = prediction


def bagOfWords(text):
	bagsofwords = collections.Counter(re.findall(r'\w+', text))
	return dict(bagsofwords)


def buildData(storage_dict, directory, true_class):
	for dir_entry in os.listdir(directory):
		dir_entry_path = os.path.join(directory, dir_entry)
		if os.path.isfile(dir_entry_path):
			with open(dir_entry_path, 'r') as text_file:
				text = text_file.read()
				storage_dict.update({dir_entry_path: Document(text, bagOfWords(text), true_class)})


def throwAwayStopWords(stop_words, data_set):
	filtered_data_set = copy.deepcopy(data_set)
	for i in stop_words:
		for j in filtered_data_set:
			if i in filtered_data_set[j].getWordFreqs():
				del filtered_data_set[j].getWordFreqs()[i]
	return filtered_data_set


def getDataVocabulary(data_set):
	vocabulary = []
	for i in data_set:
		for j in data_set[i].getWordFreqs():
			if j not in vocabulary:
				vocabulary.append(j)
	return vocabulary


def setPerceptronWeights(weights, learning_constant, spam_ham_training_set, num_iterations, classes):
	for i in num_iterations:
		for d in spam_ham_training_set:
			weight_sum = weights['init_weight']
			for f in spam_ham_training_set[d].getWordFreqs():
				if f not in weights:
					weights[f] = 0.0
				weight_sum += weights[f] * spam_ham_training_set[d].getWordFreqs()[f]
			perceptron_output = 0.0
			if weight_sum > 0:
				perceptron_output = 1.0
			target_value = 0.0
			if spam_ham_training_set[d].getTrueClass() == classes[1]:
				target_value = 1.0
			for w in spam_ham_training_set[d].getWordFreqs():
				weights[w] += float(learning_constant) * float((target_value - perceptron_output)) * \
							  float(spam_ham_training_set[d].getWordFreqs()[w])


def perceptronClassifier(weights, classes, instance):
	weight_sum = weights['init_weight']
	for i in instance.getWordFreqs():
		if i not in weights:
			weights[i] = 0.0
		weight_sum += weights[i] * instance.getWordFreqs()[i]
	if weight_sum > 0:
		return 1
	else:
		return 0


def main():

	args = str(sys.argv)
	args = ast.literal_eval(args)

	if (len(args) < 5):

		print( "You have input less than the minimum number of arguments.")
		print("Usage: python Perceptron.py <training-dir> <test-dir> <no-of-iterations> <learning-constant>")

	elif (not os.path.isdir(args[1]) and not os.path.isdir(args[2])):

		print("First 2 arguments must be directories of spam/ham training and test mails!")
		print("Usage: python Perceptron.py <training-dir> <test-dir> <no-of-iterations> <learning-constant>")

	else:

		spam_ham_train_dir = args[1]
		spam_ham_test_dir = args[2]
		iterations = args[3]
		learning_constant = args[4]

		spam_ham_training_set = {}
		spam_ham_test_set = {}
		filtered_spam_ham_training_set = {}
		filtered_spam_ham_test_set = {}

		classes = ["ham", "spam"]

		stop_words = []
		with open('stop_words.txt', 'r') as txt:
			stop_words = (txt.read().splitlines())

		
		buildData(spam_ham_training_set, spam_ham_train_dir + "/spam", classes[1])
		buildData(spam_ham_training_set, spam_ham_train_dir + "/ham", classes[0])
		buildData(spam_ham_test_set, spam_ham_test_dir + "/spam", classes[1])
		buildData(spam_ham_test_set, spam_ham_test_dir + "/ham", classes[0])

		
		filtered_spam_ham_training_set = throwAwayStopWords(stop_words, spam_ham_training_set)
		filtered_spam_ham_test_set = throwAwayStopWords(stop_words, spam_ham_test_set)

		vocabulary_spam_ham_training_set = getDataVocabulary(spam_ham_training_set)
		filtered_vocalbulary_spam_ham_training_set = getDataVocabulary(filtered_spam_ham_training_set)

		weights = {'init_weight': 1.0}
		filtered_weights = {'init_weight': 1.0}
		for i in vocabulary_spam_ham_training_set:
			weights[i] = 0.0
		for i in filtered_vocalbulary_spam_ham_training_set:
			filtered_weights[i] = 0.0   

		setPerceptronWeights(weights, learning_constant, spam_ham_training_set, iterations, classes)
		setPerceptronWeights(filtered_weights, learning_constant, filtered_spam_ham_training_set, iterations, classes)

		correct_predictions = 0
		for i in spam_ham_test_set:
			prediction = perceptronClassifier(weights, classes, spam_ham_test_set[i])
			if prediction == 1:
				spam_ham_test_set[i].setLearnedClass(classes[1])
				if spam_ham_test_set[i].getTrueClass() == spam_ham_test_set[i].getLearnedClass():
					correct_predictions += 1
			if prediction == 0:
				spam_ham_test_set[i].setLearnedClass(classes[0])
				if spam_ham_test_set[i].getTrueClass() == spam_ham_test_set[i].getLearnedClass():
					correct_predictions += 1

		filtered_correct_predictions = 0
		for i in filtered_spam_ham_test_set:
			prediction = perceptronClassifier(filtered_weights, classes, filtered_spam_ham_test_set[i])
			if prediction == 1:
				filtered_spam_ham_test_set[i].setLearnedClass(classes[1])
				if filtered_spam_ham_test_set[i].getTrueClass() == filtered_spam_ham_test_set[i].getLearnedClass():
					filtered_correct_predictions += 1
			if prediction == 0:
				filtered_spam_ham_test_set[i].setLearnedClass(classes[0])
				if filtered_spam_ham_test_set[i].getTrueClass() == filtered_spam_ham_test_set[i].getLearnedClass():
					filtered_correct_predictions += 1

		
		print( "Learning constant: %.4f" % float(learning_constant))
		print( "Number of iterations: %d" % int(iterations))
		print( "Emails classified correctly: %d/%d" % (correct_predictions, len(spam_ham_test_set)))
		print( "Accuracy before filtering: %.4f%%" % (float(correct_predictions) / float(len(spam_ham_test_set)) * 100.0))
		print( "Filtered emails classified correctly: %d/%d" % (filtered_correct_predictions, len(filtered_spam_ham_test_set)))
		print( "Filtered accuracy: %.4f%%" % (float(filtered_correct_predictions) / float(len(filtered_spam_ham_test_set)) * 100.0))


if __name__ == '__main__':
	main()