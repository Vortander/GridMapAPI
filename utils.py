# coding: utf-8
import cPickle as pickle


def saveMap(obj, filename):
	with open(filename, 'wb') as output:
		pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def loadMap(filename):
	with open(filename, 'rb') as input:
		return pickle.load(input)


