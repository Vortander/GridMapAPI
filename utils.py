# coding: utf-8
import cPickle as pickle


def save_PointMap(obj, filename):
	with open(filename, 'wb') as output:
		pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_PointMap(filename):
	with open(filename, 'rb') as input:
		return pickle.load(input)



def save_BairrosMap():
	pass

def load_BairrosMap():
	pass

