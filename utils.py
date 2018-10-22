# coding: utf-8
import cPickle as pickle
import matplotlib.pyplot as plt


def saveMap(obj, filename):
	with open(filename, 'wb') as output:
		pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def loadMap(filename):
	with open(filename, 'rb') as input:
		return pickle.load(input)

def plot_histogram_distribution(cell_distribution, logscale=True):
	#cell_distribution format (attr_value, cell)
	distribution = [a[0] for a in cell_distribution]
	N, bins, patches = plt.hist(distribution)
	jet = plt.get_cmap('jet', len(patches))

	for i in range(len(patches)):
		patches[i].set_facecolor(jet(i))

	if logscale == True:
		plt.yscale('log')

	plt.ylabel('Number of Cells', fontsize = 15)
	plt.xlabel('Normalized DF/DHF Rates', fontsize = 15)
	plt.show()
