# coding: utf-8
import numpy as np
import random

def stratified_proportional_sampling(attr_cell_distribution, trainpercent, testpercent, debug=False):
	attr_distribution = sorted([value[0] for value in attr_cell_distribution])
	histogram_values, bin_edges = np.histogram(attr_distribution)

	total_cells = float(len(attr_cell_distribution))
	total_train = (trainpercent * total_cells)/100.0
	total_test = (testpercent * total_cells)/100.0
	
	#separate samples in bins
	trainpercent_bins = []
	testpercent_bins = []
	cells = {}
	
	for i, b in zip( range(0, len(histogram_values)), histogram_values ):
					
		bin_trainp = np.rint((trainpercent * b)/100.0)
		bin_testp = np.rint((testpercent * b)/100.0)
		
		while( (bin_trainp + bin_testp > b) and (bin_trainp + bin_testp) != b):
			bin_trainp -= 1
			
		trainpercent_bins.append(int(bin_trainp))
		testpercent_bins.append(int(bin_testp))
		
		cells[(i,b)] = list()
		for cell in attr_cell_distribution:
			if cell[0] >= bin_edges[i] and cell[0] <= bin_edges[i+1]:
				cells[(i,b)].append(cell)
				

	#separate samples in train and test
	train_cells = {}
	test_cells = {}
	
	for i, b_train, b_test, all_b in zip(range(0, len(histogram_values)), trainpercent_bins, testpercent_bins, histogram_values):
		
		#train set
		train_cells[(i, b_train)] = list()
		for index in range(0, b_train):
			random.shuffle(cells[(i, all_b)])
			sample = cells[(i, all_b)].pop()
			train_cells[(i, b_train)].append(sample)
			
		#test_set
		test_cells[(i, b_test)] = list()
		for index in range(0, b_test):
			random.shuffle(cells[(i, all_b)])
			sample = cells[(i, all_b)].pop()
			test_cells[(i, b_test)].append(sample)
			
	if debug == True:
		print("-----------------------------------------------------------------------------------------")
		print("Stratified Proportional Sampling of cells in Train and Test sets")
		print("acording to the attribute value distribution")
		print("-----------------------------------------------------------------------------------------")
		print("Lenght of attribute distribution, or Total Number of cells: ", len(attr_distribution))
		print("Histogram values: ", histogram_values)
		print("Bin edges: ", bin_edges)
		print("..........................................................................................")
		print("Lenght of train distribution, or Total Train cells: ", str(trainpercent)+"%", total_train)
		print("Lenght of train distribution, or Total Test cells: ", str(testpercent)+"%", total_test)
		print("Train histogram values: ", trainpercent_bins)
		print("Test histogram values: ", testpercent_bins)
		print("..........................................................................................")
		print("Train cells Keys (index, histogram_value): ")
		print(train_cells.keys())
		print("Test cells Keys (index, histogram_value): ")
		print(test_cells.keys())
		 
		for train_key in train_cells:
			for test_key in test_cells:
				if train_key[0] == test_key[0]:
					print(train_key, test_key, ", Total bins: ", train_key[1] + test_key[1], ", Orig. Hist. value", histogram_values[train_key[0]], ", Total equal elements: ", np.sum(train_cells[train_key] == test_cells[test_key]))
			
	return train_cells, test_cells