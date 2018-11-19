# coding: utf-8
import numpy as np
import random
import os

#Temporary
from shapely.geometry import Point as Pt

def stratified_proportional_sampling(attr_cell_distribution, trainpercent, testpercent, bins=10, debug=True):
	attr_distribution = sorted([value[0] for value in attr_cell_distribution])
	print(attr_distribution)

	histogram_values, bin_edges = np.histogram(attr_distribution, bins=bins)

	print(histogram_values)
	print(bin_edges)

	total_cells = float(len(attr_cell_distribution))
	total_train = (trainpercent * total_cells)/100.0
	total_test = (testpercent * total_cells)/100.0

	#separate samples in bins
	trainpercent_bins = []
	testpercent_bins = []
	cells = {}

	for i, b in zip( range(0, len(histogram_values)), histogram_values ):

		bin_trainp = np.floor((trainpercent * b)/100.0)

		# Assures that at least one sample goes to train set.
		if bin_trainp < 1:
			bin_trainp = 1
		bin_testp = b - bin_trainp

#		while( (bin_trainp + bin_testp > b) and (bin_trainp + bin_testp) != b):
#			bin_trainp -= 1

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

# Generate distribution for variables (for various sector_objects)

def gen_attr_cell_distribution ( sector_objects=list(), variable_keys=list(), sort=False ):
	# If variable_keys indicates more than one key, returns the sum of the values
	# return tuple array (total_variable, code, variable_key, name)
	attr_cell_distribution = []
	for sector_object in sector_objects:
		name = sector_object.name
		for code in sector_object.sector_codes:
			total_variable = 0
			for key in variable_keys:
				if key in sector_object.grid_sectors[code]['total_variable_list'].keys():
					total_variable += sector_object.grid_sectors[code]['total_variable_list'][key]

			attr_cell_distribution.append((total_variable, code, variable_keys, name))
	if sort == True:
		attr_cell_distribution = sorted(attr_cell_distribution, key=lambda x: x[0])

	return attr_cell_distribution

# Generate distribution for variables
# Pass a SectorMap object and config parameters
# return arrays (tuple with code, target_variable, independent_variable), numpy array y, numpy array X, sectors with errors.

def gen_variable_distribution ( sector_object, variable_keys=list(), target_variable_key=None, sort=False, sortidx=2 ):
	target_var = []
	error_target_var = []

	for code in sector_object.sector_codes:
		target_value = 0.0
		total_variable = 0.0
		try:
			if np.isnan(sector_object.grid_sectors[code]['total_variable_list'][target_variable_key]):
				target_value = 0.0
			else:
				target_value = sector_object.grid_sectors[code]['total_variable_list'][target_variable_key]
			for key in variable_keys:
				total_variable += sector_object.grid_sectors[code]['total_variable_list'][key]
			target_var.append((code, total_variable, target_value))
		except:
			error_target_var.append(code)

	if sort:
		target_var = sorted(target_var, key=lambda x: x[sortidx])

	# Transform into np.matrix
	varx = []
	tary = []
	for c, target, var in target_var:
		tary.append(target)
		varx.append(var)

	y = np.transpose(np.array(tary))
	X = np.transpose(np.array(varx))

	return target_var, y, X.reshape(-1, 1), error_target_var

# def get_variable_distribution(gridmap, order='dsc'):
# 		distribution = []
# 		for l in range(0, gridmap.step):
# 			for c in range(0, gridmap.step):
# 				if gridmap.grid[l][c]['in_territory'] == True:
# 					attr = gridmap.grid[l][c]['total_variable']
# 					distribution.append([attr, (l, c)])

# 		if order=='asc':
# 			reverse = False
# 		elif order=='dsc':
# 			reverse = True

# 		return sorted(distribution, key = lambda x: float(x[0]), reverse=reverse)

def linear_transformation(cell_distribution, mode="minmax"):
	#cell_distribution format (attr_value, cell)
	if mode == "zscore":
		mean_attr = np.mean(np.array([attr[0] for attr in cell_distribution]))
		sd_attr = np.std(np.array([attr[0] for attr in cell_distribution]))
		ord_distribution = [[(value[0] - float(mean_attr))/float(sd_attr), value[1]] for value in cell_distribution]
		parameters = (mean_attr, sd_attr)

	elif mode == "minmax":
		max_attr = max([attr[0] for attr in cell_distribution])
		min_attr = min([attr[0] for attr in cell_distribution])
		ord_distribution = [[(value[0] - float(min_attr))/(float(max_attr) - float(min_attr)), value[1]] for value in cell_distribution]
		parameters = (min_attr, max_attr)

	elif mode == "centered":
		mean_attr = np.mean(np.array([attr[0] for attr in cell_distribution]))
		ord_distribution = [[(value[0] - float(mean_attr)), value[1]] for value in cell_distribution]
		parameters = (mean_attr, '')

	else:
		ord_distribution = [[value[0], value[1]] for value in cell_distribution]
		parameters = None

	return ord_distribution, parameters

import math

def _haversine(lat1, lon1, lat2, lon2):
	R = 6378.137
	dLat = lat2 * math.pi / 180 - lat1 * math.pi / 180
	dLon = lon2 * math.pi / 180 - lon1 * math.pi / 180
	a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(lat1 * math.pi / 180) * math.cos(lat2 * math.pi / 180) * math.sin(dLon/2) * math.sin(dLon/2)
	c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
	d = R * c
	d * 1000

	return d

def _get_perp( x1, y1, x2, y2, x, y):
	xx = x2 - x1
	yy = y2 - y1
	shortest_length = ((xx * (x - x1)) + (yy * (y - y1))) / ((xx * xx) + (yy * yy))
	x4 = x1 + xx * shortest_length
	y4 = y1 + yy * shortest_length
	if (x4 <= x2) and (x4 >= x1) and (y4 <= y2) and (y4 >= y1):
		return (x4, y4)
	return None

def _near_line(x2, y2, x1, y1, x3, y3, meters):
	cpoint = _get_perp(float(x1), float(y1), float(x2), float(y2), float(x3), float(y3))
	if cpoint != None:
		distance = _haversine(cpoint[0], cpoint[1], float(x3), float(y3))
		if distance <= meters:
			return True

	return False

def _near_point(x2, y2, x1, y1, meters):
	distance = _haversine(float(x2), float(y2), float(x1), float(y1))
	if distance <= meters:
		return True
	else:
		return False


def gen_traintest_lists_sectormap( cities_objects = {}, border_distance=0.150, filename="PointDistribution", path="" ):
	#cities_objects = SectorMap datastructs and points_dataframe = Pandas dataframe with points, accessed by key: 'RJ': [gridsector, points_dataframe, variable_keys]
	#border_distance = Distance to consider with points near cell or polygon border.
	#filename = Filename prefix
	#path = Dir path to save distribution lists

	#attr_distribution = gen_attr_cell_distribution( sector_objects=sector_objects, variable_keys=variable_keys, sort=True )
	fw_distribution = open(os.path.join(path, filename + ".csv"), 'w')
	fw_train = open(os.path.join(path, filename + "_train.csv"), 'w')
	fw_test = open(os.path.join(path, filename + "_test.csv"), 'w')

	for city in cities_objects.keys():
		gridsector = cities_objects[city][0]
		points_dataframe = cities_objects[city][1]
		variable_keys = cities_objects[city][2]

		count = 0
		for row in points_dataframe.itertuples():
			count+=1
			#Find what sector the point belongs - Use external function
			x, y = gridsector.basemap( float(row.lon), float(row.lat) )
			point = Pt(x, y)
			sector_code = None
			for code in gridsector.sector_codes:
				if point.within(gridsector.grid_sectors[code]['sector_polygon']):
					sector_code = code
					break
			#

			if sector_code != None:
				train_or_test = gridsector.grid_sectors[sector_code]['train_or_test']

				#Find total_variable value - Use external function
				total_variable = 0
				for key in variable_keys:
					if key in gridsector.grid_sectors[sector_code]['total_variable_list'].keys():
						total_variable += gridsector.grid_sectors[sector_code]['total_variable_list'][key]
				#
				attr_value = float(total_variable)
				fw_distribution.write(row.id + ";" + str(city)+"-"+str(sector_code) + ";" + row.lat + ";" + row.lon + ";" + str(attr_value) + ";" + str(train_or_test) + ";" + str(variable_keys) + "\n")

				#Check if point is near polygon lines.
				lons = [ lon for lon in gridsector.grid_sectors[sector_code]['sector_shape_lon'] ]
				lats = [ lat for lat in gridsector.grid_sectors[sector_code]['sector_shape_lat'] ]
				bordercheck = []

				for lonpol, latpol in zip(lons, lats):
					for lop, lap in zip(lonpol, latpol):
						bordercheck.append( _near_point( float(lop), float(lap), float(row.lon), float(row.lat), border_distance ) )

				if train_or_test == 'train':
					if True not in bordercheck:
						fw_train.write(row.id + ";" + str(city)+"-"+str(sector_code) + ";" + row.lat + ";" + row.lon + ";" + str(attr_value) + "\n")

				if train_or_test == 'test':
					if True not in bordercheck:
						fw_test.write(row.id + ";" + str(city)+"-"+str(sector_code) + ";" + row.lat + ";" + row.lon + ";" + str(attr_value) + "\n")


	fw_distribution.close()
	fw_train.close()
	fw_test.close()


def gen_traintest_lists( gridmap, points_dataframe, lintransform="minmax", border_distance=0.150, filename="PointDistribution", path="" ):
	#gridmap = GridMap or SectorMap datastruct
	#points_dataframe = Pandas dataframe with points
	#linetransform = "minmax", "centered", "zorder"
	#border_distance = Distance to consider with points near cell or polygon border.
	#filename = Filename prefix
	#path = Dir path to save distribution lists


	#Get parameters for linear transformation
	cell_distribution = gridmap.get_variable_distribution(order='asc')
	_dist, parameters = linear_transformation(cell_distribution, mode=lintransform)

	print(parameters)

	fw_distribution = open(os.path.join(path, filename + ".csv"), 'w')
	fw_train = open(os.path.join(path, filename + "_train.csv"), 'w')
	fw_test = open(os.path.join(path, filename + "_test.csv"), 'w')

	count = 0
	points = points_dataframe
	for row in points.itertuples():
		count+=1
		cell = gridmap.find_cell(float(row.lat), float(row.lon))
		train_or_test = gridmap.grid[cell[0]][cell[1]]['train_or_test']
		original_value = float(gridmap.grid[cell[0]][cell[1]]['total_variable'])

		if lintransform == "minmax":
			attr_value = ( float(gridmap.grid[cell[0]][cell[1]]['total_variable']) - float(parameters[0]) ) / ( float(parameters[1]) - float(parameters[0]) )
		elif lintransform == "centered":
			attr_value = float(gridmap.grid[cell[0]][cell[1]]['total_variable']) - float(parameters[0])
		elif lintransform == "zscore":
			attr_value = ( float(gridmap.grid[cell[0]][cell[1]]['total_variable']) - float(parameters[0]) ) / float(parameters[1])

		else:
			attr_value = float(gridmap.grid[cell[0]][cell[1]]['total_variable'])

		fw_distribution.write(row.id + ";" + str(cell[0])+"-"+str(cell[1]) + ";" + row.lat + ";" + row.lon + ";" + str(attr_value) + ";" + str(train_or_test) + ";" + str(original_value) + "\n")
		if train_or_test == 'train':
			# Test if point is above borderlines
			border1_online = gridmap._near_line( gridmap.grid[cell[0]][cell[1]]['leftlon'], gridmap.grid[cell[0]][cell[1]]['upperlat'], gridmap.grid[cell[0]][cell[1]]['leftlon'], gridmap.grid[cell[0]][cell[1]]['lowerlat'], float(row.lon), float(row.lat), border_distance)
			border2_online = gridmap._near_line( gridmap.grid[cell[0]][cell[1]]['rightlon'], gridmap.grid[cell[0]][cell[1]]['upperlat'], gridmap.grid[cell[0]][cell[1]]['rightlon'], gridmap.grid[cell[0]][cell[1]]['lowerlat'], float(row.lon), float(row.lat), border_distance)
			border3_online = gridmap._near_line( gridmap.grid[cell[0]][cell[1]]['rightlon'], gridmap.grid[cell[0]][cell[1]]['lowerlat'], gridmap.grid[cell[0]][cell[1]]['leftlon'], gridmap.grid[cell[0]][cell[1]]['lowerlat'], float(row.lon), float(row.lat), border_distance)
			border4_online = gridmap._near_line( gridmap.grid[cell[0]][cell[1]]['rightlon'], gridmap.grid[cell[0]][cell[1]]['upperlat'], gridmap.grid[cell[0]][cell[1]]['leftlon'], gridmap.grid[cell[0]][cell[1]]['upperlat'], float(row.lon), float(row.lat), border_distance)
			if (border1_online or border2_online or border3_online or border4_online) == False:
				fw_train.write(row.id + ";" + str(cell[0])+"-"+str(cell[1]) + ";" + row.lat + ";" + row.lon + ";" + str(attr_value) + "\n")

		elif train_or_test == 'test':
			# Test if point is above borderlines
			border1_online = gridmap._near_line( gridmap.grid[cell[0]][cell[1]]['leftlon'], gridmap.grid[cell[0]][cell[1]]['upperlat'], gridmap.grid[cell[0]][cell[1]]['leftlon'], gridmap.grid[cell[0]][cell[1]]['lowerlat'], float(row.lon), float(row.lat), border_distance)
			border2_online = gridmap._near_line( gridmap.grid[cell[0]][cell[1]]['rightlon'], gridmap.grid[cell[0]][cell[1]]['upperlat'], gridmap.grid[cell[0]][cell[1]]['rightlon'], gridmap.grid[cell[0]][cell[1]]['lowerlat'],float(row.lon), float(row.lat), border_distance)
			border3_online = gridmap._near_line( gridmap.grid[cell[0]][cell[1]]['rightlon'], gridmap.grid[cell[0]][cell[1]]['lowerlat'], gridmap.grid[cell[0]][cell[1]]['leftlon'], gridmap.grid[cell[0]][cell[1]]['lowerlat'], float(row.lon), float(row.lat), border_distance)
			border4_online = gridmap._near_line( gridmap.grid[cell[0]][cell[1]]['rightlon'], gridmap.grid[cell[0]][cell[1]]['upperlat'], gridmap.grid[cell[0]][cell[1]]['leftlon'], gridmap.grid[cell[0]][cell[1]]['upperlat'], float(row.lon), float(row.lat), border_distance)
			if (border1_online or border2_online or border3_online or border4_online) == False:
				fw_test.write(row.id + ";" + str(cell[0])+"-"+str(cell[1]) + ";" + row.lat + ";" + row.lon + ";" + str(attr_value) + "\n")

	fw_distribution.close()
	fw_train.close()
	fw_test.close()






