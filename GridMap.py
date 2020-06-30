# coding: utf-8

import numpy as np
from mpl_toolkits.basemap import Basemap
from matplotlib.mlab import griddata
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import json

from shapely.geometry import Polygon
from shapely.geometry import Point
import math
import os, sys

from shutil import copy2
from random import randint
from scipy.misc import imread

def _create_basemap( llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat, resolution='l', epsg = 5641):
	return Basemap(llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat, resolution, epsg)

def _haversine(lat1, lon1, lat2, lon2):
	R = 6378.137
	dLat = lat2 * math.pi / 180 - lat1 * math.pi / 180
	dLon = lon2 * math.pi / 180 - lon1 * math.pi / 180
	a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(lat1 * math.pi / 180) * math.cos(lat2 * math.pi / 180) * math.sin(dLon/2) * math.sin(dLon/2)
	c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
	d = R * c
	#d * 1000

	return d

def _random_list(sample_size, range_size):
	rlist = list()
	while(sample_size):
		x = randint(1, range_size)
		if x not in rlist:
			rlist.append(x)
			sample_size-=1

	return rlist

def _on_line(lat1, lon1, lat2, lon2, lat_test, lon_test):
	if (lon2 - lon1) == 0:
		if (lat_test <= lat2) and (lat_test >= lat1) and (lon_test == lon2):
			return True
	elif (lat2 - lat1) == 0:
		if(lon_test <= lon2) and (lon_test >= lon1) and (lat_test == lat2):
			return True
		else:
			return False
	else:
		return False

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

class Window:
	def __init__(self, wintype, ystart, yend, mstart = 1, mend = 12):
		# default all struct set by year
		self.years = [i for i in range(ystart, yend + 1)]
		self.window = list()

		wincount = 1
		if wintype == 'year':
			for y in self.years:
				self.window.append((y, wincount))
				wincount += 1

		else:
			wincount = 1
			if wintype == 'month':
				for y in self.years:
					for m in range(mstart, mend + 1):
						self.window.append((y, int(str(m).zfill(2)), wincount))
						wincount += 1

class VariablePoint:
	def __init__(self, basemap, lon, lat, name=''):
		self.lon = lon
		self.lat = lat
		self.x, self.y = basemap( lon, lat )
		self.name = name

		#Use values when computing total distributions
		self.value = None
		self.norm_value = None

class StreetPoint:
	def __init__(self, basemap, id, lon, lat):
		self.id = id
		self.lon = lon
		self.lat = lat
		self.x, self.y = basemap( lon, lat )

		#Use values when computing total distributions
		self.value = None
		self.norm_value = None


class GridMap:
	#lowerleftlon, lowerleftlat = x1,y1
	#uprightlon, uprightlat
	def __init__(self, lowerleftlon, lowerleftlat, uprightlon, uprightlat, step):
		self.grid = [[0 for x in range(step)] for y in range(step)]
		self.step = step
		self.lowerleftlon = lowerleftlon
		self.lowerleftlat = lowerleftlat
		self.uprightlon = uprightlon
		self.uprightlat = uprightlat
		self.window = ""
		self.labels = list()

		inc_lon = (uprightlon - lowerleftlon) / step
		inc_lat = (uprightlat - lowerleftlat) / step
		cen_lon = ((uprightlon - lowerleftlon) / step) / 2
		cen_lat = ((uprightlat - lowerleftlat) / step) / 2

		cell_lowerleftlat = lowerleftlat

		for l in range(0, step):
			cell_lowerleftlon = lowerleftlon
			for c in range(0, step):
				cell = {'leftlon': cell_lowerleftlon,
						'lowerlat': cell_lowerleftlat,
						'rightlon': cell_lowerleftlon + inc_lon,
						'upperlat': cell_lowerleftlat + inc_lat,
						'cell': (l, c),
						'centroid': [cell_lowerleftlon + cen_lon, cell_lowerleftlat + cen_lat],
						'total_variable': 0,
						'variable_points': list(),
						'total_street_points': 0,
						'street_points': list(),
						'date_time': list(),
						'in_territory': False,
						'variable_per_window': {},
						'label': None,
						'train_or_test': None}

				self.grid[l][c] = cell

				cell_lowerleftlon += inc_lon
			cell_lowerleftlat += inc_lat


	def read(self):
		return self.grid

	def read_cell(self, l, c):
		return self.grid[l][c]

	def find_cell(self, lat, lon):
		#retorna linha e coluna
		for l in range(0, self.step):
			for c in range(0, self.step):
				if (self.grid[l][c]['leftlon'] <= lon <= self.grid[l][c]['rightlon']) and (self.grid[l][c]['lowerlat'] <= lat <= self.grid[l][c]['upperlat']):
					return (l, c)

	def test_borders(self, l, c, lon, lat, distance_meters):
		border1_online = self._near_line( self.grid[l][c]['leftlon'], self.grid[l][c]['upperlat'], self.grid[l][c]['leftlon'], self.grid[l][c]['lowerlat'], float(lon), float(lat), float(distance_meters))
		border2_online = self._near_line( self.grid[l][c]['rightlon'], self.grid[l][c]['upperlat'], self.grid[l][c]['rightlon'], self.grid[l][c]['lowerlat'], float(lon), float(lat), float(distance_meters))
		border3_online = self._near_line( self.grid[l][c]['rightlon'], self.grid[l][c]['lowerlat'], self.grid[l][c]['leftlon'], self.grid[l][c]['lowerlat'], float(lon), float(lat), float(distance_meters))
		border4_online = self._near_line( self.grid[l][c]['rightlon'], self.grid[l][c]['upperlat'], self.grid[l][c]['leftlon'], self.grid[l][c]['upperlat'], float(lon), float(lat), float(distance_meters))

		#Return FALSE if NOT near border, TRUE it near border
		return (border1_online or border2_online or border3_online or border4_online)

	def add_variable(self, l, c, VariablePoint, test_borders=True, distance_meters=0.150, in_territory=True, date=0, time=0):
		# add variable to cell, test if is near borders, and if inside territory (needs set_borders method first)
		if test_borders == True and in_territory == True:
			if self.test_borders( l, c, VariablePoint.lon, VariablePoint.lat, distance_meters ) == False:
				if self.grid[l][c]['in_territory'] == True:
					self.grid[l][c]['variable_points'].append(VariablePoint)
					self.grid[l][c]['total_variable'] += 1

		if test_borders == True and in_territory == False:
			if self.test_borders( l, c, VariablePoint.lon, VariablePoint.lat, distance_meters ) == False:
				self.grid[l][c]['variable_points'].append(VariablePoint)
				self.grid[l][c]['total_variable'] += 1

		if test_borders == False and in_territory == True:
			if self.grid[l][c]['in_territory'] == True:
				self.grid[l][c]['variable_points'].append(VariablePoint)
				self.grid[l][c]['total_variable'] += 1

		if test_borders == False and in_territory == False:
			self.grid[l][c]['variable_points'].append(VariablePoint)
			self.grid[l][c]['total_variable'] += 1

			#self.grid[l][c]['date_time'].append((date, time))

	def add_street_point(self, l, c, StreetPoint, test_borders=True, distance_meters=0.150, in_territory=True):
		# add street point to cell, test if is near borders and if is inside territory (needs set_borders first)
		if test_borders == True and in_territory == True:
			if self.test_borders( l, c, StreetPoint.lon, StreetPoint.lat, distance_meters ) == False:
				if self.grid[l][c]['in_territory'] == True:
					self.grid[l][c]['street_points'].append(StreetPoint)
					self.grid[l][c]['total_street_points'] += 1

		if test_borders == True and in_territory == False:
			if self.test_borders( l, c, StreetPoint.lon, StreetPoint.lat, distance_meters ) == False:
				self.grid[l][c]['street_points'].append(StreetPoint)
				self.grid[l][c]['total_street_points'] += 1

		if test_borders == False and in_territory == True:
			if self.grid[l][c]['in_territory'] == True:
				self.grid[l][c]['street_points'].append(StreetPoint)
				self.grid[l][c]['total_street_points'] += 1

		if test_borders == False and in_territory == False:
			self.grid[l][c]['street_points'].append(StreetPoint)
			self.grid[l][c]['total_street_points'] += 1

	def set_variable_by_cell(self, l, c, variable_value=None):
		self.grid[l][c]['total_variable'] = variable_value

	def get_max_variable(self):
		maxim = 0
		for l in range(0, self.step):
			for c in range(0, self.step):
				if self.grid[l][c]['total_variable'] > maxim:
					maxim = self.grid[l][c]['total_variable']
					all_max = self.grid[l][c]
		return all_max

	def get_min_variable(self):
		maxim = self.get_max_variable()['total_variable']
		minim = maxim
		for l in range(0, self.step):
			for c in range(0, self.step):
				#if (self.grid[l][c]['total_variable'] < minim) and (self.grid[l][c]['total_variable'] >= 0):
				if (self.grid[l][c]['total_variable'] < minim):
					minim = self.grid[l][c]['total_variable']
					all_min = self.grid[l][c]
		return all_min

	def get_variable_by_cell(self, lowercell, uppercell, inTerritory=False):
		variable_by_cell = list()
		for l in range(lowercell[0], uppercell[0]+1):
			for c in range(lowercell[1], uppercell[1]+1):
				if inTerritory == True:
					if self.grid[l][c]['in_territory'] == True:
						variable_by_cell.append([(l, c), self.grid[l][c]['total_variable']])
				else:
					variable_by_cell.append([(l, c), self.grid[l][c]['total_variable']])

		return variable_by_cell

	def get_maxtotal_streetpoints(self):
		maxim = 0
		for l in range(0, self.step):
			for c in range(0, self.step):
				if self.grid[l][c]['total_street_points'] > maxim:
					maxim = self.grid[l][c]['total_street_points']
					all_max = self.grid[l][c]
		return all_max

	def get_mintotal_streetpoints(self):
		maxim = self.get_maxtotal_streetpoints()['total_street_points']
		minim = maxim
		for l in range(0, self.step):
			for c in range(0, self.step):
				if (self.grid[l][c]['total_street_points'] < minim) and (self.grid[l][c]['total_street_points'] > 0):
					minim = self.grid[l][c]['total_street_points']
					all_min = self.grid[l][c]
		return all_min

	def set_borders(self, basemap, shapearray, force=False, key=None, value=None, shapeinfo=None):
		in_borders = []
		defined_shape = []
		point_list = []

		for l in range(0, self.step):
			for c in range(0, self.step):
				point_list.append((l, c))

		if force == True:
			for l in range(0, self.step):
				for c in range(0, self.step):
					self.grid[l][c]['in_territory'] = True
					in_borders.append((l, c))

		else:
			if key != None and value != None and shapeinfo != None:
				for info, shape in zip(shapeinfo, shapearray):
					if info[key] == value:
						defined_shape.append(shape)
			else:
				for shape in shapearray:
					defined_shape.append(shape)

			for l, c in point_list:
				x, y = basemap(self.grid[l][c]['centroid'][0], self.grid[l][c]['centroid'][1])
				pt = Point(x, y)

				for shape in defined_shape:
					poly = Polygon(shape)
					if pt.within(poly) == True:
						in_borders.append((l, c))

			for l, c in in_borders:
				self.grid[l][c]['in_territory'] = True

		return in_borders


	def add_attributes(self, l, c, attribute, window):
		if attribute in self.grid[l][c]['attributes_per_window']:
			if window in self.grid[l][c]['attributes_per_window'][attribute]:
				self.grid[l][c]['attributes_per_window'][attribute][window] += 1
			else:
				self.grid[l][c]['attributes_per_window'][attribute][window] = 1
		else:
			self.grid[l][c]['attributes_per_window'][attribute] = { window: 1 }


	def add_variable_per_window(self, l, c, window):
		if window in self.grid[l][c]['variable_per_window']:
			self.grid[l][c]['variable_per_window'][window] += 1
		else:
			self.grid[l][c]['variable_per_window'][window] = 1


	def set_window(self, windowstruct):
		self.window = windowstruct


	def set_labels_per_range(self, variable_cell_list, label_list):

		variable_list = variable_cell_list[:]
		variable_with_label = list()
		variable_list = sorted(variable_list, key = lambda x: int(x[1]))

		n = len(label_list)
		block_label = 0
		for i in range(n, 0, -1):
			group = round(len(variable_list)/i)
			c = variable_list[:int(group)]
			variable_list = variable_list[int(group):]
			for j in c:
				j.append(label_list[block_label])
				variable_with_label.append(j)
			block_label+=1

		for cells in variable_with_label:
			cell = cells[0]
			self.grid[cell[0]][cell[1]]['label'] = cells[2]

		return variable_with_label


	def set_grid_labels(self, labelslist):
		self.labels = labelslist


	def set_labels_per_cell(self, lowercell, uppercell, label, train_or_test):
		for l in range(lowercell[0], uppercell[0]+1):
			for c in range(lowercell[1], uppercell[1]+1):
				if label != "":
					self.grid[l][c]['label'] = label
				self.grid[l][c]['train_or_test'] = train_or_test


	def _near_line(self, x2, y2, x1, y1, x3, y3, meters):
		cpoint = _get_perp(float(x1), float(y1), float(x2), float(y2), float(x3), float(y3))
		if cpoint != None:
			distance = _haversine(cpoint[0], cpoint[1], float(x3), float(y3))
			if distance <= meters:
				return True

		return False

	def save(self, filename = "Model.grid"):
		try:
			with open(filename, 'w') as fout:
				json.dump(self.grid, fout)
			return True

		except:
			return "Error while saving file."

	def load(self, filename = "Model.grid"):
		try:
			with open(filename, 'r') as fin:
				g = json.load(fin)
			self.grid = g
			return True
		except:
			return "Error while loading file."


	def get_variable_distribution(self, order='dsc', streetpoints=False, train_or_test=None, cell_list=None, streetpoints_only=True):
		#train_or_test = True include only train and test cells
		#cell_list = ['23-24',..., '50-90', '30-10']
		distribution = []
		if cell_list == None:
			for l in range(0, self.step):
				for c in range(0, self.step):
					if self.grid[l][c]['in_territory'] == True:
						if train_or_test == None:
							attr = self.grid[l][c]['total_variable']
							label = self.grid[l][c]['train_or_test']
							street = self.grid[l][c]['total_street_points']

							if streetpoints == False:
								distribution.append([attr, (l, c), label])
							else:
								distribution.append([attr, (l, c), label, street])

						if type(train_or_test) is str:
							if self.grid[l][c]['train_or_test'] == train_or_test:
								attr = self.grid[l][c]['total_variable']
								label = self.grid[l][c]['train_or_test']
								street = self.grid[l][c]['total_street_points']

								if streetpoints == False:
									distribution.append([attr, (l, c), label])
								else:
									distribution.append([attr, (l, c), label, street])

						elif type(train_or_test) is list:
							if self.grid[l][c]['train_or_test'] in train_or_test and streetpoints_only == False:
								attr = self.grid[l][c]['total_variable']
								label = self.grid[l][c]['train_or_test']
								street = self.grid[l][c]['total_street_points']

								if streetpoints == False and street == 0:
									distribution.append([attr, (l, c), label])
								elif street != 0:
									distribution.append([attr, (l, c), label, street])

							if self.grid[l][c]['train_or_test'] in train_or_test and streetpoints_only == True:
								attr = self.grid[l][c]['total_variable']
								label = self.grid[l][c]['train_or_test']
								street = self.grid[l][c]['total_street_points']

								if streetpoints == False and street != 0:
									distribution.append([attr, (l, c), label])
								elif street != 0:
									distribution.append([attr, (l, c), label, street])

		else:
			for lc in cell_list:
				l, c = lc.split('-')
				attr = self.grid[int(l)][int(c)]['total_variable']
				label = self.grid[int(l)][int(c)]['train_or_test']
				street = self.grid[int(l)][int(c)]['total_street_points']

				if streetpoints == False:
					distribution.append([attr, (l, c), label])
				else:
					distribution.append([attr, (l, c), label, street])

		if order=='asc':
			reverse = False
		elif order=='dsc':
			reverse = True

		return sorted(distribution, key = lambda x: float(x[0]), reverse=reverse)


#### Plot Methods ####
	def plot_grid(self, basemap, marksize, linewidth):
		for l in range(0, self.step):
			for c in range(0, self.step):
				left_lon, right_lon = basemap([self.grid[l][c]['leftlon'], self.grid[l][c]['rightlon']], [self.grid[l][c]['lowerlat'], self.grid[l][c]['lowerlat']])
				low_lat, up_lat = basemap([self.grid[l][c]['rightlon'], self.grid[l][c]['rightlon']], [self.grid[l][c]['lowerlat'], self.grid[l][c]['upperlat']])
				centroid = basemap(self.grid[l][c]['centroid'][0], self.grid[l][c]['centroid'][1])
				basemap.plot([left_lon[0], left_lon[1]], [right_lon[0], right_lon[1]], 'bo-', markersize=marksize, linewidth=0.6)
				basemap.plot([low_lat[0], low_lat[1]], [up_lat[0], up_lat[1]], 'bo-', markersize=marksize, linewidth=0.6)
				basemap.plot(centroid[0], centroid[1], 'bo-', markersize=marksize, linewidth=linewidth)

	def plot_subgrid(self, subbasemap, lower_left_cell, upper_right_cell, marksize=0, linewidth=0.6 ):
		for l in range(lower_left_cell[0], upper_right_cell[0] + 1):
			for c in range(lower_left_cell[1], upper_right_cell[1] + 1):
				left_lon, right_lon = subbasemap([self.grid[l][c]['leftlon'], self.grid[l][c]['rightlon']], [self.grid[l][c]['lowerlat'], self.grid[l][c]['lowerlat']])
				low_lat, up_lat = subbasemap([self.grid[l][c]['rightlon'], self.grid[l][c]['rightlon']], [self.grid[l][c]['lowerlat'], self.grid[l][c]['upperlat']])
				centroid = subbasemap(self.grid[l][c]['centroid'][0], self.grid[l][c]['centroid'][1])
				subbasemap.plot([left_lon[0], left_lon[1]], [right_lon[0], right_lon[1]], 'bo-', markersize=marksize, linewidth=0.6)
				subbasemap.plot([low_lat[0], low_lat[1]], [up_lat[0], up_lat[1]], 'bo-', markersize=marksize, linewidth=0.6)
				subbasemap.plot(centroid[0], centroid[1], 'bo-', markersize=marksize, linewidth=linewidth)

	def get_subbasemap(self, lower_left_cell, upper_right_cell, resolution='l', epsg= 5641 ):
		return _create_basemap( llcrnrlon=self.grid[lower_left_cell[0]][lower_left_cell[1]]['leftlon'],
									  llcrnrlat=self.grid[lower_left_cell[0]][lower_left_cell[1]]['lowerlat'],
									  urcrnrlon=self.grid[upper_right_cell[0]][upper_right_cell[1]]['rightlon'],
									  urcrnrlat=self.grid[upper_right_cell[0]][upper_right_cell[1]]['upperlat'],
									  resolution='l', epsg = 5641 )

	def plot_variable(self, basemap, lon, lat, mark, size, linewidth):
		self.plot_grid(basemap, 1, linewidth)
		lon_, lat_ = basemap(lon, lat)
		basemap.plot(lon_, lat_, mark, markersize=size)


	def distribute_variable(self, basemap, data, linewidth):
		self.plot_grid(basemap, 0, linewidth)

		total_variable = []
		ylat = []
		xlon = []
		for dlat, dlon in zip(data.lat.values, data.lon.values):
			c = self.find_cell(dlat, dlon)
			print(c)
			if not c:
				total_variable.append(0)
			else:
				total_variable.append(self.grid[c[0]][c[1]]['total_variable'])

		for dlat, dlon in zip(data.lat.values, data.lon.values):
			x, y = basemap(dlon, dlat)
			xlon.append(x)
			ylat.append(y)

		numcols, numrows = self.step, self.step

		xi = np.linspace(basemap.llcrnrx, basemap.urcrnrx, numcols)
		yi = np.linspace(basemap.llcrnry, basemap.urcrnry, numrows)
		xi, yi = np.meshgrid(xi, yi)

		x, y, z = xlon, ylat, total_variable
		zi = griddata(x, y, z, xi, yi, interp='linear')
		m = basemap.contourf(xi, yi, zi, vmin=1, vmax=self.get_max_variable()['total_variable'], alpha=0.5)
		cb = basemap.colorbar(m, location='bottom', pad="5%")


	def hot_spot(self, basemap, linewidth, colorbar=False, labelsize=25, cell_list=None, force_max_min=(None, None), alpha=0.5, cmap=plt.cm.jet, nticks=12):
		self.plot_grid(basemap, 0, linewidth)

		distribution = []
		total = 0

		for l in range(0, self.step):
			for c in range(0, self.step):
				if self.grid[l][c]['total_variable'] >= 0:
					if self.grid[l][c]['in_territory'] == True:
						if cell_list == None or (cell_list != None and (l, c) in cell_list):
							total+=1
							attr = self.grid[l][c]['total_variable']
							distribution.append(attr)

							left_lon1, right_lon1 = basemap([self.grid[l][c]['leftlon'], self.grid[l][c]['rightlon']], [self.grid[l][c]['lowerlat'], self.grid[l][c]['lowerlat']])
							left_lon2, right_lon2 = basemap([self.grid[l][c]['leftlon'], self.grid[l][c]['rightlon']], [self.grid[l][c]['upperlat'], self.grid[l][c]['upperlat']])
							low_lat1, up_lat1 = basemap([self.grid[l][c]['rightlon'], self.grid[l][c]['rightlon']], [self.grid[l][c]['lowerlat'], self.grid[l][c]['upperlat']])
							low_lat2, up_lat2 = basemap([self.grid[l][c]['leftlon'], self.grid[l][c]['leftlon']], [self.grid[l][c]['upperlat'], self.grid[l][c]['lowerlat']])

							data = pd.DataFrame([(left_lon1[0], right_lon1[0], self.grid[l][c]['total_variable']),
												(low_lat1[1], up_lat1[1], self.grid[l][c]['total_variable']),
												(low_lat2[0], up_lat2[0], self.grid[l][c]['total_variable']),
												(low_lat1[0], up_lat1[0], self.grid[l][c]['total_variable'])], columns=list('XYZ'))
							numcols, numrows = self.step + 2, self.step + 2
							xi = np.linspace(data.X.min(), data.X.max(), numcols)
							yi = np.linspace(data.Y.min(), data.Y.max(), numrows)
							xi, yi = np.meshgrid(xi, yi)

							x, y, z = data.X.values, data.Y.values, data.Z.values
							zi = griddata(x, y, z, xi, yi, interp='linear')
							if force_max_min == (None, None):
								cs = basemap.contourf(xi, yi, zi, vmin=self.get_min_variable()['total_variable'], vmax=self.get_max_variable()['total_variable'], cmap=cmap, alpha=alpha)
							else:
								cs = basemap.contourf(xi, yi, zi, vmin=force_max_min[0], vmax=force_max_min[1], cmap=cmap, alpha=alpha)

				sys.stdout.write("Progress/Total: %d/%d   \r" % (total, self.step * self.step))
				sys.stdout.flush()

		if colorbar == True:
			ord_distribution = np.sort(distribution)
			if force_max_min == (None, None):
				norm = mpl.colors.Normalize(vmin=self.get_min_variable()['total_variable'], vmax=ord_distribution[total-1])
			else:
				norm = mpl.colors.Normalize(vmin=force_max_min[0], vmax=force_max_min[1])
			sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
			sm.set_array([])
			cbar = basemap.colorbar(sm)

			if force_max_min == (None, None):
				m0=int(self.get_min_variable()['total_variable'])
				m4=int(self.get_max_variable()['total_variable'])
			else:
				m0=int(force_max_min[0])
				m4=int(force_max_min[1])

			# Use this for 4 spaced ticks
			# m1=int(1*(m4-m0)/4.0 + m0)
			# m2=int(2*(m4-m0)/4.0 + m0)
			# m3=int(3*(m4-m0)/4.0 + m0)
			# print(m0, m1, m2, m3, m4)
			# cbar.set_ticks([m0,m1,m2,m3,m4])
			# cbar.set_ticklabels([m0,m1,m2,m3,m4])

			# Use this for 12 spaced ticks
			# nticks = 12
			ticks = [m0] + [(i * (m4 - m0) / int(nticks) + m0) for i in range(1, int(nticks))] + [m4]
			print(ticks, len(ticks))
			cbar.set_ticks(ticks)
			cbar.set_ticklabels([int(t) for t in  ticks])

			cbar.ax.tick_params(labelsize=labelsize)


	def plot_variable_points(self, basemap, submap=False, lower_left_cell=None, upper_right_cell=None, name=None, color='g', mark='b', mark_size=5):
		lons = []
		lats = []
		if submap == False:
			self.plot_grid(basemap, 0, 1.0)
			start_l, start_c = 0, 0
			end_l, end_c = self.step, self.step

		else:
			self.plot_subgrid( basemap, lower_left_cell, upper_right_cell, 0, 1.0 )
			start_l, start_c = lower_left_cell[0], lower_left_cell[1]
			end_l, end_c = upper_right_cell[0]+1, upper_right_cell[1]+1

		for l in range(start_l, end_l):
			for c in range(start_c, end_c):
				for point in self.grid[l][c]['variable_points']:
					lons.append(point.lon)
					lats.append(point.lat)


		# for l in range(0, self.step):
		# 	for c in range(0, self.step):
		# 		for point in self.grid[l][c]['variable_points']:
		# 			lons.append(point.x)
		# 			lats.append(point.y)


		# for l in range(lower_left_cell[0], upper_right_cell[0]+1):
		# 	for c in range(lower_left_cell[1], upper_right_cell[1]+1):
		# 		for point in self.grid[l][c]['variable_points']:
		# 			lons.append(point.lon)
		# 			lats.append(point.lat)

		basemap.scatter(lons, lats, marker=mark, color=color, s=mark_size, alpha=0.5)

	def plot_street_points(self, basemap, color='b', mark='o', mark_size=8, alpha=0.5):
		self.plot_grid(basemap, 0, 1.0)
		lons = []
		lats = []
		for l in range(0, self.step):
			for c in range(0, self.step):
				for point in self.grid[l][c]['street_points']:
					if type(point) is tuple:
						x, y = basemap(float(point[2]), float(point[1]))
						print(x, y)
					else:
						x = point.x
						y = point.y

					lons.append(x)
					lats.append(y)

		basemap.scatter(lons, lats, marker=mark, color=color, s=mark_size, alpha=alpha)

	def street_points(self, basemap, corner_list, interpol_list, cmark='g', imark='b'):
		#corner_list, interpol_list = lat, lon
		self.plot_grid(basemap, 0, 1.0)

		corner_lats = []
		corner_lons = []
		interpol_lats = []
		interpol_lons = []

		for latlon in corner_list:
			lon, lat = basemap(latlon[1], latlon[0])
			corner_lons.append(lon)
			corner_lats.append(lat)

		for latlon in interpol_list:
			lon, lat = basemap(latlon[1], latlon[0])
			interpol_lons.append(lon)
			interpol_lats.append(lat)

		basemap.scatter(corner_lons, corner_lats, marker='D',color='m')
		basemap.scatter(interpol_lons, interpol_lats, marker='o',color='b')

	def image_points(self, basemap, filenameListFromDir, color='m'):
		self.plot_grid(basemap, 0, 0.6)
		lons = []
		lats = []

		fr = open(filenameListFromDir, 'r')
		pointListFromDir = fr.readlines()
		for point in pointListFromDir:
			point = point.replace("\n","")
			l, c = point.split(',')
			lon, lat = basemap(float(c), float(l))
			lons.append(lon)
			lats.append(lat)

		basemap.scatter(lons, lats, marker='+', color=color)

	def event_points(self, basemap, pointlist, marker='D', color='m'):
		self.plot_grid(basemap, 0, 1.0)
		lons = []
		lats = []
		for latlon in pointlist:
			lon, lat = basemap(latlon[1], latlon[0])
			lons.append(lon)
			lats.append(lat)

		basemap.scatter(lons, lats, marker=marker, color=color)


	def mark_cells(self, basemap, lowercell, uppercell, color='r'):
		self.plot_grid(basemap, 0, 1.0)
		for l in range(lowercell[0], uppercell[0]+1):
			for c in range(lowercell[1], uppercell[1]+1):
				centroid = basemap(self.grid[l][c]['centroid'][0], self.grid[l][c]['centroid'][1])
				basemap.plot(centroid[0], centroid[1], 'D', markersize=3.0, linewidth=3.0, color=color)


	def show_info(self, basemap, position=True, variable_value=False):
		self.plot_grid(basemap, 0, 1.0)
		if position==True:
			for l in range(0, self.step):
				for c in range(0, self.step):
				  lon, lat = basemap(self.grid[l][c]['centroid'][0], self.grid[l][c]['centroid'][1])
				  plt.annotate((l,c), (lon, lat), size=8)

		if variable_value==True:
			for l in range(0, self.step):
				for c in range(0, self.step):
				  lon, lat = basemap(self.grid[l][c]['centroid'][0], self.grid[l][c]['centroid'][1])
				  plt.annotate(self.grid[l][c]['total_variable'], (lon, lat), size=8)


# Street and shapefile manipulation methods

	# Convert shapearray to lon lat coordinates
	def convert_shape(self, basemap, shapearray):
		lonlatlines = []
		finalshape = {}
		street_counter = 0
		for shape in shapearray:
			x, y = zip(*shape)
			lonx, laty = basemap(x, y, inverse=True)
			finalshape[street_counter] = [(i,j) for i,j in zip(laty, lonx)]
			street_counter+=1

		return finalshape


	# Generate interpolated points
	def generate_inter_points(self, finalshape):
		new_points = {}
		differences = {}
		for key in finalshape.copy().keys():
			new_points[key] = []
			differences[key] = []
			for i in range(0, len(finalshape[key])):
				if i < len(finalshape[key]) - 1:
					current_lat = finalshape[key][i][0]
					current_lon = finalshape[key][i][1]
					next_lat = finalshape[key][i+1][0]
					next_lon = finalshape[key][i+1][1]

					interpos = []
					# check if the distance between the 2 points is more than 100 meters
					meters = _haversine(current_lat, current_lon, next_lat, next_lon)
					if meters > 0.1:
						#Interpolar 10 pontos
						distance_lat = next_lat - current_lat
						distance_lon = next_lon - current_lon
						inclat = distance_lat/4
						inclon = distance_lon/4

						for i in range(0, 3):
							prox_lat = current_lat + inclat
							prox_lon = current_lon + inclon
							interpos.append((prox_lat, prox_lon))
							current_lat = prox_lat
							current_lon = prox_lon

						new_points[key].append(interpos)
						differences[key].append(meters)

					else:
						continue

		return (new_points, differences)


	# Save pointlist corner and interpolated in file, return same arrays
	def gen_pointlist_from_shapefile(self, finalshape, interpolated, filename="pointListFromShapefile.list", in_territory=False):
		# finalshape from self.convert_shape, interpolated from self.generate_inter_points
		# format:
		# corner_list = dictionary {line: [points]}
		# interpol_list = tuple(dictionary {line: [points]}, diferences)
		# return corner_list, interpol_list

		corner_list = list()
		interpol_list = list()

		fw = open(filename, 'w')
		for line in finalshape.keys():
			for point in finalshape[line]:
				if in_territory==True:
					cell = self.find_cell(float(point[0]), float(point[1]))
					if cell and self.grid[cell[0]][cell[1]]['in_territory'] == True:
						corner_list.append(point)
						fw.write(str(point) + '\n')
				elif in_territory==False:
					corner_list.append(point)
					fw.write(str(point) + '\n')

		for line in interpolated[0].keys():
			for sect in interpolated[0][line]:
				for point in sect:
					if in_territory==True:
						cell = self.find_cell(float(point[0]), float(point[1]))
						if cell and self.grid[cell[0]][cell[1]]['in_territory'] == True:
							interpol_list.append(point)
							fw.write(str(point) + '\n')
					elif in_territory==False:
						interpol_list.append(point)
						fw.write(str(point) + '\n')

		fw.close()
		return corner_list, interpol_list


	# Generate a pointlist from a dir containing lat_lon_camera images
	def gen_pointlist_from_dir(self, path, tot='all', filename="pointListFromDir.list", metacheck=True):

		if metacheck == False:
			imagepath = path
			metapath = ""
		else:
			imagepath = path + "/images"
			metapath = path + "/meta"

		print(imagepath)
		file_list = sorted(os.listdir(imagepath))

		fw = open(filename, 'w')
		fw1 = open('removed_' + filename + '.log', 'w')
		fw2 = open('damaged_' + filename + '.log', 'w')

		latlons = list()
		for imagename in file_list:
			slices = imagename.split('_')
			if len(slices) > 3:
				line = slices[0]
				lat = slices[1]
				lon = slices[2]
			elif len(slices) == 3:
				line = None
				lat = slices[0]
				lon = slices[1]
			else:
				sys.exit("Wrong file name format: accept only [line_lat_lon_cam] or [lat_lon_cam]")

			latlons.append((line, lat, lon))

		latlons = list(set(latlons))

		if tot != 'all':
			lim = tot
		else:
			lim = len(latlons)

		for point in latlons[:lim]:
			allcamera = list()
			status = list()

			#Check if all camera views are present, can read, shape and dimensions
			for c in ['0', '90', '180', '270']:
				try:
					image = imread(str(imagepath) + '/' + str(point[1]) + '_' + str(point[2]) + '_' + c + '.jpg')
					if len(image.shape)!= 3 or image.shape[0] != 640 or image.shape[1] != 640:
						allcamera.append(c)
						fw2.write("channel or wrong shape," + str(imagepath) + '/' + str(point[1]) + '_' + str(point[2]) + '_' + c + '.jpg\n')
					if os.stat(str(imagepath) + '/' + str(point[1]) + '_' + str(point[2]) + '_' + c + '.jpg').st_size < 17000:
						allcamera.append(c)
						fw2.write("small size," + str(imagepath) + '/' + str(point[1]) + '_' + str(point[2]) + '_' + c + '.jpg\n')

				except:
					allcamera.append(c)
					fw2.write("Read error," +str(imagepath) + '/' + str(point[1]) + '_' + str(point[2]) + '_' + c + '.jpg\n')

			#Check metafile for OK status in images
			if metacheck == True:
				for c in ['0', '90', '180', '270']:
					try:
						with open(str(metapath) + '/' + str(point[1]) + '_' + str(point[2]) + '_' + c + '.txt', 'r') as fr:
							meta = json.load(fr)
						if meta['status'] != 'OK':
							status.append(meta['status'])
					except:
						fw1.write("Not found: " + str(metapath) + '/' + str(point[1]) + '_' + str(point[2]) + '_' + c + '.txt\n')
						pass

			if len(status) == 0 and len(allcamera) == 0:
				fw.write(point[1] + ',' + point[2] + '\n')
			else:
				fw1.write("Status Not OK," + str(status) +  ',' + str(metapath) + '/' + str(point[1]) + '_' + str(point[2]) + '\n')

		fw.close()
		fw1.close()
		fw2.close()


	#Read a point list in tuple format (lat, lon) and generate a pointlist
	def read_point_list(self, filename):
		pointlist = []
		fw = open(filename, 'r')
		lines = fw.readlines()
		for line in lines:
			line = line.replace('(','').replace(')','').replace('\n','')
			slices = line.split(',')
			lat = slices[0]
			lon = slices[1]
			pointlist.append((float(lat),float(lon)))

		return pointlist










