# coding: utf-8

import numpy as np
#from rtree import index as Indx

from shapely.geometry import Polygon
from shapely.geometry import Point as Pt
from shapely.geometry import shape as ShapeGeo

import os, sys
from shutil import copy2
from random import randint
from scipy.misc import imread

class Point:
	def __init__( self, basemap, lon, lat ):
		self.lon = lon
		self.lat = lat
		self.x, self.y = basemap( lon, lat )

		self.code_sector = None
		self.total_variable = None
		self.label = None
		self.train_or_test = None
		self.in_territory = None

class StreetMap:
	def __init__( self, basemap, street_shapeinfo, street_shapearray ):

		pointmap = []
		lonlat = []

		for info, shape in zip( street_shapeinfo, street_shapearray ):
			# extract points from streets
			for point in shape:
				lon, lat = basemap( point[0], point[1], inverse=True )
				lonlat.append( (lon, lat) )

		# Remove duplicate points (crossovers, corners)
		lonlat = list( set(lonlat) )
		for lon, lat in lonlat:
			pointmap.append( Point(basemap, lon, lat) )

		self.pointmap = pointmap
		self.size = len(pointmap)

	def get_pointstreetpoint( self, index ):
		return { 'index': index,
				 'lon' : self.pointmap[index].lon,
				 'lat' : self.pointmap[index].lat,
				 'x' : self.pointmap[index].x,
				 'y' : self.pointmap[index].y,
				 'code_sector' : self.pointmap[index].code_sector,
				 'total_variable' : self.pointmap[index].total_variable,
				 'label': self.pointmap[index].label,
				 'train_or_test': self.pointmap[index].train_or_test,
				 'in_territory': self.pointmap[index].in_territory
				}

class PointMap:
	def __init__( self, basemap, street_shapeinfo, street_shapearray ):

		pointmap = []
		lonlat = []

		for info, shape in zip( street_shapeinfo, street_shapearray ):
			# extract points from streets
			for point in shape:
				lon, lat = basemap( point[0], point[1], inverse=True )
				lonlat.append( (lon, lat) )

		# Remove duplicate points (crossovers, corners)
		lonlat = list( set(lonlat) )
		for lon, lat in lonlat:
			pointmap.append( Point(basemap, lon, lat) )

		self.pointmap = pointmap
		self.size = len(pointmap)

	def get_point( self, index ):
		return { 'index': index,
				 'lon' : self.pointmap[index].lon,
				 'lat' : self.pointmap[index].lat,
				 'x' : self.pointmap[index].x,
				 'y' : self.pointmap[index].y,
				 'code_sector' : self.pointmap[index].code_sector,
				 'total_variable' : self.pointmap[index].total_variable,
				 'label': self.pointmap[index].label,
				 'train_or_test': self.pointmap[index].train_or_test,
				 'in_territory': self.pointmap[index].in_territory
				}

class SectorMap:
	def __init__(self, basemap, shapeinfo, shapearray, code_sector_key, sector_name_key=None):
		self.basemap = basemap
		self.shape_info = shapeinfo
		self.shape_array = shapearray
		self.code_sector_key = code_sector_key
		self.sector_name_key = sector_name_key

		#Compute sector list
		sector_codes = list()
		for info, shape in zip( self.shape_info, self.shape_array ):
			sector_codes.append(int(info[code_sector_key]))
		self.sector_codes = set(sector_codes)

		self.size = len(self.sector_codes)

		#self.grid_sectors = [i for i in range(self.size+1)]
		self.grid_sectors = {}

		for code in self.sector_codes:
			sector = { 'index': code,
						'sector_shape_x': list(),
						'sector_shape_y': list(),
						'sector_shape_lon': list(),
						'sector_shape_lat': list(),
						'sector_shape' : list(),
						'sector_polygon': None, #TODO: sector_polygon should be a polygon with holes or a collection of polygons
						'sector_code': code,
						'sector_name': list(),
						'attributes' : None,
						'variable_points': {},
						'total_variable_list': {},
						'total_variable' : 0,
						'label': None,
						'train_or_test': None,
						}
			self.grid_sectors[code] = sector

		for info, shape in zip( self.shape_info, self.shape_array ):
			if self.sector_name_key != None:
				sector_name = info[self.sector_name_key]
			else:
				sector_name = None

			code = int(info[code_sector_key])
			x, y = zip(*shape)
			lon, lat = basemap(x, y, inverse=True)

			self.grid_sectors[code]['sector_shape_x'].append(x)
			self.grid_sectors[code]['sector_shape_y'].append(y)
			self.grid_sectors[code]['sector_shape_lon'].append(lon)
			self.grid_sectors[code]['sector_shape_lat'].append(lat)
			self.grid_sectors[code]['sector_shape'].append(shape)
			self.grid_sectors[code]['sector_name'].append(sector_name)

		for code in self.sector_codes:
			if len(self.grid_sectors[code]['sector_shape']) > 1: #If more than one shape make the sector, add the small areas as rings. TODO: Check in future if is a multipolygon.
				get_max_area = {}
				for i, shape in enumerate(self.grid_sectors[code]['sector_shape']):
					get_max_area[i] = { 'shape':shape, 'area':ShapeGeo(Polygon(shape)).area }

				max_area = max([(node, get_max_area[node]['area'], get_max_area[node]['shape']) for node in get_max_area], key=lambda x:x[1])
				del get_max_area[max_area[0]]

				self.grid_sectors[code]['sector_polygon'] = Polygon(max_area[2], holes=[ get_max_area[node]['shape'] for node in get_max_area ])

			else:
				self.grid_sectors[code]['sector_polygon'] = Polygon(self.grid_sectors[code]['sector_shape'][0])


	def sector_size(self, code, objct):
		return len(self.grid_sectors[int(code)][str(objct)])

	#TODO: This Method is possibly deprecated...check if is still needed.
	def check_point(self, code_sector, point):
		for i, info, shape in zip( range(self.size), self.shape_info, self.shape_array ):
			if info[self.code_sector_key] == code_sector:
				#x, y = basemap(lon, lat)
				pt = Pt(point.x, point.y)
				poly = Polygon(shape)
				iswithin = pt.within(poly)

				return iswithin

	def check_point_sector( self, code_sector, point):
		pt = Pt(point.x, point.y)
		#TODO: this assumes that the polygon is composed by one polygon only with or without holes. Modify to check adjacent non internal polygons.
		polygon = self.grid_sectors[code_sector]['sector_polygon']
		if pt.within(polygon):
			return True

	#Set a Point object as belonging to a specific sector
	def set_point_sector(self, point):
		for i, info, shape in zip( range(self.size), self.shape_info, self.shape_array ):
			iswithin = self.check_sector(info[self.code_sector_key], point)
			if iswithin == True:
				point.in_territory = True
				point.code_sector = info[self.code_sector_key]
				point.total_variable = self.grid_sectors[i]['total_variable']
				point.label = self.grid_sectors[i]['label']
				point.train_or_test = self.grid_sectors[i]['train_or_test']

	def set_streetmap_sector( self, streetmap ):
		for i, point in zip ( range(streetmap.size), streetmap.pointmap ):
			self.set_point_sector( point )

	def set_pointlist_sector( self, pointlist ):
		for i, point in zip ( range(pointlist), pointlist ):
			pass

	#TODO: Check if deprecated
	def set_variable_codsector( self, code_sector, value=1 ):
		for sector in self.grid_sectors:
			if sector['sector_code'] == code_sector:
				sector['total_variable'] += value

	def set_variable_point( self, point ):
		for code_sector in self.sector_codes:
			if self.check_point_sector( code_sector, point ):
				self.grid_sectors[code_sector]['total_variable'] += 1

	# Add variables with labels in total_variable_list dictionary.
	# Options include use rtree (needs rtree package) or not.
	def set_variable_point_to_list( self, point, variable_name='attribute', use_rtree=False, idx=None ):
		if use_rtree == True:
			#Construct rtree from shape_info and shape_array(polygons) from shapefile, reference by code_sector. See generate_rtree in shape.py.
			candidates = list(idx.intersection((point.x, point.y, point.x, point.y)))
		elif use_rtree == False:
			candidates = self.sector_codes
		for code_sector in candidates:
			if self.check_point_sector( code_sector, point ):
				if variable_name not in self.grid_sectors[code_sector]['total_variable_list'].keys():
					self.grid_sectors[code_sector]['total_variable_list'][variable_name] = 1
					self.grid_sectors[code_sector]['variable_points'][variable_name] = [point]
				else:
					self.grid_sectors[code_sector]['total_variable_list'][variable_name] += 1
					self.grid_sectors[code_sector]['variable_points'][variable_name].append(point)
				if use_rtree : break

	# Add variable value with label to total_variable_list, providing a code_sector.
	def set_variable_value_to_list( self, code_sector, variable_value=1, variable_name='attribute' ):
		if variable_name not in self.grid_sectors[code_sector]['total_variable_list'].keys():
			self.grid_sectors[code_sector]['total_variable_list'][variable_name] = variable_value
		else:
			localtotal = self.grid_sectors[code_sector]['total_variable_list'][variable_name]
			self.grid_sectors[code_sector]['total_variable_list'][variable_name] = localtotal + variable_value


	# TODO: Check if still needed
	def set_total_variable_name( self, total_variable_name='total_attribute', variable_name_list=None ):
		for code_sector in self.sector_codes:
			total_variable = 0.0

			if variable_name_list == None:
				variable_name_list == self.grid_sectors[code_sector]['total_variable_list'].keys()

			print(variable_name_list)

			for variable in variable_name_list:
				total_variable += float(self.grid_sectors[code_sector][variable])

			self.grid_sectors[code_sector]['total_variable_list'][total_variable_name] = total_variable




















