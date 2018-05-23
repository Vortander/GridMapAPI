# coding: utf-8

import numpy as np
from shapely.geometry import Polygon
from shapely.geometry import Point as Pt

import os, sys
from shutil import copy2
from random import randint
from scipy.misc import imread

#Bairros = Neighborhoods


class Point:
	
	def __init__( self, basemap, lon, lat ):
		self.lon = lon
		self.lat = lat
		self.x, self.y = basemap( lon, lat )
		
		self.cod_bairro = None
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
				 'cod_bairro' : self.pointmap[index].cod_bairro,
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
				 'cod_bairro' : self.pointmap[index].cod_bairro,
				 'total_variable' : self.pointmap[index].total_variable,
				 'label': self.pointmap[index].label,
				 'train_or_test': self.pointmap[index].train_or_test,
				 'in_territory': self.pointmap[index].in_territory
				}


class BairrosMap:
	
	def __init__(self, basemap, shapeinfo, shapearray, code_bairro_key, bairro_name_key=None):
		self.basemap = basemap
		self.shape_info = shapeinfo
		self.shape_array = shapearray
		self.code_bairro_key = code_bairro_key

		
		#Compute bairro list
		bairro_codes = list()
		for info, shape in zip( self.shape_info, self.shape_array ):
			bairro_codes.append(int(info[code_bairro_key]))
		self.code_bairros = set(bairro_codes)

		self.size = len(self.code_bairros)
		self.grid_bairros = [i for i in range(self.size+1)]

		for code in self.code_bairros:
			bairro = { 'index': code,
						'bairro_shape_x': list(),
						'bairro_shape_y': list(),
						'bairro_shape_lon': list(),
						'bairro_shape_lat': list(),
						'bairro_shape' : list(),
						'bairro_polygon': list(),
						'bairro_code': code,
						'bairro_name': list(),
						'attributes' : None,
						'total_variable' : 0,
						'label': None,
						'train_or_test': None,
						}
			self.grid_bairros[code] = bairro

		for info, shape in zip( self.shape_info, self.shape_array ):

			if bairro_name_key!=None:
				bairro_name = info[bairro_name_key]
			else:
				bairro_name = None

			code = int(info[code_bairro_key])

			x, y = zip(*shape)
			lon, lat = basemap(x, y, inverse=True)

			self.grid_bairros[code]['bairro_shape_x'].append(x)
			self.grid_bairros[code]['bairro_shape_y'].append(y)
			self.grid_bairros[code]['bairro_shape_lon'].append(lon)
			self.grid_bairros[code]['bairro_shape_lat'].append(lat)
			self.grid_bairros[code]['bairro_shape'].append(shape)
			self.grid_bairros[code]['bairro_polygon'].append(Polygon(shape))
			self.grid_bairros[code]['bairro_name'].append(bairro_name)


	def bairro_size(self, code, objct):
		return len(self.grid_bairros[int(code)][str(objct)])
	

	def check_point(self, cod_bairro, point):
		for i, info, shape in zip( range(self.size), self.shape_info, self.shape_array ):
			if info[self.code_bairro_key] == cod_bairro:
				#x, y = basemap(lon, lat)
				pt = Pt(point.x, point.y)
				poly = Polygon(shape)
				iswithin = pt.within(poly)
		
				return iswithin

	def check_point_bairro( self, cod_bairro, point):
		pt = Pt(point.x, point.y)
		for polygon in self.grid_bairros[cod_bairro]['bairro_polygon']:
			if pt.within(polygon):
				return True

		
	def set_point_bairro(self, point):
		for i, info, shape in zip( range(self.size), self.shape_info, self.shape_array ):
			iswithin = self.check_bairro(info[self.code_bairro_key], point)
			if iswithin == True:
				point.in_territory = True
				point.cod_bairro = info[self.code_bairro_key]
				point.total_variable = self.grid_bairros[i]['total_variable']
				point.label = self.grid_bairros[i]['label']
				point.train_or_test = self.grid_bairros[i]['train_or_test']


	def set_streetmap_bairro( self, streetmap ):
		for i, point in zip ( range(streetmap.size), streetmap.pointmap ):
			self.set_point_bairro( point )

	def set_pointlist_bairro( self, pointlist ):
		for i, point in zip ( range(pointlist), pointlist ):
			pass

			
	def set_variable_codbairro( self, cod_bairro, value=1 ):
		for bairro in self.grid_bairros:
			if bairro['bairro_code'] == cod_bairro:
				bairro['total_variable'] += value

	def set_variable_point( self, point ):
		for cod_bairro in self.code_bairros:
			if self.check_point_bairro( cod_bairro, point ):
				self.grid_bairros[cod_bairro]['total_variable'] += 1




			












