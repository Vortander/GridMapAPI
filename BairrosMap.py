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
		self.total_attributes = None
		self.label = None
		self.train_or_test = None
		self.in_territory = None




class Streets:
	pass


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
				 'total_attributes' : self.pointmap[index].total_attributes,
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
		self.size = len(shapearray)
		self.grid_bairros = [i for i in range(self.size)]


		for i, info, shape in zip( range(self.size), self.shape_info, self.shape_array ):
			if bairro_name_key!=None:
				bairro_name = info[bairro_name_key]
			else:
				bairro_name = None

			x, y = zip(*shape)
			lon, lat = basemap(x, y, inverse=True)
			
			bairro = {  'index': i,
						'bairro_shape_x': x,
						'bairro_shape_y': y,
						'bairro_shape_lon': lon,
						'bairro_shape_lat': lat,
						'bairro_polygon': Polygon(shape),
						'bairro_code': info[code_bairro_key],
						'bairro_name': bairro_name,
						'attributes' : None,
						'total_attributes' : None,
						'label': None,
						'train_or_test': None,
						}

			self.grid_bairros[i] = bairro


	def check_bairro(self, cod_bairro, point):
		for i, info, shape in zip( range(self.size), self.shape_info, self.shape_array ):
			if info[self.code_bairro_key] == cod_bairro:
				#x, y = basemap(lon, lat)
				pt = Pt(point.x, point.y)
				poly = Polygon(shape)
				iswithin = pt.within(poly)
		
				return iswithin

	def set_point_bairro(self, point):
		for i, info, shape in zip( range(self.size), self.shape_info, self.shape_array ):
			iswithin = self.check_bairro(info[self.code_bairro_key], point)
			if iswithin == True:
				point.in_territory = True
				point.cod_bairro = info[self.code_bairro_key]
				point.total_attributes = self.grid_bairros[i]['total_attributes']
				point.label = self.grid_bairros[i]['label']
				point.train_or_test = self.grid_bairros[i]['train_or_test']


	def set_pointmap_bairro( self, pmap ):
		for i, point in zip ( range(pmap.size), pmap.pointmap ):
			self.set_point_bairro( point )

	def set_pointlist_bairro( self, pointlist ):
		for i, point in zip ( range(pointlist), pointlist ):
			pass

			
			












