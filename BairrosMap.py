# coding: utf-8

import numpy as np
from shapely.geometry import Polygon
from shapely.geometry import Point

import os, sys
from shutil import copy2
from random import randint
from scipy.misc import imread

#Bairros = Neighborhood


class StreetPoint:
	
	def __init__(self, basemap, lon, lat):
		self.lon = lon
		self.lat = lat

		self.x, self.y = basemap(lon, lat)
		self.bairro = None


class BairrosMap:
	
	def __init__(self, basemap, shape_info, shapearray, code_bairro_key, bairro_name_key=None):
		self.basemap = basemap
		self.shape_info = shape_info
		self.shape_array = shapearray
		self.cod_bairro_key = code_bairro_key
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
		for info, shape in zip(self.shape_info, self.shape_array):
			if info[self.code_bairro_key] == cod_bairro:
				#x, y = basemap(lon, lat)
				pt = Point(point.x, point.y)
				poly = Polygon(shape)
				iswithin = pt.within(poly)
				if iswithin == True:
					point.bairro = cod_bairro

				return iswithin












