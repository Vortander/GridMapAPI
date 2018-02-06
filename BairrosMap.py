# coding: utf-8

import numpy as np
from shapely.geometry import Polygon
from shapely.geometry import Point

import os, sys
from shutil import copy2
from random import randint
from scipy.misc import imread

#Bairros = Neighborhood


class Point:
	
	def __init__(self, basemap, lon, lat):
		self.lon = lon
		self.lat = lat

		self.x, self.y = basemap(lon, lat)
		self.bairro = None


class BairrosMap:
	
	def __init__(self, basemap, shape_info, shapearray, code_bairro_key):
		self.basemap = basemap
		self.shape_info = shape_info
		self.shape_array = shapearray
		self.cod_bairro_key = code_bairro_key

		for info, shape in zip(self.shape_info, self.shapearray):
			
			self.bairro = { 'bairro_shape': None,
						'bairro_polygon': None,
						'attributes' : None
						'attributes_time' : {}
						'total_attributes' : None
						}




		




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














