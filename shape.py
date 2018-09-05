# coding: utf-8
import numpy as np
import math
from mpl_toolkits.basemap import Basemap
import ast

from rtree import index as Indx
from shapely.geometry import shape as GeoShape
from shapely.geometry import Polygon
from shapely.geometry import Point as Pt

# Street and shapefile manipulation methods

def _haversine(lat1, lon1, lat2, lon2):
	R = 6378.137
	dLat = lat2 * math.pi / 180 - lat1 * math.pi / 180
	dLon = lon2 * math.pi / 180 - lon1 * math.pi / 180
	a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(lat1 * math.pi / 180) * math.cos(lat2 * math.pi / 180) * math.sin(dLon/2) * math.sin(dLon/2)
	c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
	d = R * c
	d * 1000

	return d

# Convert shapearray to lon lat coordinates
def convert_shape(basemap, shapearray):
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
def generate_inter_points(finalshape):
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
def gen_pointlist_from_shapefile(finalshape, interpolated, filename="pointListFromShapefile.list"):
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
			corner_list.append(point)
			fw.write(str(point) + '\n')

	for line in interpolated[0].keys():
		for sect in interpolated[0][line]:
			for point in sect:
				interpol_list.append(point)
				fw.write(str(point) + '\n')

	fw.close()
	return corner_list, interpol_list

def plot_street_points(basemap, corner_list, interpol_list, cmark='g', imark='b'):
		#corner_list, interpol_list = lat, lon

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

def plot_street_points_fromfile(basemap, filename, color='m'):

	fr = open(filename, 'r')
	points = fr.readlines()

	plats = []
	plons = []

	for point in points:
		p = ast.literal_eval(point)
		lon, lat = basemap(p[1], p[0])
		plons.append(lon)
		plats.append(lat)

	basemap.scatter(plons, plats, marker='o',color=color)

# Generate rtree from shapefile
def generate_rtree( shapeinfo, shapearray, sector_key_name=None ):
	idx = Indx.Index()
	for pos, poly in zip( shapeinfo, shapearray ):
		idx.insert(int(pos[sector_key_name]), GeoShape(Polygon(poly)).bounds)

	return idx





