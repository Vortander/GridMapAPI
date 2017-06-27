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

def _haversine(lat1, lon1, lat2, lon2):
    R = 6378.137
    dLat = lat2 * math.pi / 180 - lat1 * math.pi / 180
    dLon = lon2 * math.pi / 180 - lon1 * math.pi / 180
    a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(lat1 * math.pi / 180) * math.cos(lat2 * math.pi / 180) * math.sin(dLon/2) * math.sin(dLon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c
    d * 1000

    return d

def _random_list(sample_size, range_size):
    rlist = list()
    while(sample_size):
        x = randint(1, range_size)
        if x not in rlist:
            rlist.append(x)
            sample_size-=1

    return rlist

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
        self.image_distribution = {}
        self.window = ""

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
                        'total_crimes': 0,
                        'date_time': list(),
                        'in_territory': False,
                        'crimes_per_window': {},
                        'attributes_per_window': {}}
                
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
    
    def add_crime(self, l, c, date, time):
        self.grid[l][c]['total_crimes'] += 1
        self.grid[l][c]['date_time'].append((date, time))
    
    def get_max_crimes(self):
        maxim = 0
        for l in range(0, self.step):
            for c in range(0, self.step):
                if self.grid[l][c]['total_crimes'] > maxim:
                    maxim = self.grid[l][c]['total_crimes']
                    all_max = self.grid[l][c]
        return all_max

    def get_min_crimes(self):
        maxim = self.get_max_crimes()['total_crimes']
        minim = maxim
        for l in range(0, self.step):
            for c in range(0, self.step):
                if (self.grid[l][c]['total_crimes'] < minim) and (self.grid[l][c]['total_crimes'] > 0):
                    minim = self.grid[l][c]['total_crimes']
                    all_min = self.grid[l][c]
        return all_min

    def get_crime_by_cell(self, inTerritory=False):
        crime_by_cell = list()        
        for l in range(0, self.step):
            for c in range(0, self.step):
                if inTerritory == True:
                    if self.grid[l][c]['in_territory'] == True:
                        crime_by_cell.append([(l, c), self.grid[l][c]['total_crimes']])
                else:
                    crime_by_cell.append([(l, c), self.grid[l][c]['total_crimes']])

        return crime_by_cell

    def set_borders(self, basemap, shapearray, force=False):
        in_borders = []
        if force == True:
            for l in range(0, self.step):
                for c in range(0, self.step):
                    self.grid[l][c]['in_territory'] = True
                    in_borders.append((l, c))

        else:
            for l in range(0, self.step):
                for c in range(0, self.step):
                    x, y = basemap(self.grid[l][c]['centroid'][0], self.grid[l][c]['centroid'][1])
                    pt = Point(x, y)
                    poly = Polygon(shapearray[0])
                    if pt.within(poly) == True:
                        in_borders.append((l, c))
                        self.grid[l][c]['in_territory'] = True
                    else:
                        self.grid[l][c]['in_territory'] = False

        return in_borders

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

    def distribute_images(self, basemap, shapearray, finalshape = list(), new_points = list()):
        cells = self.set_borders(basemap, shapearray)
        for c in cells:
            self.image_distribution[c] = list()
        
        # If finalshape is not empty
        if finalshape:
            for key in finalshape.copy().keys():
                for point in finalshape[key]:
                    c = self.find_cell(point[0], point[1])
                    if c and self.grid[c[0]][c[1]]['in_territory'] == True:
                        self.image_distribution[c].append(str(key) + "_" + str(point[0]) + "_" + str(point[1]))
                    else:
                        continue

        if new_points:
            for key in new_points[0].copy().keys():
                for segment in new_points[0][key]:
                    for point in segment:
                        c = self.find_cell(point[0], point[1])
                        if c and self.grid[c[0]][c[1]]['in_territory'] == True:
                            self.image_distribution[c].append(str(key) + "_" + str(point[0]) + "_" + str(point[1]))
                        else:
                            continue

    def add_attributes(self, l, c, attribute, window):
        if attribute in self.grid[l][c]['attributes_per_window']:
            if window in self.grid[l][c]['attributes_per_window'][attribute]:
                self.grid[l][c]['attributes_per_window'][attribute][window] += 1
            else:
                self.grid[l][c]['attributes_per_window'][attribute][window] = 1
        else:
            self.grid[l][c]['attributes_per_window'][attribute] = { window: 1 }


    def set_labels_per_point(self, labels_per_cell, filename='LabelsPerPoint.list'):
        fw = open(filename, 'w')
        fw.write('cell,label,point\n')
        for c in labels_per_cell:
            for point in self.image_distribution[c[0]]:
                fw.write(str(c[0]) + "," + str(c[2]) + "," + str(point) + "\n")
        fw.close()



    def add_crimes_per_window(self, l, c, window):
        if window in self.grid[l][c]['crimes_per_window']:
            self.grid[l][c]['crimes_per_window'][window] += 1
        else:
            self.grid[l][c]['crimes_per_window'][window] = 1
        

    def set_window(self, windowstruct):
        self.window = windowstruct

    def set_labels_per_range(self, crimes_cell_list, label_list):
        crimes_list = crimes_cell_list.copy()
        crimes_with_label = list()
        crimes_list = sorted(crimes_list, key = lambda x: int(x[1]))

        n = len(label_list)
        block_label = 0
        for i in range(n, 0, -1):
            group = round(len(crimes_list)/i)
            c = crimes_list[:group]
            crimes_list = crimes_list[group:]
            for j in c:
                j.append(label_list[block_label])
                print(j)
                crimes_with_label.append(j)
            block_label+=1

        return crimes_with_label

    def gen_imagelist_with_label(self, cell_path, crimes_with_label, filename="imageListLabels.list"):
        fw = open(filename, 'w')
        fw.write('line,column,file,label\n')
        for cell in crimes_with_label:
            l = cell[0][0]
            c = cell[0][1]
            label = cell[2]
            directory = cell_path + str(l) + "_" + str(c)
            for file in sorted(os.listdir(directory)):
                fw.write(str(l) + "," + str(c) + "," + file + "," + label + "\n")

    def gen_images_dict(self, cell_path, crimes_with_label):
        # create dictionary with all files
        dictimages = {}
        cell_labels = {}
        for cell in crimes_with_label:
            if not cell[2] in dictimages:
                dictimages[cell[2]] = {}
        for cell in crimes_with_label:
            if not cell[2] in cell_labels:
                cell_labels[cell[2]] = []
        for cell in crimes_with_label:
            cell_labels[cell[2]].append((cell[0][0], cell[0][1]))

        print(dictimages)
        print(cell_labels)
        
        for labcell in cell_labels.copy().keys():
            label = labcell
            counter = 1
            for cells in cell_labels[label]:
                l = cells[0]
                c = cells[1]
                directory = cell_path + str(l) + "_" + str(c)
                for file in sorted(os.listdir(directory)):
                    dictimages[label][counter] = [(l,c), file]
                    counter+=1

        return dictimages

    def read_point_list(filename):
        pointlist = []
        fw = open(filename, 'r')
        lines = fw.readlines()
        for line in lines:
            line = line.replace('(','').replace(')','').replace('\n','')
            lat, lon = line.split(',')
            pointlist.append((float(lat),float(lon)))

        return pointlist

# TODO: Add parameter fromJson = True -> and trainsform key in str: images_dict[label][str(i)][0][1]
    def gen_train_test(self, images_dict, source_path, destiny_path, maxfiles, trainpercent):
        lenght_per_labels = {}
        for keys in images_dict.copy().keys():
            if not keys in lenght_per_labels:
                lenght_per_labels[keys] = len(images_dict[keys])
        
        print("Number of files per label: ", lenght_per_labels)

        # create directories
        labels = list(images_dict.keys())
        try:
            os.mkdir(destiny_path + "train")
            os.mkdir(destiny_path + "test")
        except:
            print("Could not create train and test directories.")
            sys.exit()

        try:
            for label in labels:
                os.mkdir(destiny_path + "train/" + label)
                os.mkdir(destiny_path + "test/" + label)
        except:
            print("Could not create label subdirectories.")
            sys.exit()

        for label in labels:
            rand_numbers = _random_list(maxfiles, len(images_dict[label]))
            
            #trainpercent
            train_size = round(maxfiles * trainpercent)
            test_size = maxfiles - train_size
            print(label, train_size, test_size)
            
            train_set = rand_numbers[:train_size]
            test_set = rand_numbers[train_size:]
            # train
            for i in train_set:
                try:
                    copy2(source_path + str(images_dict[label][i][0][0]) + "_" + str(images_dict[label][i][0][1]) + "/" + images_dict[label][i][1], destiny_path + "train/" + label + "/")
                except:
                    print("Could not copy file ", source_path + str(images_dict[label][i][0][0]) + "_" + str(images_dict[label][i][0][1]) + "/" + images_dict[label][i][1], destiny_path + "train/" + label + "/")

            for j in test_set:
                try:
                    copy2(source_path + str(images_dict[label][j][0][0]) + "_" + str(images_dict[label][j][0][1]) + "/" + images_dict[label][j][1], destiny_path + "test/" + label + "/")
                except:
                    print("Could not copy file ", source_path + str(images_dict[label][j][0][0]) + "_" + str(images_dict[label][j][0][1]) + "/" + images_dict[label][j][1], destiny_path + "test/" + label + "/")


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



#### Métodos para Plotagem ####                    
    def plot_grid(self, basemap, marksize, linewidth):
        for l in range(0, self.step):
            for c in range(0, self.step):
                left_lon, right_lon = basemap([self.grid[l][c]['leftlon'], self.grid[l][c]['rightlon']], [self.grid[l][c]['lowerlat'], self.grid[l][c]['lowerlat']])
                low_lat, up_lat = basemap([self.grid[l][c]['rightlon'], self.grid[l][c]['rightlon']], [self.grid[l][c]['lowerlat'], self.grid[l][c]['upperlat']])
                centroid = basemap(self.grid[l][c]['centroid'][0], self.grid[l][c]['centroid'][1])
                basemap.plot([left_lon[0], left_lon[1]], [right_lon[0], right_lon[1]], 'bo-', markersize=marksize, linewidth=0.6)
                basemap.plot([low_lat[0], low_lat[1]], [up_lat[0], up_lat[1]], 'bo-', markersize=marksize, linewidth=0.6)
                basemap.plot(centroid[0], centroid[1], 'bo-', markersize=marksize, linewidth=linewidth)
                
    def plot_crime(self, basemap, lon, lat, mark, size, linewidth):
        self.plot_grid(basemap, 1, linewidth)
        lon_, lat_ = basemap(lon, lat)
        basemap.plot(lon_, lat_, mark, markersize=size)

        
    def distribute_crimes(self, basemap, data):
        self.plot_grid(basemap, 0, 1)

        total_crimes = []
        ylat = []
        xlon = []
        for dlat, dlon in zip(data.Latitude.values, data.Longitude.values):
            c = self.find_cell(dlat, dlon)
            if not c:
                total_crimes.append(0)
            else:
                total_crimes.append(self.grid[c[0]][c[1]]['total_crimes'])
        
        for dlat, dlon in zip(data.Latitude.values, data.Longitude.values):
            x, y = basemap(dlon, dlat)
            xlon.append(x)
            ylat.append(y)
        
        numcols, numrows = self.step, self.step
        
        xi = np.linspace(basemap.llcrnrx, basemap.urcrnrx, numcols)
        yi = np.linspace(basemap.llcrnry, basemap.urcrnry, numrows)
        xi, yi = np.meshgrid(xi, yi)
        
        x, y, z = xlon, ylat, total_crimes
        zi = griddata(x, y, z, xi, yi, interp='linear')
        m = basemap.contourf(xi, yi, zi, vmin=-5, vmax=self.get_max_crimes()['total_crimes'], alpha=0.5)
        cb = basemap.colorbar(m, location='bottom', pad="5%")
        
        
    def hot_spot(self, basemap, linewidth):
        self.plot_grid(basemap, 0, linewidth)

        for l in range(0, self.step):
            for c in range(0, self.step):
                if self.grid[l][c]['total_crimes'] > 0:
                    left_lon1, right_lon1 = basemap([self.grid[l][c]['leftlon'], self.grid[l][c]['rightlon']], [self.grid[l][c]['lowerlat'], self.grid[l][c]['lowerlat']])
                    left_lon2, right_lon2 = basemap([self.grid[l][c]['leftlon'], self.grid[l][c]['rightlon']], [self.grid[l][c]['upperlat'], self.grid[l][c]['upperlat']])
                    low_lat1, up_lat1 = basemap([self.grid[l][c]['rightlon'], self.grid[l][c]['rightlon']], [self.grid[l][c]['lowerlat'], self.grid[l][c]['upperlat']])
                    low_lat2, up_lat2 = basemap([self.grid[l][c]['leftlon'], self.grid[l][c]['leftlon']], [self.grid[l][c]['upperlat'], self.grid[l][c]['lowerlat']])
                    
                    data = pd.DataFrame([(left_lon1[0], right_lon1[0], self.grid[l][c]['total_crimes']),
                                        (low_lat1[1], up_lat1[1], self.grid[l][c]['total_crimes']),
                                        (low_lat2[0], up_lat2[0], self.grid[l][c]['total_crimes']),
                                        (low_lat1[0], up_lat1[0], self.grid[l][c]['total_crimes'])], columns=list('XYZ'))
                    numcols, numrows = self.step + 2, self.step + 2
                    xi = np.linspace(data.X.min(), data.X.max(), numcols)
                    yi = np.linspace(data.Y.min(), data.Y.max(), numrows)
                    xi, yi = np.meshgrid(xi, yi)
                    
                    x, y, z = data.X.values, data.Y.values, data.Z.values
                    zi = griddata(x, y, z, xi, yi, interp='linear')
                    cs = basemap.contourf(xi, yi, zi, vmin=1, vmax=self.get_max_crimes()['total_crimes'], alpha=0.5)

    def event_points(self, basemap, corner_list, interpol_list, cmark='g', imark='b'):
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

       






