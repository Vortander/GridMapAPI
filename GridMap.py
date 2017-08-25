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
                        'total_crimes': 0,
                        'date_time': list(),
                        'in_territory': False,
                        'crimes_per_window': {},
                        'attributes_per_window': {},
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
#TODO PAREI AWUI
    def get_crime_by_cell(self, lowercell, uppercell, inTerritory=False):
        crime_by_cell = list()        
        for l in range(lowercell[0], uppercell[0]+1):
            for c in range(lowercell[1], uppercell[1]+1):
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
                    for shape in shapearray:
                        pt = Point(x, y)
                        poly = Polygon(shape)
                        if pt.within(poly) == True:
                            in_borders.append((l, c))
                            self.grid[l][c]['in_territory'] = True
                        #else:
                        #    self.grid[l][c]['in_territory'] = False

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


    def set_labels_per_point(self, labels_per_cell, filename='LabelsPerPoint.list', borderline=True):
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


    def set_labels_with_threshold(self, crimes_cell_list, label_list, min_crimes=2):
        #TODO: works only for binary classification
        for cells in crimes_cell_list:
            num_crimes = cells[1]

            if num_crimes > min_crimes:
                self.grid[cells[0][0]][cells[0][1]]['label'] = label_list[1]
            else:
                self.grid[cells[0][0]][cells[0][1]]['label'] = label_list[0]

            print(cells[0], cells[1], self.grid[cells[0][0]][cells[0][1]]['label'])

    def count_labels(self, lowercell, uppercell, label_list):
        #TODO: works only for binary classification
        label_count = {}
        label_count[label_list[0]] = 0
        label_count[label_list[1]] = 0
        for l in range(lowercell[0], uppercell[0]+1):
            for c in range(lowercell[1], uppercell[1]+1):
                if self.grid[l][c]['label'] == label_list[0]:
                    label_count[label_list[0]] += 1
                elif self.grid[l][c]['label'] == label_list[1]:
                    label_count[label_list[1]] += 1

        return label_count


    def set_labels_per_range(self, crimes_cell_list, label_list):

        crimes_list = crimes_cell_list[:]
        crimes_with_label = list()
        crimes_list = sorted(crimes_list, key = lambda x: int(x[1]))

        n = len(label_list)
        block_label = 0
        for i in range(n, 0, -1):
            group = round(len(crimes_list)/i)
            c = crimes_list[:int(group)]
            crimes_list = crimes_list[int(group):]
            for j in c:
                j.append(label_list[block_label])
                crimes_with_label.append(j)
            block_label+=1

        for cells in crimes_with_label:
            cell = cells[0]
            self.grid[cell[0]][cell[1]]['label'] = cells[2]
    
        return crimes_with_label

    def set_train_test_distribution(self, crimes_cell_list, slices=4, train_size=0.7):
        range_distribution = {'train': list(), 'test': list()} 
        crimes_list = crimes_cell_list[:]
        crimes_with_label = list()
       
        for i in range(slices, 0, -1):
            group = round(len(crimes_list)/i)
            c = crimes_list[:int(group)]
            crimes_list = crimes_list[int(group):]
            #split c block in train and test
            train = round(len(c) * train_size)
            test = len(c) - train
            trainset = c[:int(train)]
            testset = c[int(train):len(c)]
            
            for item in trainset:
                cell = item[0]
                self.grid[cell[0]][cell[1]]['train_or_test'] = 'train'
                range_distribution['train'].append((cell, self.grid[cell[0]][cell[1]]['total_crimes'], self.grid[cell[0]][cell[1]]['train_or_test']))
     
            for item in testset:
                cell = item[0]
                self.grid[cell[0]][cell[1]]['train_or_test'] = 'test'
                range_distribution['test'].append((cell, self.grid[cell[0]][cell[1]]['total_crimes'], self.grid[cell[0]][cell[1]]['train_or_test']))

        return range_distribution

    def set_grid_labels(self, labelslist):
        self.labels = labelslist

    def gen_pointlist_from_dir(self, path, tot='all', filename="pointListFromDir.list", metacheck=True):
        print(metacheck)
        if metacheck == False:
            imagepath = path
            metapath = ""
        else:
            imagepath = path + "/images"
            metapath = path + "/meta"

        print(imagepath)
        # try:
        #     file_list = sorted(os.listdir(imagepath))
        # except:
        #     file_list = sorted(os.listdir(path))
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

    def set_labels_per_cell(self, lowercell, uppercell, label, train_or_test):
        for l in range(lowercell[0], uppercell[0]+1):
            for c in range(lowercell[1], uppercell[1]+1):
                if label != "":
                    self.grid[l][c]['label'] = label
                self.grid[l][c]['train_or_test'] = train_or_test
                    

    
    # def gen_point_labels(self, pointListFromDir, filename="pointsLabel.list"):
    #     fw = open(filename, 'w')
    #     for point in pointListFromDir:
    #         cell = self.find_cell(point[0], point[1])
    #         if not cell:
    #             continue
    #         else:
    #             fw.write(str(point[0]) + ',' + str(point[1]) + ',' + str(cell['label']) + ',' + str(cell['train_or_test']) + '\n')
    #     fw.close()

    def _near_line(self, x2, y2, x1, y1, x3, y3, meters):
    cpoint = _get_perp(float(x1), float(y1), float(x2), float(y2), float(x3), float(y3))
    if cpoint != None:
        distance = _haversine(cpoint[0], cpoint[1], float(x3), float(y3))
        if distance <= meters:
            return True
    
    return False

    def copy_images_to_dir(self, sourcepath, destinypath, filenameListFromDir, borderline=False):
        
        online = False
        fr = open(filenameListFromDir, 'r')
        pointListFromDir = fr.readlines()
        fw = open(filenameListFromDir + '_copy.log', 'w')
        for point in pointListFromDir:
            point = point.replace("\n","")
            lat, lon = point.split(',')
            cell = self.find_cell(float(lat), float(lon))
            if not cell:
                continue
            else:
                train_or_test = self.grid[cell[0]][cell[1]]['train_or_test']
                label = self.grid[cell[0]][cell[1]]['label']
                
                # Test if point is above borderlines 
                if borderline == True:
                    border1_online = self._near_line( self.grid[cell[0]][cell[1]]['leftlon'], self.grid[cell[0]][cell[1]]['upperlat'], self.grid[cell[0]][cell[1]]['leftlon'], self.grid[cell[0]][cell[1]]['lowerlat'], lon, lat, 0.2)
                    border2_online = self._near_line( self.grid[cell[0]][cell[1]]['rightlon'], self.grid[cell[0]][cell[1]]['upperlat'], self.grid[cell[0]][cell[1]]['rightlon'], self.grid[cell[0]][cell[1]]['lowerlat'], lon, lat, 0.2)
                    border3_online = self._near_line( self.grid[cell[0]][cell[1]]['rightlon'],  self.grid[cell[0]][cell[1]]['lowerlat'], self.grid[cell[0]][cell[1]]['leftlon'], self.grid[cell[0]][cell[1]]['lowerlat'], lat, lon, 0.2)
                    border4_online = self._near_line( self.grid[cell[0]][cell[1]]['rightlon'], self.grid[cell[0]][cell[1]]['upperlat'], self.grid[cell[0]][cell[1]]['leftlon'], self.grid[cell[0]][cell[1]]['upperlat'], lon, lat, 0.2)
                
                if train_or_test != None or label != None:
                   if (border1_online or border2_online or border3_online or border4_online) == False:
                       for c in ['0', '90', '180', '270']:
                            try:
                                copy2(sourcepath + '/' + str(lat) + "_" + str(lon) + "_" + c + '.jpg', destinypath + '/' + str(train_or_test) + '/' + str(label) + '/')
                            except:
                                fw.write("Could not copy file, " + sourcepath + '/' + str(lat) + "_" + str(lon) + "_" + c + '.jpg' + " " + destinypath + '/' + str(train_or_test) + '/' + str(label) + '\n')
        fw.close()
        fr.close()


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

#TODO: Mudar esse nome para 'gen_point_list'
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

    def get_attr_distribuition(self, order='dsc'):
        distribution = list()
        for l in range(0, self.step):
            for c in range(0, self.step):
                if self.grid[l][c]['in_territory'] == True:
                    distribution.append([(l, c), self.grid[l][c]['total_crimes']])

        if order=='asc':
            reverse = False
        elif order=='dsc':
            reverse = True

        return sorted(distribution, key=lambda x: x[1], reverse=reverse)

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
        for dlat, dlon in zip(data.lat.values, data.lon.values):
            c = self.find_cell(dlat, dlon)
            if not c:
                total_crimes.append(0)
            else:
                total_crimes.append(self.grid[c[0]][c[1]]['total_crimes'])
        
        for dlat, dlon in zip(data.lat.values, data.lon.values):
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
                    cs = basemap.contourf(xi, yi, zi, vmin=1, vmax=self.get_max_crimes()['total_crimes'], cmap = plt.cm.jet, alpha=0.5)
                    

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







       






