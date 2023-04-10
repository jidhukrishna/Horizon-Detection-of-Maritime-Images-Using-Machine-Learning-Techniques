from __future__ import division
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import time
import math
current_dir = os.getcwd()
files = [x for x in os.listdir(current_dir) if x.endswith(’.jpeg’)]
def get_plane_indicator_coordinates2(img,angle,distance):
x1 = int(round(max(0,
Horizon Detection of Maritime Images Using Machine Learning Techniques
(img.shape[1]/2)-((img.shape[0]/2)/math.tan(abs(angle)*np.pi/180)))))
x2 = img.shape[1] - x1
if i > 0:
y1 = int(round(min(img.shape[0],
(img.shape[0]/2) + ((img.shape[1]/2) *
np.tan(angle*np.pi/180)))))
else:
y1 = int(round(max(0,
(img.shape[0]/2) + ((img.shape[1]/2) *
np.tan(angle*np.pi/180)))))
y2 = img.shape[0] - y1
return [(x1,y1),(x1,y2)]
def get_plane_indicator_coord(img,angle,dist_as_perc):
’’’
img: image opened in cv2
angle: angle of plane indicator in degrees
dist_as_perc: pitch, as percent of 100 in which 0.50 is the
origin
’’’
heading = 90 + angle
heading_from_hor =
min(math.radians(180-heading),math.radians(heading))
radius = int(math.ceil(math.sqrt((img.shape[1]/2)**2 +
(img.shape[0]/2)**2)))
x0 = img.shape[1]/2
y0 = img.shape[0]/2
Horizon Detection of Maritime Images Using Machine Learning Techniques
x1 = (img.shape[1]/2) - radius
x2 = (img.shape[1]/2) + radius
y1 = (img.shape[0]/2)
y2 = (img.shape[0]/2)
x1_n = x0 + math.cos(math.radians(angle)) * (x1 - x0) -
math.sin(math.radians(angle)) * (y1 - y0)
y1_n = y0 + math.sin(math.radians(angle)) * (x2 - x0) +
math.cos(math.radians(angle)) * (y2 - y0)
x2_n = x0 + math.cos(math.radians(angle)) * (x2 - x0) -
math.sin(math.radians(angle)) * (y2 - y0)
y2_n = y0 + math.sin(math.radians(angle)) * (x1 - x0) +
math.cos(math.radians(angle)) * (y1 - y0)
if heading_from_hor < math.atan(img.shape[0]/img.shape[1]):
avail_dist =
int(img.shape[1]/math.cos(math.radians(heading)))
else: #heading_from_hor >= np.arctan(img.shape[0]/img.shape[1])
avail_dist =
int(img.shape[0]/math.sin(math.radians(heading)))
origin = avail_dist / 2
heading_transform = (dist_as_perc * avail_dist) - origin
x_transform = heading_transform * math.cos(math.radians(heading))
y_transform = heading_transform * math.sin(math.radians(heading))
return [(int(x1_n+x_transform),
int(y1_n+y_transform)),
Horizon Detection of Maritime Images Using Machine Learning Techniques
(int(x2_n+x_transform),
int(y2_n+y_transform))]
def show_horizon(img_file):
img = cv2.imread(img_file,0)
img = cv2.resize(img,dsize=None,fx=0.25,fy=0.25)
img_ann = img
edges = cv2.Canny(img, 40, 60, apertureSize = 3)
lines = cv2.HoughLines(edges, 1, np.pi/180, 80)
line_info =
np.array([0,720,1280,720,10000,10000,10000],dtype=int)
for i in range(len(lines)):
rho,theta = lines[i][0]
a = np.cos(theta)
b = np.sin(theta)
angle_from_horizon = np.pi/2 - theta
if np.abs(angle_from_horizon) < np.radians(45):
c = np.tan(angle_from_horizon)
x0 = a*rho
y0 = b*rho
x1 = 0
y1 = int(y0 + (x0*c))
x2 = img.shape[1]
y2 = int(y1 - (x2*c))
global_sky_mat = np.empty(0,dtype=np.uint8)
global_sea_mat = np.empty(0,dtype=np.uint8)
Horizon Detection of Maritime Images Using Machine Learning Techniques
local_sky_mat = np.empty(0,dtype=np.uint8)
local_sea_mat = np.empty(0,dtype=np.uint8)
for x_coor in range(0,x2,2):
y_d = int(y1 - (x_coor * c))
global_sky_mat = np.append(global_sky_mat,
img[0:y_d,x_coor] )
#global_sea_mat = np.append(global_sea_mat,
img[y_d:img.shape[0], x_coor])
local_sky_mat = np.append(local_sky_mat,
img[(y_d-10):y_d, x_coor] )
local_sea_mat = np.append(local_sea_mat,
img[y_d:(y_d+10), x_coor] )
global_sky_var = int(np.var(global_sky_mat))
#global_sea_var = int(np.var(global_sea_mat)
local_sky_mean = int(np.mean(local_sky_mat))
local_sea_mean = int(np.mean(local_sea_mat))
stats = np.array([x1,y1,x2,y2,
global_sky_var,
local_sky_mean,
local_sea_mean])
#if global_sky_var <= global_sea_var:
line_info = np.vstack((line_info,stats))
start = time.time()
print(’finding best line...’)
Horizon Detection of Maritime Images Using Machine Learning Techniques
# minimize intraclass variance of sky and sea
# maximize interclass mean of sky and sea
#(line_info[:,4]+line_info[:,5])/
#best_line = line_info[np.argmin(
#
(line_info[:,4])**2/np.abs(line_info[:,6]-line_info[:,7]))]
#best_line = line_info[np.argmin(
# (line_info[:,4])**2 +
line_info[:,5])/np.abs(line_info[:,6]-line_info[:,7]))]
#best_line = line_info[np.argmin(line_info[:,4]+line_info[:,5])]
best_line = line_info[np.argmin(
(line_info[:,4])/np.abs(line_info[:,5]-line_info[:,6]))]
end = time.time()
cv2.line(img_ann,(best_line[0],best_line[1]),(best_line[2],best_line[3]),(0,0,255),2)
print(img_file)
print(best_line)
print(len(lines))
plt.imshow(img_ann)
plt.show()
for f in files:
show_horizon(f)
