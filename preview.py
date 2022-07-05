import collections
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from tabulate import tabulate

def compute_fill_rate(depth_frame):
  total_size = depth_frame.size
  valid_pixels = np.count_nonzero(depth_frame)

  fill_rate = (valid_pixels / total_size) * 100
  return fill_rate

def compute_noise(depth_frame):
  total_size = depth_frame.size
  rows, columns = depth_frame.shape

  noise_pixels = 0
  pixel_neighbourhood = 2

  for row in range(pixel_neighbourhood,rows-pixel_neighbourhood):
    for column in range(pixel_neighbourhood,columns-pixel_neighbourhood):
      start_r, finish_r = row-pixel_neighbourhood, row+pixel_neighbourhood
      start_c, finish_c = column-pixel_neighbourhood, column+pixel_neighbourhood
      
      local_matrix = depth_frame[start_r:finish_r,start_c:finish_c]
      
      local_mean = local_matrix.mean()

      pixel_value = depth_frame[row][column]
      if abs(local_mean - pixel_value) > 0:
        noise_pixels += 1
  
  noise_rate = (noise_pixels / total_size) * 100
  return noise_rate

def get_img(path):
  depth = np.loadtxt(f'{path}/input/frame.txt')
  centroid = np.loadtxt(f'{path}/input/centroid.txt')
  clipping_distance = 400 # centroid[2] + 10

  depth = np.flip(depth, 1)
  
  y_mass, x_mass = np.where((depth > 0) & (depth < clipping_distance))

  top, bottom = min(y_mass) - 10, max(y_mass) + 10
  left, right = min(x_mass) - 10, max(x_mass) + 10

  bg_removed = np.where((depth == 0) | (depth > clipping_distance), 0, 1)

  cropped = bg_removed[top:bottom, left:right]

  return cropped, centroid

root = './images/real_time/gold_standard'

setup_dirs = os.listdir(root)

fill_per_light_position = collections.defaultdict(list)
fill_per_distance = collections.defaultdict(list)


noise_per_light_position = collections.defaultdict(list)
noise_per_distance = collections.defaultdict(list)

avg_per_light = {}
avg_per_distance = {}

noise_avg_per_light = {}
noise_avg_per_distance = {}

# std_per_light = {}
# std_per_distance = {}

fill_rates = []

for idx, setup in enumerate(setup_dirs):
  last_dash_idx = setup.rindex('-')
  setup_name = setup[:last_dash_idx]
  distance = setup[last_dash_idx+1:]

  img, centroid = get_img(f'{root}/{setup}')
  fill_rate = compute_fill_rate(img)
  noise_rate = compute_noise(img)

  fill_rates.append(fill_rate)

  fill_per_light_position[setup_name].append(fill_rate)
  fill_per_distance[distance].append(fill_rate)

  noise_per_light_position[setup_name].append(noise_rate)
  noise_per_distance[distance].append(noise_rate)

  plt.figure(1)
  plt.subplot(3, 3, idx+1)
  plt.imshow(img, 'gray')
  plt.title(setup)
  plt.xticks([])
  plt.yticks([])

  if idx == 0:
    continue

  avg_per_light[setup_name] = np.average(fill_per_light_position[setup_name])
  avg_per_distance[distance] = np.average(fill_per_distance[distance])

  # std_per_light[setup_name] = np.std(fill_per_light_position[setup_name])
  # std_per_distance[distance] = np.std(fill_per_distance[distance])

  noise_avg_per_light[setup_name] = np.average(noise_per_light_position[setup_name])
  noise_avg_per_distance[distance] = np.average(noise_per_distance[distance])

print('fill_rates', fill_rates)
print('avg_per_light fill', avg_per_light)
print('avg_per_distance fill', avg_per_distance)
# print('std_per_light', std_per_light)
# print('std_per_distance', std_per_distance)
print('avg_per_light noise', noise_avg_per_light)
print('avg_per_distance noise', noise_avg_per_distance)

plt.tight_layout() 
plt.show()