import collections
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import pandas as pd
import math

def compute_fill_rate(depth_frame):
  total_size = depth_frame.size
  valid_pixels = np.count_nonzero(depth_frame)

  fill_rate = (valid_pixels / total_size) * 100
  return fill_rate

def mean_filter(depth_frame):
  rows, columns = depth_frame.shape
  new_frame = np.ndarray((rows, columns))
  pixel_neighbourhood = 5

  for row in range(pixel_neighbourhood,rows-pixel_neighbourhood):
    for column in range(pixel_neighbourhood,columns-pixel_neighbourhood):
      start_r, finish_r = row-pixel_neighbourhood, row+pixel_neighbourhood + 1
      start_c, finish_c = column-pixel_neighbourhood, column+pixel_neighbourhood + 1
      
      local_matrix = depth_frame[start_r:finish_r,start_c:finish_c]
      
      local_mean = local_matrix.mean()

      new_frame[row][column] = round(local_mean)
      # print('local_mean', depth_frame[row][column])

  return new_frame

# 1 1 1 1 1
# 1 1 1 1 1
# 1 1 1 1 1
# 1 1 1 1 1
# 1 1 1 1 1


def erosion_filter(depth_frame):
  rows, columns = depth_frame.shape
  new_frame = np.zeros((rows, columns))

  kernel_size = 5
  kernel = np.ones((kernel_size, kernel_size), dtype=np.int64)
  skip_pixels = math.floor(kernel_size / 2)

  for row in range(skip_pixels, rows - skip_pixels):
    for column in range(skip_pixels, columns - skip_pixels):
      
      start_r, finish_r = row - skip_pixels, row + skip_pixels + 1
      start_c, finish_c = column - skip_pixels, column + skip_pixels + 1

      # print('start', start_r, 'finish', finish_r)
      # print('start', start_c, 'finish', start_c)
      
      local_matrix = np.asarray(depth_frame[start_r:finish_r,start_c:finish_c])
      # print('kernel', kernel)
      # print('local', local_matrix)
      
      are_equal = (kernel == local_matrix).all()

      new_frame[row][column] = 1 if are_equal else 0
      # print('local_mean', depth_frame[row][column])

  return new_frame

def get_img(path):
  depth = np.loadtxt(f'{path}/input/frame.txt')
  centroid = np.loadtxt(f'{path}/input/centroid.txt')
  clipping_distance = 400 # centroid[2] + 10

  depth = np.flip(depth, 1)
  
  y_mass, x_mass = np.where((depth > 0) & (depth < clipping_distance))
  padding = 20

  top, bottom = min(y_mass) - padding, max(y_mass) + padding
  left, right = min(x_mass) - padding, max(x_mass) + padding

  bg_removed = np.where((depth <= 0) | (depth > clipping_distance), 0, 1)

  cropped = bg_removed[top:bottom, left:right]

  return cropped, centroid

def rms(original_frames, filtered_frames):
  return np.sqrt(((original_frames - filtered_frames) ** 2).mean())

def calculate_rms(original: dict, filtered: dict):
  rms_per_setup = collections.defaultdict(dict)

  light_setups = original.keys()
  # print('light_setups', light_setups)

  for light_setup in light_setups:
    original_setup = original[light_setup]
    filtered_setup = filtered[light_setup]
    
    distances = original_setup.keys()

    for distance in distances:
      original_frame = original_setup[distance]
      filtered_frame = filtered_setup[distance]

      rms_error = rms(original_frame, filtered_frame)

      rounded = float("{:.3f}".format(rms_error))

      rms_per_setup[light_setup][distance] = rounded
  

  df = pd.DataFrame(rms_per_setup)
  df.plot(kind='bar')

  plt.suptitle('RMSE por experimento')
  # plt.show()

  for key, value in rms_per_setup.items():
    print(key, ' : ', value)

root = './images/real_time/gold_standard'

setup_dirs = os.listdir(root)

original_imgs = collections.defaultdict(dict)
filtered_imgs = collections.defaultdict(dict)

for idx, setup in enumerate(setup_dirs):
  last_dash_idx = setup.rindex('-')
  setup_name = setup[:last_dash_idx]
  distance = setup[last_dash_idx+1:]

  img, centroid = get_img(f'{root}/{setup}')

  original_imgs[setup_name][distance] = img

  plt.figure(1)
  plt.suptitle('Original')
  plt.subplot(3, 3, idx+1)
  plt.imshow(img, 'gray')
  plt.title(setup)
  plt.xticks([])
  plt.yticks([])

  copy_img = img.copy()
  filtered_img = mean_filter(copy_img) # mean_filter(copy_img)

  filtered_imgs[setup_name][distance] = filtered_img

  plt.figure(2)
  plt.suptitle('Mean Filter')
  plt.subplot(3, 3, idx+1)
  plt.imshow(filtered_img, 'gray')
  plt.title(setup)
  plt.xticks([])
  plt.yticks([])


calculate_rms(original_imgs, filtered_imgs)

plt.figure(1)
plt.tight_layout() 

plt.figure(2)
plt.tight_layout() 

plt.show()