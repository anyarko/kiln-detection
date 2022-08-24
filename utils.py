import os
import pickle
import random
import sys
import time
from copy import copy
from math import ceil, floor
import itertools

import geopandas as gpd
from matplotlib import patches as patches
from matplotlib import pyplot as plt
import numpy as np
import pandas
import pandas as pd
import rasterio
import requests
import rioxarray
# import seaborn as sns
from requests.auth import HTTPBasicAuth
from skimage.measure import label, regionprops
from sklearn.metrics import pairwise_distances
from tensorflow.image import (adjust_brightness, adjust_contrast, adjust_gamma,
                              adjust_hue, adjust_jpeg_quality,
                              adjust_saturation)


def load_shapefile(path):
    data = gpd.read_file(path)
    return data.to_crs(epsg=4326)

def building_link(index, zoom, shapefile):
    os.environ['PL_API_KEY']='ksdlks'
    PLANET_API_KEY = os.getenv('PL_API_KEY')
    auth = HTTPBasicAuth(PLANET_API_KEY, '')

    x = float(*shapefile[shapefile.index == index].geometry.representative_point().x)
    y = float(*shapefile[shapefile.index == index].geometry.representative_point().y)
    polygon = str(*shapefile[shapefile.index == index].geometry).replace(' ','')

    # old_mosaic = 'global_monthly_2022_02_mosaic'
    link = f'https://www.planet.com/basemaps/#/mosaic/global_quarterly_2020q1_mosaic/center/{x},{y}/zoom/{zoom}/geometry/{polygon}'

    res = requests.get(url=link, auth=auth)
    
    return link

def load_tiff_paths(files_dir):
    path = os.path.join(files_dir)
    files = os.listdir(path)
    tiff_files = [ files_dir + '//' + file for file in files if file.endswith('.tif') or file.endswith('.TIF') ]
    return sorted(tiff_files)

def load_tiff(tiff_path, plot=False):
    source = rasterio.open(tiff_path, 'r')
    rgb = source.read([1,2,3])
    source.close()
    rgb = np.divide(np.transpose(rgb, (1,2,0)), 255)
    if plot:
        f, ax = plt.subplots(1, figsize=(12, 12))
        ax.imshow(rgb, interpolation='nearest')
        ax.axis('off')
    return rgb

def areas_covered(tiff_path, shapefile):
    tiff_array = rioxarray.open_rasterio(tiff_path, variable=["green"], mask_and_scale=True)
    crs = rasterio.crs.CRS.from_epsg(4326)
    long_min, lat_min, long_max, lat_max = tiff_array.rio.transform_bounds(crs)

    file_name = tiff_path[23:38]
    if f'kilns_{file_name}.pkl' in os.listdir('cached'):
        with open(f'cached/kilns_{file_name}.pkl', 'rb') as file:
            covered_regions = pickle.load(file)
            return (long_min, lat_min, long_max, lat_max), tiff_array, covered_regions

    covered_regions = shapefile[shapefile.geometry.bounds.minx > long_min]
    covered_regions = covered_regions[covered_regions.geometry.bounds.maxx < long_max]
    covered_regions = covered_regions[covered_regions.geometry.bounds.miny > lat_min]
    covered_regions = covered_regions[covered_regions.geometry.bounds.maxy < lat_max]
    
    with open(f'cached/kilns_{file_name}.pkl', 'wb') as file:
        pickle.dump(covered_regions, file)

    return (long_min, lat_min, long_max, lat_max), tiff_array, covered_regions

def get_selection(tiff_path, shapefile, kiln_num):
    tiff = load_tiff(tiff_path)
    coords, tiff_array, covered_regions = areas_covered(tiff_path, shapefile)
    long_min, lat_min, long_max, lat_max = coords
    kiln_long_min, kiln_lat_min, kiln_long_max, kiln_lat_max = covered_regions.geometry.iloc[kiln_num].bounds

    x = covered_regions.geometry.iloc[kiln_num].representative_point().x
    y = covered_regions.geometry.iloc[kiln_num].representative_point().y

    left = floor(((kiln_long_min-long_min)/(long_max-long_min))*tiff_array.shape[-1])
    right = ceil(((kiln_long_max-long_min)/(long_max-long_min))*tiff_array.shape[-1])
    top = ceil(((lat_max-kiln_lat_max)/(lat_max-lat_min))*tiff_array.shape[-2])
    bottom =  floor(((lat_max-kiln_lat_min)/(lat_max-lat_min))*tiff_array.shape[-2])

    trans_x = floor(((x-long_min)/(long_max-long_min))*tiff_array.shape[-1])
    trans_y = floor(((lat_max-y)/(lat_max-lat_min))*tiff_array.shape[-2])
    scaled_x = left-25-trans_x
    scaled_y = top-25-trans_y

    f, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(tiff[top-25:bottom+25,left-25:right+25])  
    
    ax.scatter(-scaled_y, -scaled_x, color='r')
    ax.set_title(covered_regions.kiln_type.iloc[kiln_num])
    
    rect = patches.Rectangle((25,25), right-left, bottom-top, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

def view_kilns(tiff_path, shapefile):
    tiff = load_tiff(tiff_path)
    coords, tiff_array, covered_regions = areas_covered(tiff_path, shapefile)
    long_min, lat_min, long_max, lat_max = coords

    kilns_in_tiff = covered_regions.shape[0]
    num_cols = 4
    num_rows = 5

    # f, axs = plt.subplots(ceil(kilns_in_tiff/num_per_row), num_per_row, figsize=(60, 60))
    f, axs = plt.subplots(num_rows, num_cols, figsize=(60, 60))
    for kiln_num in range(kilns_in_tiff):
        if kiln_num == num_cols*num_cols:
            break

        row = floor(kiln_num / num_cols)
        col = kiln_num % num_cols

        kiln_long_min, kiln_lat_min, kiln_long_max, kiln_lat_max = covered_regions.geometry.iloc[kiln_num].bounds
        x = covered_regions.geometry.iloc[kiln_num].representative_point().x
        y = covered_regions.geometry.iloc[kiln_num].representative_point().y


        left = floor(((kiln_long_min-long_min)/(long_max-long_min))*tiff_array.shape[-1])
        right = ceil(((kiln_long_max-long_min)/(long_max-long_min))*tiff_array.shape[-1])
        top = ceil(((lat_max-kiln_lat_max)/(lat_max-lat_min))*tiff_array.shape[-2])
        bottom =  floor(((lat_max-kiln_lat_min)/(lat_max-lat_min))*tiff_array.shape[-2])


        trans_x = floor(((x-long_min)/(long_max-long_min))*tiff_array.shape[-1])
        trans_y = floor(((lat_max-y)/(lat_max-lat_min))*tiff_array.shape[-2])
        scaled_x = left-25-trans_x
        scaled_y = top-25-trans_y

        # axs[row, col].imshow(tiff[top-25:bottom+25,left-25:right+25])  
        try:
            axs[row, col].imshow(tiff[top-25:bottom+25,left-25:right+25])  
        except:
            pass

        axs[row, col].scatter(-scaled_y, -scaled_x, color='r')

        rect = patches.Rectangle((25,25), right-left, bottom-top, linewidth=1, edgecolor='r', facecolor='none')
        axs[row, col].add_patch(rect)
        axs[row, col].set_title(covered_regions.kiln_type.iloc[kiln_num])
        axs[row, col].axis('off')

    plt.subplots_adjust( wspace=0, hspace=0.1)

def create_kiln_mask(tiff_path, shapefile, plot=False, trio=False):
    coords, tiff_array, covered_regions = areas_covered(tiff_path, shapefile)
    long_min, lat_min, long_max, lat_max = coords
    kiln_type_axs_mapping = {'bull_trench':0, 'zigzag':1, 'false_positive':2}

    if trio:
        kiln_mask = np.zeros(shape=(4096,4096,3))
        kiln_mask[:,:,2] = 1
    else:
        covered_regions = covered_regions[covered_regions.kiln_type != 'false_positive']
        kiln_mask = np.zeros(shape=(4096,4096))
    kiln_locations = []
    
    if plot:
        if trio:
            f, axs = plt.subplots(1, 5, figsize=(30, 30))
        else:
            f, axs = plt.subplots(1, 3, figsize=(20, 15))

    for idx, kiln in covered_regions.iterrows():    
        if plot:
            x = kiln.geometry.representative_point().x
            y = kiln.geometry.representative_point().y
            trans_x = floor(((x-long_min)/(long_max-long_min))*tiff_array.shape[-1])
            trans_y = floor(((lat_max-y)/(lat_max-lat_min))*tiff_array.shape[-2])
            if not trio:
                axs[0].scatter(trans_x, trans_y, color='r')
            else:
                if kiln.kiln_type == 'bull_trench':
                    axs[0].scatter(trans_x, trans_y, color='r')
                elif kiln.kiln_type == 'zigzag':
                    axs[0].scatter(trans_x, trans_y, color='g')
                else:
                    axs[0].scatter(trans_x, trans_y, color='b')    

        kiln_long_min, kiln_lat_min, kiln_long_max, kiln_lat_max = kiln.geometry.bounds
        left = floor(((kiln_long_min-long_min)/(long_max-long_min))*tiff_array.shape[-1])
        right = ceil(((kiln_long_max-long_min)/(long_max-long_min))*tiff_array.shape[-1])
        top = ceil(((lat_max-kiln_lat_max)/(lat_max-lat_min))*tiff_array.shape[-2])
        bottom =  floor(((lat_max-kiln_lat_min)/(lat_max-lat_min))*tiff_array.shape[-2])
        kiln_locations.append((left, right, top, bottom))

        if plot:
            if not trio:
                rect = patches.Rectangle(((left+right)/2, (top+bottom)/2), right-left, bottom-top, linewidth=1, edgecolor='r', facecolor='none')
                axs[1].add_patch(rect)
            else:
                if kiln.kiln_type == 'bull_trench':
                    rect = patches.Rectangle(((left+right)/2, (top+bottom)/2), right-left, bottom-top, linewidth=1, edgecolor='r', facecolor='none')
                    axs[1].add_patch(rect)
                elif kiln.kiln_type == 'zigzag':
                    rect = patches.Rectangle(((left+right)/2, (top+bottom)/2), right-left, bottom-top, linewidth=1, edgecolor='g', facecolor='g')
                    axs[1].add_patch(rect)
                else:
                    rect = patches.Rectangle(((left+right)/2, (top+bottom)/2), right-left, bottom-top, linewidth=1, edgecolor='b', facecolor='none')
                    axs[1].add_patch(rect)

        for row_i in range(floor(top), ceil(bottom)):
            for col_i in range(floor(left), ceil(right)):
                # kiln_mask[row_i, col_i] = 1 
                lat = abs((top+bottom)/2 - row_i)
                long = abs((left+right)/2 - col_i)
                if trio:
                    kiln_mask[row_i, col_i, kiln_type_axs_mapping[kiln.kiln_type]] = np.exp(-(lat**2 + long**2)/(2*36))
                    kiln_mask[row_i, col_i, 2] = 1 - np.exp(-(lat**2 + long**2)/(2*36))
                else:
                    kiln_mask[row_i, col_i] = np.exp(-(lat**2 + long**2)/(2*36))
    
    if not plot:
        return kiln_mask, np.array(kiln_locations), covered_regions.kiln_type

    tiff = load_tiff(tiff_path, plot=False)
    if trio:
        titles = ['kilns marked', 'kilns bounded', 'bull trenches', 'zigzags', 'false postives']
    else:
        titles = ['kilns marked', 'kilns bounded', 'kilns']

    for idx, ax in enumerate(axs):
        ax.set_title(titles[idx])
        ax.axis('off')
        if idx < 2:
            ax.imshow(tiff, interpolation='nearest')
        else:
            if trio:
                axs[idx].imshow(kiln_mask[:,:,idx-2:idx-1])
            else:
                axs[idx].imshow(kiln_mask)

    plt.subplots_adjust( wspace=0.01, hspace=0.1)

def view_augmentation(image, func, interval, num_values=5):
    augmentation_values = np.linspace(interval[0], interval[1], num=num_values)
    f, axs = plt.subplots(1, num_values, figsize=(20,20))
    
    for plot_num, value in enumerate(augmentation_values):
        new_img = np.clip(func(image, value), 0, 1)
        axs[plot_num].imshow(new_img)
        axs[plot_num].set_title(func.__name__ + ' for ' + str(round(value, 2)))
        axs[plot_num].axis('off')
    plt.subplots_adjust( wspace=0.01, hspace=0.1)

def identity_augment(image, fake_input=0):
    return image

AUGMENTATIONS = [adjust_brightness, adjust_contrast, adjust_gamma, adjust_hue, adjust_jpeg_quality, adjust_saturation, identity_augment]
INTERVALS = [[-0.5,0.5], [0.5,2.5], [0.5,2.5], [-0.75,0.75], [0,75], [0,4.5], None]
AUGMENTATIONS_INTERVALS = list(zip(AUGMENTATIONS, INTERVALS))

def apply_augmentation(tiff): 
    function, interval =  random.choice(AUGMENTATIONS_INTERVALS)

    if  function.__name__ == 'identity_augment':
        return tiff

    if function.__name__ == 'adjust_jpeg_quality':
        new_t = [function(tiff[i], random.randint(*interval)) for i in range(tiff.shape[0])]
        return np.asarray(new_t)
        
    return function(tiff, random.uniform(*interval))

def get_tiff_selection_probs(tiff_paths, validation_tiff_num, shapefile):
    if 'weights.pkl' in os.listdir('cached'):
        with open('cached/weights.pkl', 'rb') as f:
            return pickle.load(f)
    tiff_paths = copy(tiff_paths)
    validation_tiff_path = tiff_paths.pop(validation_tiff_num)
    weights = []    
    for idx, tiff_file in enumerate(tiff_paths):
        _,_, regions = areas_covered(tiff_file, shapefile)
        weights.append(regions.shape[0]-abs(regions[regions.kiln_type == 'zigzag'].shape[0]-regions[regions.kiln_type == 'bull_trench'].shape[0]))

    with open('cached/weights.pkl', 'wb') as file:
        pickle.dump(weights, file)
        
    return weights

def image_data_generator(batch_size, is_training, tiff_paths, validation_tiff_num, shapefile, weights, trio=False, plot=False):
    image_dim = (300,300,3)
    tiff_paths = copy(tiff_paths)
    validation_tiff_path = tiff_paths.pop(validation_tiff_num)
    while True:
        file_name = validation_tiff_path
        if is_training:
            file_name = random.choice(tiff_paths)#, weights=weights)[0]
        # print(f'    {file_name}')
        tiff = load_tiff(file_name)
        mask, locations, kiln_type = create_kiln_mask(file_name, shapefile, trio=trio)

        batch_original_images = np.empty(shape=(batch_size, *image_dim))
        if trio:
            batch_masked_images = np.empty(shape=(batch_size, *image_dim))
        else:
            batch_masked_images = np.empty(shape=(batch_size, *image_dim[:2]))
        
        if plot:
          f, axs = plt.subplots(batch_size, 2, figsize=(5,15))
        
        k = itertools.cycle(sorted(set(kiln_type), reverse=True))
        for image in range(0, batch_size, 2):
            
            left, right, top, bottom = random.choice(locations[kiln_type == next(k)])
            fill_width = image_dim[0]-(right-left)
            fill_height = image_dim[1]-(bottom-top)
            lb = random.randint(0, fill_width)
            rb = fill_width - lb
            tb = random.randint(0, fill_height)
            bb = fill_height - tb

            while right+rb > tiff.shape[0] or left-lb < 0:
                lb = random.randint(0, fill_width)
                rb = fill_width - lb
            
            while top-tb < 0 or bottom+bb > tiff.shape[1]:
                tb = random.randint(0, fill_height)
                bb = fill_height - tb

            if plot:
                axs[image, 0].imshow(tiff[top-tb:bottom+bb,left-lb:right+rb])  
                axs[image, 1].imshow(mask[top-tb:bottom+bb,left-lb:right+rb])
                
            batch_original_images[image] = tiff[top-tb:bottom+bb,left-lb:right+rb]
            batch_masked_images[image] = mask[top-tb:bottom+bb,left-lb:right+rb]

        left = random.randint(0,tiff.shape[0]-image_dim[0])
        top = random.randint(0,tiff.shape[1]-image_dim[1])

            if plot:
                axs[image+1, 0].imshow(tiff[top:top+image_dim[1],left:left+image_dim[0]])  
                axs[image+1, 1].imshow(mask[top:top+image_dim[1],left:left+image_dim[0]])

            batch_original_images[image+1] = tiff[top:top+image_dim[1],left:left+image_dim[0]]
            batch_masked_images[image+1] = mask[top:top+image_dim[1],left:left+image_dim[0]]
          
        if plot:
          for ax in axs.flatten():
            ax.axis('off')
          plt.subplots_adjust( wspace=0.05, hspace=-0.65)

        
        if not trio:
            batch_masked_images = np.expand_dims(batch_masked_images, axis=3)

        # batch_original_images = apply_augmentation(batch_original_images)

        yield (batch_original_images, batch_masked_images)

def get_mask_kilns(shapefile, tiff=None, model=None, mask=None, threshold=0.2, trio=False):
    threshold = threshold
    if not model:
        threshold = 0
    if trio:
        a1, a2 = [], []
    else:
        a = []

    if isinstance(mask, np.ndarray):
        predicted_kilns = np.where(mask > threshold, 1, 0)
        if trio:
            for i in range(2):
                p_kilns = label(predicted_kilns[:,:,i])
                p_kilns = regionprops(p_kilns)
                if i == 0:
                    a1.extend([ np.array(potential_kiln.centroid)  for potential_kiln in p_kilns ])
                else:
                    a2.extend([ np.array(potential_kiln.centroid)  for potential_kiln in p_kilns ])
            return a1, a2

        predicted_kilns = label(predicted_kilns)
        predicted_kilns = regionprops(predicted_kilns)
        a.extend([ np.array(potential_kiln.centroid) for potential_kiln in predicted_kilns ])
        return a
    
    for row in range(0,3900,300):
        for col in range(0,3900,300):
            test_img = np.expand_dims(tiff[row:row+300, col:col+300], axis=0)
            mask = np.squeeze(model.predict(test_img, verbose=False))
            predicted_kilns = np.where(mask > threshold, 1, 0)
            if trio:
                for i in range(2):
                    p_kilns = label(predicted_kilns[:,:,i])
                    p_kilns = regionprops(p_kilns)
                    if i == 0:
                        a1.extend([ np.array(potential_kiln.centroid) + [row, col] for potential_kiln in p_kilns ])
                    else:
                        a2.extend([ np.array(potential_kiln.centroid) + [row, col] for potential_kiln in p_kilns ])
            else:
                predicted_kilns = label(predicted_kilns)
                predicted_kilns = regionprops(predicted_kilns)
                a.extend([ np.array(potential_kiln.centroid) + [row, col] for potential_kiln in predicted_kilns ])
    
    for row in [3795]:
        for col in range(0,3900,300):
            test_img = np.expand_dims(tiff[row:row+300, col:col+300], axis=0)
            mask = np.squeeze(model.predict(test_img, verbose=False))
            predicted_kilns = np.where(mask > threshold, 1, 0)
            if trio:
                for i in range(2):
                    p_kilns = label(predicted_kilns[:,:,i])
                    p_kilns = regionprops(p_kilns)
                    if i == 0:
                        a1.extend([ np.array(potential_kiln.centroid) + [row, col] for potential_kiln in p_kilns ])
                    else:
                        a2.extend([ np.array(potential_kiln.centroid) + [row, col] for potential_kiln in p_kilns ])
            else:
                predicted_kilns = label(predicted_kilns)
                predicted_kilns = regionprops(predicted_kilns)
                a.extend([ np.array(potential_kiln.centroid) + [row, col] for potential_kiln in predicted_kilns ])
    
    for col in [3795]:
        for row in range(0,3900,300):
            test_img = np.expand_dims(tiff[row:row+300, col:col+300], axis=0)
            mask = np.squeeze(model.predict(test_img, verbose=False))
            predicted_kilns = np.where(mask > threshold, 1, 0)
            if trio:
                for i in range(2):
                    p_kilns = label(predicted_kilns[:,:,i])
                    p_kilns = regionprops(p_kilns)
                    if i == 0:
                        a1.extend([ np.array(potential_kiln.centroid) + [row, col] for potential_kiln in p_kilns ])
                    else:
                        a2.extend([ np.array(potential_kiln.centroid) + [row, col] for potential_kiln in p_kilns ])
            else:
                predicted_kilns = label(predicted_kilns)
                predicted_kilns = regionprops(predicted_kilns)
                a.extend([ np.array(potential_kiln.centroid) + [row, col] for potential_kiln in predicted_kilns ])

    for col in [3795]:
        for row in [3795]:
            test_img = np.expand_dims(tiff[row:row+300, col:col+300], axis=0)
            mask = np.squeeze(model.predict(test_img, verbose=False))
            predicted_kilns = np.where(mask > threshold, 1, 0)
            if trio:
                for i in range(2):
                    p_kilns = label(predicted_kilns[:,:,i])
                    p_kilns = regionprops(p_kilns)
                    if i == 0:
                        a1.extend([ np.array(potential_kiln.centroid) + [row, col] for potential_kiln in p_kilns ])
                    else:
                        a2.extend([ np.array(potential_kiln.centroid) + [row, col] for potential_kiln in p_kilns ])
            else:
                predicted_kilns = label(predicted_kilns)
                predicted_kilns = regionprops(predicted_kilns)
                a.extend([ np.array(potential_kiln.centroid) + [row, col] for potential_kiln in predicted_kilns ])
    if trio:
        remove_duplicate_kilns(a1)
        remove_duplicate_kilns(a2)
        return a1, a2
    
    remove_duplicate_kilns(a)
    return a
    
def get_accuracy(tiff_path, area_of_interest, model, labels=None, threshold=0.2, trio=False):
    tiff = load_tiff(tiff_path)
    masked, _, _ = create_kiln_mask(tiff_path, area_of_interest, trio=trio)

    if trio:
        if not labels:
            labels1, labels2 = get_mask_kilns(area_of_interest, mask=masked, trio=trio)
        labels1, labels2 = labels
        predictions1, predictions2 = get_mask_kilns(area_of_interest, tiff=tiff, model=model, threshold=threshold, trio=trio)
    else:
        if not labels:
            labels = get_mask_kilns(area_of_interest, mask=masked, trio=trio)
        predictions = get_mask_kilns(area_of_interest, tiff=tiff, model=model, threshold=threshold, trio=trio)

    
    if not trio:
        if not labels:
            return 0, 0, len(predictions)
        if not predictions:
            return len(labels), 0, 0

        centroid_diffs_actual_predicted = pairwise_distances(labels, predictions)
        true_positives = np.count_nonzero(np.amin(centroid_diffs_actual_predicted, axis=1) < 20)
        false_positives = np.count_nonzero(np.amin(centroid_diffs_actual_predicted, axis=0) > 20)
        return len(labels), true_positives, false_positives


    true_bull_trenches = 0
    true_zigzags = 0
    false_bull_trenches = len(predictions1)
    false_zigzags = len(predictions2)

    if labels1:
        if predictions1:
            centroid_diffs_actual_predicted = pairwise_distances(labels1, predictions1)
            true_bull_trenches = np.count_nonzero(np.amin(centroid_diffs_actual_predicted, axis=1) < 20)
            false_bull_trenches = np.count_nonzero(np.amin(centroid_diffs_actual_predicted, axis=0) > 20)
    
    if labels2:
        if predictions2:
            centroid_diffs_actual_predicted = pairwise_distances(labels2, predictions2)
            true_zigzags = np.count_nonzero(np.amin(centroid_diffs_actual_predicted, axis=1) < 20)
            false_zigzags = np.count_nonzero(np.amin(centroid_diffs_actual_predicted, axis=0) > 20)
    
    return len(labels1), len(labels2), true_bull_trenches, false_bull_trenches, true_zigzags, false_zigzags

def contains(file_name, conditions):
    file_name = file_name[:-4:].split('_')
    for c in conditions:
        if c not in file_name:
            return False
    return True

def load_selected_losses(conditions):
    path = os.path.join('histories')
    files = os.listdir(path)
    files = ['histories/'+ f for f in files]
    if conditions:
        files =  [f for f in files if contains(f, conditions)]

    data = pd.DataFrame(data={'epoch':list(range(1,26))})
    for f in files:
        with open(f, 'rb') as f:
            name = f.name.split('/')[1].split('_')[5:]
            name.remove('psnr')
            name = ' '.join(name)[:-4:]
            d = pickle.load(f)
            data[name+' training'] = d['loss']
            data[name+' validating'] = d['val_loss']

    data = pd.melt(data, ['epoch'], ignore_index=True)
    data[['regularisation function', 'activation function', 'learning rate', 'upsampling interpolation', 'training mode']] = data['variable'].str.split(' ', expand=True, n=4)
    data.drop(data.columns[1], axis=1, inplace=True)
    data.set_index('epoch')

    return data

def show_predictions(model, inputs, masks, save=False, epoch=None, backend='PDF'):
    p = model.predict(inputs, verbose=0)
    num_inputs = inputs.shape[0]
    f, axs = plt.subplots(num_inputs, 3, figsize=(12,15))
    titles = ['Input Image', 'Mask', 'Prediction']
    plot_data = zip(titles, [inputs, masks, p])
    for idx, ax, info in zip(np.repeat(range(num_inputs), 3), axs.flatten(), itertools.cycle(plot_data)):
        title, array = info
        if idx == 0:
            ax.set_title(title)
        array = np.clip(array, a_min=0, a_max=1)
        ax.imshow(array[idx])
        ax.axis('off')
    f.tight_layout()

    if save:
        os.makedirs(f'visualisations/training_progress/{model.name}/', exist_ok=True)
        plt.savefig(f'visualisations/training_progress/{model.name}/model_at_epoch_{epoch}.pdf', backend=backend)
        # plt.close()

def remove_duplicate_kilns(centroids, min_distance=20):
    if centroids:
        dist = pairwise_distances(centroids)
        same = []
        for i in range(0, len(dist[0])):
            for j in range(i+1, len(dist[0])):
                if dist[i, j] < min_distance:
                    same.append(j)
        
        removed = 0
        for idx in set(same):
            del centroids[idx-removed]
            removed += 1