from numba import njit, jit, prange
from numba.typed import Dict
import numba
import glob
import os
import pandas as pd
import PIL 
from PIL import Image, ImageDraw
from collections import Counter
import numpy as np
import yaml
import tqdm 
def read_project_variables():
    with open("variables.yaml", 'r') as file:
        return yaml.safe_load(file)

variables = read_project_variables()



@njit(parallel=True)
def map_(array, adict): 
    result = np.zeros_like(array)
    for i in prange(len(array)): 
        result[i] = adict[array[i]]
    return result

@njit
def histogram_equalization(image, pixel_number): 
    d_m = image.max()
    unique_pixels = np.unique(image.flatten())
    pixel_counter = np.bincount(image.flatten())
    pixel_counter_cumsum = np.cumsum(pixel_counter)
    values_map = Dict.empty(
        key_type=numba.types.uint8,
        value_type=numba.types.int64
    )
    for pixel in unique_pixels: 
        new_value = round(d_m/pixel_number * pixel_counter_cumsum[pixel])
        values_map[pixel] = new_value
    #new_image = np.vectorize(values_map.get)(image)
    new_image = map_(image.flatten(), values_map)
    new_image = new_image.reshape(image.shape)
    return new_image

@jit(forceobj=True)
def _histogram_equalization_3d(image): 
    image = image.mean(axis=-1).astype(np.uint8)
    #print(image.shape)
    w = image.shape[0]
    h = image.shape[1]
    #new_planes = [histogram_equalization(image[:, :, 0], w*h).reshape(w, h, 1),
    #             histogram_equalization(image[:, :, 1],  w*h).reshape(w, h, 1),
    #             histogram_equalization(image[:, :, 2],  w*h).reshape(w, h, 1)]
    #new_planes = np.concatenate(new_planes, axis=-1)
    new_planes = histogram_equalization(image,  w*h)#.reshape(w, h, 1)
    return Image.fromarray(new_planes.astype(np.uint8))

def histogram_equalization_3d(image):
    return _histogram_equalization_3d(np.array(image))

def get_file_data(x, images_directory, use_hist_equalization, bar):
    bbs = []
    classes = []
    for i in range(len(x)): 
        bbs.append((x['xmin'].iloc[i], x['ymin'].iloc[i], x['xmax'].iloc[i], x['ymax'].iloc[i]))
        classes.append(x['class'].iloc[i])
    
    image = Image.open(os.path.join(images_directory, x['filename'].iloc[0]))
    result = {
        'image': image,
        'bounding_boxes': bbs,
        
        'classes': classes
        
    }
    
    if use_hist_equalization: 
        image_eq = histogram_equalization_3d(image)
        result['image_eq'] = image_eq
    bar.update(1)
    return pd.Series(result, index=result.keys())

def return_whole_dataset(path_to_annotations='annotations.csv', 
                        images_directory='data', 
                        use_hist_equalization=False):
    df = pd.read_csv(path_to_annotations)
    bar = tqdm.tqdm(total=len(set(df['filename'])))
    df = df.groupby('filename').apply(lambda x: get_file_data(x, 
                                    images_directory, 
                                    use_hist_equalization, bar)).reset_index()
    bar.close()                                
    return df.values
    
def count_IoU(calculated_bb, true_bb): 
    def _count_IoU(c_bb, t_bb):
        c_pixels = {(pixel_x, pixel_y) for pixel_x in range(c_bb[0], c_bb[2]+1) for pixel_y in range(c_bb[1], c_bb[3]+1)}
        t_pixels = {(pixel_x, pixel_y) for pixel_x in range(t_bb[0], t_bb[2]+1) for pixel_y in range(t_bb[1], t_bb[3]+1)}
        inter = len(c_pixels.intersection(t_pixels))
        union = len(c_pixels.union(t_pixels))
        return inter/union
    def matrix_distance(x, y): 
        import numpy as np
        distances = np.zeros((len(x), len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                distance = np.linalg.norm(np.array(x[i])-np.array(y[j]))
                distances[i,j] = distance
        return distances

    def return_pairings(x, y): 
        import itertools
        x, y = np.arange(len(x)), np.arange(len(y))
        combinations = []
        reversed_ = False
        if len(x) < len(y): 
            reversed_ = True
            x, y = y, x
        permutations = itertools.permutations(list(x), int(y.max() + 1))
        for permutation in permutations:
            combinations.append(list(zip(permutation, y)))

        if reversed_: 
            combinations = [[row[::-1] for row in rows] for rows in combinations]
        return combinations 

    def return_best_pairs(x, y): 
        distances = matrix_distance(x, y)
        possible_pairings = return_pairings(x, y)
        sums = []
        for pairing in possible_pairings:
            sum_ = 0 
            for pair in pairing: 
                sum_ += distances[pair[0], pair[1]]
            sums.append(sum_)
        best_paring = possible_pairings[np.argmin(sums)]
        return best_paring


    calculated_bb = np.array(calculated_bb)
    true_bb = np.array(true_bb)

    calculated_centers = np.concatenate([
       (calculated_bb[:, 0] + calculated_bb[:, 2]).reshape(-1, 1)/2,
       (calculated_bb[:, 1] + calculated_bb[:, 3]).reshape(-1, 1)/2
    ], axis=-1)

    true_centers = np.concatenate([
       (true_bb[:, 0] + true_bb[:, 2]).reshape(-1, 1)/2,
       (true_bb[:, 1] + true_bb[:, 3]).reshape(-1, 1)/2
    ], axis=-1)
    IoUs = []
    best_pairings = return_best_pairs(true_centers, calculated_centers) 
    for pair in best_pairings: 
        IoUs.append(_count_IoU(calculated_bb[pair[1]], true_bb[pair[0]]))
    return IoUs + [0]*(max(len(true_bb) - len(best_pairings), len(calculated_bb) - len(best_pairings)))
