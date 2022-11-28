import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class WeedData(Dataset):
    def __init__(self, pandas_file, device, transform=None, class_map=None):
        self.df = pd.read_pickle(pandas_file)
        self.device = device
        self.transform = transform        
        self.class_map = class_map

    def __getitem__(self, idx):       
        entry = self.df.iloc[idx]  
        box = entry["points"]
        #print(entry['shape_type'])
        #print(box)
        xmin=0
        xmax=0
        ymin=0
        ymax=0
        try:
            img_path = entry["img_path"]
            print('img_path: ' +  img_path)
            if(img_path[0] == '/'):
                #starting with absolute path
                path_split =  img_path.split('/')
                img_path = '/weed_data/' + '/'.join(path_split[2:])
                print('new img path: ' + img_path)
            img = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, TypeError) as e:
            print(e, flush=True)
            raise FileNotFoundError
        if(entry['shape_type'] == 'rectangle'):
            xmin = float(box["xmin"])
            xmax = float(box["xmax"])
            ymin = float(box["ymin"])
            ymax = float(box["ymax"])
        elif(entry['shape_type'] == 'polygon' or entry['shape_type'] == 'points'):
            #take extreme points from polygon and create a new box
            x_points = box[0:len(box):2]
            y_points = box[1:len(box):2]
            xmin = x_points[np.argmin(x_points)]
            xmax = x_points[np.argmax(x_points)]
            ymin = y_points[np.argmin(y_points)]
            ymax = y_points[np.argmax(y_points)]
        else:
            print('unhandled type')
            print(entry['shape_type'])

        label = entry["object_class"]
        #use index as int representation of object_class
        #where is no good returns array, use index, need only first occurence!
        label_index = self.class_map.index(label)
        
        bbox_img = img.crop((xmin, ymin, xmax, ymax))
        bbox_img_arr = np.asarray(bbox_img)

        # convert everything into a torch.Tensor                
        label_tensor = torch.as_tensor(label_index, dtype=torch.int8)
        img = np.array(bbox_img_arr)/255.0
        img_tensor = torch.as_tensor(np.transpose(img, (2,0,1)), dtype=torch.float32)
        
        if(self.transform is not None):
            img_tensor = self.transform(img_tensor)         
        return img_tensor, label_tensor

    def load_image(self, index):
        entry = self.df.iloc[index]  
        box = entry["points"]
        try:
            img_path = entry["img_path"]
            print('img_path: ' +  img_path)
            if(img_path[0] == '/'):
                #starting with absolute path
                path_split =  img_path.split('/')
                img_path = '/weed_data/' + '/'.join(path_split[2:])
                print('new img path: ' + img_path)
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError as e:
            print('failed to read img data!')
            raise
                
        xmin=0
        xmax=0
        ymin=0
        ymax=0
        
        if(entry['shape_type'] == 'rectangle'):
            xmin = float(box["xmin"])
            xmax = float(box["xmax"])
            ymin = float(box["ymin"])
            ymax = float(box["ymax"])
        elif(entry['shape_type'] == 'polygon' or entry['shape_type'] == 'points'):
            #take extreme points from polygon and create a new box
            x_points = box[0:len(box):2]
            y_points = box[1:len(box):2]
            xmin = x_points[np.argmin(x_points)]
            xmax = x_points[np.argmax(x_points)]
            ymin = y_points[np.argmin(y_points)]
            ymax = y_points[np.argmax(y_points)]

        
        label = entry["object_class"]
        #use index as int representation of object_class
        label_index = np.where(self.class_map == label)

        bbox_img = img.crop((xmin, ymin, xmax, ymax))
        bbox_img_arr = np.asarray(bbox_img)

        # convert everything into a torch.Tensor                
        label_tensor = torch.as_tensor(label_index, dtype=torch.int8)
        img = np.array(bbox_img_arr)/255.0
        img_tensor = torch.as_tensor(np.transpose(img, (2,0,1)), dtype=torch.float32)
        
        return img_tensor

    def __len__(self):
        return len(self.df.index)

    def get_classmap(self):
        return self.class_map
        