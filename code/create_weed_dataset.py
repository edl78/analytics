from weeds import Weeds
import pandas
import os


class PickledWeed():
    def __init__(self, task_name, save_dir, dataset_dir):
        self.task_name = task_name
        self.save_dir = save_dir
        self.dataset_dir = dataset_dir
        os.makedirs(name=self.save_dir, mode=0o755, exist_ok=True)
        self.weeds = Weeds()
        self.class_map = self.weeds.get_object_classes_for_all_annotations()

    def make_pandas_dataset(self):        
        cursor = self.weeds.get_task_data(self.task_name)
        matches = {}
        for index, item in enumerate(cursor):            
            matches[item['name']] = item
        
        annotations = []
        for key in matches:
            local_task_name = key
            annotations_cursor = self.weeds.get_annotations_for_task(local_task_name)
            for index, item in enumerate(annotations_cursor):
                annotations.append(item)
        
        df = pandas.DataFrame()
        for annotation in annotations:        
            dataEntry = {}      
            dataEntry['task_name'] = annotation['task_name'] 
            dataEntry['object_class'] = annotation['object_class']
            
            if(annotation['shape_type'] == 'rectangle'): 
                dataEntry['points'] = {
                    'xmin': annotation['points'][0],
                    'ymin': annotation['points'][1],
                    'xmax': annotation['points'][2],
                    'ymax': annotation['points'][3]
                }
            elif(annotation['shape_type'] == 'polygon' or annotation['shape_type'] == 'points'):
                dataEntry['points'] = annotation['points']                

            dataEntry['shape_type'] = annotation['shape_type']              
            dataEntry['img_path'] = self.dataset_dir + '/' + annotation['img_path']
            dataEntry['img_width'] = annotation['img_width']
            dataEntry['img_height'] = annotation['img_height']
            #sneaky, df.append will return the new dataframe...
            df = df.append(dataEntry, ignore_index=True)
                    
        
        df.to_pickle(self.save_dir + '/pd.pkl')

    

#for isolated testing
def main():
    print('create pandas dataset of weeds')
    save_dir = '/pickled_weed'    
    pickledWeed = PickledWeed(task_name='FieldData 20200520145736 1L GH020068', save_dir=save_dir, dataset_dir='/weed_data')
    pickledWeed.make_pandas_dataset()
    print('done, pickled pandas frame found at: ' + save_dir)
    


if __name__ == '__main__':
    main()