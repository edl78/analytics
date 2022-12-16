from flask import Flask, send_file, Response, send_from_directory, request
from t_sne import T_SNE
import torch
from weeds import Weeds
from create_weed_dataset import PickledWeed
import threading
import os


app = Flask(__name__)

tsne_lock = threading.Lock()

@app.route('/tsne/<string:task_name>/<string:img_name>', methods=["GET"])
def serve_tsne_plot(task_name, img_name):
    save_dir = '/tsne/'
    file_name = save_dir + task_name +'/' + img_name
    if(os.path.exists(file_name)):        
        return send_file(file_name, mimetype='image/png', as_attachment=True)
    else:
        return 404


@app.route('/')
def welcome_page():
    return 'RESTful API for flask server with T-SNE plots, use GET on /tsne/task_name/tsne_plot.png or tsne_plot_imgs.png to retrieve an image but replace blanks with underscore in task_name! :)'


def start_worker_thread(task_name):
    #limit access to compute heavy algorithms
    tsne_lock.acquire()
    weeds = Weeds()      
    if(task_name == 'all'):
        task_cursor = weeds.get_all_db_tasks()        
        for task in task_cursor:
            run_tsne_on_task(task['name'])    
    else:
        #bug: in the tasks collection, name is given with underscore
        #task_name = task_name.replace('_', ' ')
        task_cursor = weeds.get_task_data(task_name)
        print('check db for matching task...')
        for task in task_cursor:
            #safe guard since seg_(task_name) and (task_name)
            #is returned
            print(task['name'])
            if(task['name'] == task_name):
                print('found task, start t-sne')
                run_tsne_on_task(task['name'])
    
    #test this and see if memory allocation after t-sne run is back to normal,
    #otherwise it has ~10GB allocated without doing any work
    print('empty cache')
    torch.cuda.empty_cache()
    print('done')
    
    tsne_lock.release()


@app.route('/<string:task_name>/run_tsne')
def run_tsne(task_name):
    print('calc tsne request for ' + task_name)
    thread = threading.Thread(target=start_worker_thread, kwargs={'task_name': task_name})
    thread.start()
    return '200'


def run_tsne_on_task(task_name):
    print('run tsne for task: ' + task_name)
    task_name_path = task_name.replace(' ', '_')
    pickle_save_dir = '/pickled_weed/' + task_name_path
    print('create pandas dataset')
    pickledWeed = PickledWeed(task_name=task_name, save_dir=pickle_save_dir, dataset_dir='/obdb')
    #/fs/sefs1/obdb 
    pickledWeed.make_pandas_dataset()
    weeds = Weeds()
    class_map = weeds.get_object_classes_for_all_annotations()
    print('prepare t-sne task')
    tsne_save_dir = '/tsne/'+task_name_path
    tsne_model = os.environ['TSNE_MODEL']
    tsne = T_SNE(save_dir=tsne_save_dir, model=tsne_model, model_path='/model/resnet18_model.pth')
    pickle_file_path = pickle_save_dir + '/pd.pkl'
    tsne.set_pd_dataset(pd_dataset_file=pickle_file_path, class_map=class_map)
    print('generate clusters')
    try:
        tsne.generate_clusters()
    except (FileNotFoundError, TypeError) as e:
        print('error reading file for task: ' +  task_name)
    print('done, clean up', flush=True)
    #remove big file after cluster generation
    os.remove(pickle_file_path)    



def main():    
    print('starting flask server for analytics')
    app.run(debug=True, host='0.0.0.0', port=int(os.environ['ANALYTICS_PORT']), threaded=True)



if __name__ == '__main__':
    main()

    #FieldData_20200520145736_1L_GH020068
    #manual testing:
    #http://localhost:5001/FieldData_20200520145736_1L_GH020068/run_tsne
