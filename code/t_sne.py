import pandas
import torch
import torchvision.models as models
from torch.hub import load_state_dict_from_url
from weed_data_class import WeedData
from torchvision import transforms
import numpy as np
#from sklearn.manifold import TSNE
#from tsnecuda import TSNE
from cuml.manifold import TSNE
#from cuml.manifold import TSNE as cumlTSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os


# Define the architecture by modifying resnet.
# Original code is here http://tiny.cc/8zpmmz 
class ResNet101(models.ResNet):
    def __init__(self, num_classes=1000, pretrained=True, **kwargs):
        # Start with the standard resnet101
        super().__init__(
            block=models.resnet.Bottleneck,
            layers=[3, 4, 23, 3],
            num_classes=num_classes,
            **kwargs
        )
        if pretrained:
            state_dict = load_state_dict_from_url(
                models.resnet.model_urls['resnet101'],
                progress=True
            )
            self.load_state_dict(state_dict)

    # Reimplementing forward pass.
    # Replacing the forward inference defined here 
    # http://tiny.cc/23pmmz
    def _forward_impl(self, x):
        # Standard forward for resnet
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Notice there is no forward pass through the original classifier.
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x



#class to perform t-sne methods on annotation data
class T_SNE():
    def __init__(self, save_dir):
        self.save_dir = save_dir 
        os.makedirs(name=self.save_dir, mode=0o755, exist_ok=True)       
        # train on the GPU or on the CPU, if a GPU is not available
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.data_transform = transforms.Compose([transforms.Normalize(
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])])
        self.model = ResNet101()
        self.model.eval()
        self.model.to(self.device)
        self.dataset = None
        self.cluster_data = None        
        if(torch.cuda.is_available()):
            print('run on gpu')
        else:
            print('run on cpu')


    def set_pd_dataset(self, pd_dataset_file, class_map):
        self.dataset = WeedData(pandas_file=pd_dataset_file, transform=self.data_transform, device=self.device, class_map=class_map)    


    def generate_clusters(self):       
        # scale and move the coordinates so they fit [0; 1] range
        def scale_to_01_range(x):
            # compute the distribution range
            value_range = (np.max(x) - np.min(x))

            # move the distribution so that it starts from zero
            # by extracting the minimal value from all its values
            starts_from_zero = x - np.min(x)

            # make the distribution fit [0; 1] by dividing by its range
            return starts_from_zero / value_range        

        data_loader = torch.utils.data.DataLoader(
            dataset=self.dataset, batch_size=1, shuffle=False, num_workers=2)

        features = []
        labels = []
        label_map = self.dataset.get_classmap()

        try:
            for i, data in enumerate(data_loader, 0):
                print(i)            
                image, label = data
                image = image.to(self.device)                               
                output = self.model.forward(image)

                current_outputs = output.detach().cpu().numpy()
                features.append(current_outputs) 
                label_cpu = int(label.detach().cpu().numpy())
                
                labels.append(label_map[label_cpu])
        
        except FileNotFoundError as e:
            print('error reading img file on path')                       
            return
            

        #loop over 5-50 in perplexity and save images for get request? Very fast compute on gpu.
        print('run t-sne')
        tsne = TSNE(n_components=2, perplexity=25).fit_transform(np.squeeze(features))
        
        # extract x and y coordinates representing the positions of the images on T-SNE plot
        tx = tsne[:, 0]
        ty = tsne[:, 1]

        tx = scale_to_01_range(tx)
        ty = scale_to_01_range(ty)

        # initialize a matplotlib plot
        fig = plt.figure()
        ax = fig.add_subplot(111)

        #handle arbitrary number of classes and give colors, up to 10 + 148 classes
        #use tab colors first since they are strong and not so faint. Will be
        #easier to see in Dashboard plots.
        plot_colors = []
        for key in mcolors.CSS4_COLORS:
            plot_colors.append(key)

        tab_colors = []
        for key in mcolors.TABLEAU_COLORS:
            tab_colors.append(key)

        color_step = int(len(mcolors.CSS4_COLORS)/len(label_map))
        j = 0
        colors_per_class = {}
        for object_label in label_map:
            if(j < len(tab_colors)-1):
                colors_per_class[label_map[j]] =  tab_colors[j]    
            else:
                colors_per_class[label_map[j]] =  plot_colors[j*color_step]
            j += 1

        # for every class, we'll add a scatter plot separately
        for label in colors_per_class:
            # find the samples of the current class in the data
            indices = [i for i, l in enumerate(labels) if l == label]

            # extract the coordinates of the points of this class only
            current_tx = np.take(tx, indices)
            current_ty = np.take(ty, indices)

            col = colors_per_class[label]
            # add a scatter plot with the corresponding color and label
            ax.scatter(current_tx, current_ty, c=col, label=label)
            

        # build a legend using the labels we set previously
        ax.legend(loc='best')
        
        plt.savefig(self.save_dir + '/tsne_plot.png')


       # init the plot as white canvas
        plot_size = 2048
        tsne_plot = 255 * np.ones((plot_size*2, plot_size*2, 3), np.uint8)
        offset = 1024
        print('plot with images')
        #get all images and insert in tsne_plot
        for i in range(0, len(labels)):
            print(i)            
            image = self.dataset.load_image(i)
            im = image.detach().numpy()
            im_rgb = np.transpose(im, (1,2,0)) 
            height, width, channels = im_rgb.shape
            try:
                tsne_x_min = int((tx[i] * plot_size)) + offset
                tsne_y_min = int((ty[i] * plot_size)) + offset
            except:
                print('error in parsing tsne data, quit, no data in: ' + self.save_dir)
                return            
            tsne_x_max = tsne_x_min + width
            tsne_y_max = tsne_y_min + height
            im_rgb_scaled = im_rgb * 255
            im_uint = np.asarray(im_rgb_scaled).astype(np.uint8)            

            try:         
                tsne_plot[tsne_y_min: tsne_y_max, tsne_x_min: tsne_x_max, :] = im_uint
            except:
                print('wrong size for tsne img in tsne plot!')
                continue            

        #save the img, no plot
        plt.imsave(self.save_dir + '/tsne_plot_imgs.png', tsne_plot)

        
        
        

def main():
    t_sne = T_SNE(save_dir='/tsne')
    t_sne.set_pd_dataset('/pickled_weed/pd.pkl')
    t_sne.generate_clusters()    



if __name__ == '__main__':
    main()
