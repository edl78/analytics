# T-SNE analytics 
- T-distributed Stochastic Neighbour Embedding. PCA-like analysis of dimensionality reduction, applied on featurespace from Resnet101 up until softmax layer. Trained on imagenet. So features are general for imagenet. Do the same analysis on fine tuned network for better class separation. The main idea here is to map clusters of data in feature space and gain insight into feature space distributions.

## Note on feature extraction for tSNE
The feature extraction employed in the tSNE class is done by employing a resnet101 model pretrained on imagenet, i.e. non of our weed data has been used to modify it. There is also code that is currently broken (incompatibility between old torch version of model and newer version of torch employed in the code, thus the model needs updating of some layer keys etc) which could, if fixed, use the resnet18 model pre-trained on the openweeds dataset. **TBD**

## Build and run
- Ensure that you are running this on the same machine as the MongoDB instance.
- First fill in `env.list` with username, password and other data such as downloaded weed_data path, do not check this in.
- Edit `.env` and set paths for `WEED_DATA_PATH` and `TSNE_DATA_PATH`.
- Note that this service also needs access to the downloaded weed_data for analysis.
- For better separation of clusters we use the resnet18 backbone from the model trained on our weed data. Put this model in a folder called `model/`
- To use the reset18 model fill in `TSNE_MODEL=resnet18` in the env.list file and put follow the next instuction below on model placement. If a standard resnet101 pretrained on imagenet is desired for feature extraction fill in `TSNE_MODEL=standard`
- The pretrained model with backbone used for feature extraction can be found in the artefacts folder on the download server.  
- Build with `sh build_analytics_no_conda.sh` and run with `docker-compose up -d` but beware that Nvidia Rapids pip support is experimental so the Dockerfile_no_conda might need some attention from time to time.
- run_analytics.sh is provided for easier debug but require a manual start of code/app.py. If desired, analytics can also be added to the weed_annotation docker-compose. Analytics should be started before weed_annotations.


## Architecture
The analytics container is built with classes for pandas dataset generation, pytorch dataset class for feature extraction, a t-sne class which holds the pretrained network and the t-sne clustering algorithm. This is held together with a flask http rest api server which handles reqests and servs results.

![](doc_img/architecture.png)


## Example t-sne graph
![](doc_img/tsne_plot.png)

# Academic citation

Please see the [main webpage](https://openweeds.linkoping-ri.se/howtocite.html) for information on how to cite.

# Issues?

Please use the github issues for this repo to report any issues on it.