# T-SNE analytics 
- T-distributed Stochastic Neighbour Embedding. PCA-like analysis of dimensionality reduction, applied on featurespace from Resnet101 up until softmax layer. Trained on imagenet. So features are general for imagenet. Do the same analysis on fine tuned network for better class separation. The main idea here is to map clusters of data in feature space and gain insight into feature space distributions.

## Build and run
- First fill in env.list with username, password and other data such as downloaded weed_data path, do not check this in.
- Note that this service also needs access to the downloaded weed_data for analysis.
- Build with `sh build_analytics_no_conda.sh` and run with `docker-compose up -d` 
- run_analytics.sh is provided for easier debug but require a manual start of code/app.py. If desired, analytics can also be added to the weed_annotation docker-compose. Analytics should be started before weed_annotations.

## Architecture
The analytics container is built with classes for pandas dataset generation, pytorch dataset class for feature extraction, a t-sne class which holds the pretrained network and the t-sne clustering algorithm. This is held together with a flask http rest api server which handles reqests and servs results.

![](doc_img/architecture.png)


## Example t-sne graph
![](doc_img/tsne_plot.png)