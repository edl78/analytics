# T-SNE analytics 
- T-distributed Stochastic Neighbour Embedding. PCA-like analysis of dimensionality reduction, applied on featurespace from Resnet101 up until softmax layer. Trained on imagenet. So features are general for imagenet. Do the same analysis on fine tuned network for better class separation. The main idea here is to map clusters of data in feature space and extract training and validation splits. 

## Build and run
First fill in env.list with username and password, do not check this in.
Either run_analytics.sh which require a manual start of code/app.py or run it together with the whole application in docker-compose.all.yml from the cvat_stuff root.

## Architecture
The analytics container is built with classes for pandas dataset generation, pytorch dataset class for feature extraction, a t-sne class which holds the pretrained network and the t-sne clustering algorithm. This is held together with a flask http rest api server which handles reqests and servs results.

![](doc_img/architecture.png)


## Debug
This container runs a conda environment. Conda run command swallows std out and std err which leads to silent failures if any. To debug this start the container manually. This can be fixed by adding "--no-capture-output" in the ENTRYPOINT command.
- First change the Dockerfile to exclude the ENTRYPOINT by commenting this line out. 
- Build the image by sh build_analytics.sh
- run the analytics container. sh run_analytics(_dopamine).sh
- run: `conda init bash` in the terminal inside the container
- you will be encuraged to log out and into the shell again. Simply run `bash` or do this by finding your container id on the host by `docker ps`. Run: `docker exec -it your_container_id /bin/bash`
- in the new terminal list your envs: `conda env list`, find your predefined env from the Dockerfile and change by `conda activate myenv` for example.
- Now start your code and watch it fail with informative error messages. Find and correct the bugs!
- When happy which bugfixes, undo changes in the Dockerfile, rebuild the image and test if the flow works as intended.
## Better debug
- You can also do debugging in vscode which is a lot easier. Start the container as per above. In the terminal in vscode, do `conda init bash`, `bash`, `conda activate myenv`, press F1, write python, find choose interpreter and choose the one pointing to your env. Then run debug as usual.

## Example t-sne graph
![](doc_img/tsne_plot.png)