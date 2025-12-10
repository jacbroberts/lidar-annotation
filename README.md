# Natural Language LiDAR Annotations

## Install instructions

### Windows

* Run the install.bat folder and follow the instructions in the prompt
* Test to see if everything setup correctly by running "python test_installation.py"
* Run the quick_start.py script to see a quick demo of the project
* Run demo.py to get the full pipeline running, from downloading the kitti dataset to visualization

## Downloads

download_kitti.py: Downloads the full kitti odometry dataset and organizes it.
download_semantic_kitti.py: Downloads kitti's labels

## Salsanext

We use Salsanext for our segmentation model. If you don't have weights already, the pipeline will use uninitialized random weights from a heuristic model. Run train_salsanext.py to train the segmentation model on your dataset.

## NLP Model
We use Llama for our model. If Llama fails to load, a generic NLP model will attempt to annotate for you.