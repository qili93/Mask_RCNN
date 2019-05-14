# LMS with MaskRCNN

## MaskRCNN without LMS

```shell
# clone repository
git clone https://github.com/matterport/Mask_RCNN.git
cd Mask_RCNN
source activate paipy3

# install dependencies
pip3 install -r requirements.txt
python3 setup.py install

# download pre-trained COO weights (mask_rcnn_coco.h5)
wget https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5

# Train a new model starting from ImageNet weights. Also auto download COCO dataset
cd samples/coco/
python3 coco.py train --dataset=/home/qilibj/coco-data/ --model=coco --download=True
```

## Enable LMS

```shell
# increase batch size -> IMAGES_PER_GPU = 4 in cooc.py
# tensorflow.python.framework.errors_impl.InternalError: CUDA runtime implicit initialization on GPU:0 failed. Status: out of memory
python3 coco.py train --dataset=/home/qilibj/coco-data/ --model=coco

# enable MaskRCNN with LMS 
# vim $HOME/anaconda2/envs/paipy3/lib/python3.6/site-packages/mrcnn/model.py (line 2362)
from tensorflow_large_model_support import LMS
lms_callback = LMS()
lms_callback.batch_size = 4
callbacks.append(lms_callback)

# verify LMS
# increase batch size -> IMAGES_PER_GPU = 4 in cooc.py
python3 coco.py train --dataset=/home/qilibj/coco-data/ --model=coco

```

## Appendix: install dependencies

```shell
# resolve dependency #1
# skil opencv-python as this is included in conda environment
# dependecy list of imgaug = ['scipy', 'scikit-image (>=0.11.0)', 'numpy (>=1.15.0)', 'six', 'imageio', 'Pillow', 'matplotlib', 'Shapely', 'opencv-python']
sudo yum install geos-devel
# skip opencv-python (already in conda env)
pip install imgaug --no-dependencies

# resolve dependency #2
pip install cython
pip install pycocotools

# resolve dependency #3
pip install mrcnn
pip install keras
```