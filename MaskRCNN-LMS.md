# LMS with MaskRCNN

## MaskRCNN without LMS

```shell
# clone repository
git clone https://github.com/matterport/Mask_RCNN.git
cd Mask_RCNN
source activate paipy3

# install dependencies
# remove opencv-python as already in conda package
pip install -r requirements.txt
python3 setup.py bdist_wheel
pip install dist/mask_rcnn-2.1-py3-none-any.whl --no-cache-dir

# download pre-trained COO weights (mask_rcnn_coco.h5)
wget https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5

# Train a new model starting from ImageNet weights. Also auto download COCO dataset
cd samples/coco/
python3 coco.py train --dataset=/home/qilibj/coco-data/ --model=coco --download=True

# Increase batch size to 8 -> IMAGES_PER_GPU = 8 in cooc.py
# expected output: OOM in V100 (16G)
python3 coco.py train --dataset=/home/qilibj/coco-data/ --model=coco
```

## Enable LMS

```shell
# enable MaskRCNN with LMS 
# vim $HOME/anaconda2/envs/paipy3/lib/python3.6/site-packages/mrcnn/model.py (line 2362)
from tensorflow_large_model_support import LMS
lms_callback = LMS()
lms_callback.batch_size = 8
callbacks.append(lms_callback)

# Increase batch size to 8 -> IMAGES_PER_GPU = 8 in cooc.py
# 2019-05-15 04:20:34.896244: W tensorflow/c/c_api.cc:696] Operation '{name:'training/SGD/gradients/AddN_26' id:14849 op device:{} def:{{{node training/SGD/gradients/AddN_26}} = AddN[N=2, T=DT_FLOAT, _class=["loc:@Square_36"]](lms/swapin_training_SGD_gradients_Square_36_grad_Mul_1_0:0, training/SGD/gradients/mrcnn_mask_deconv/conv2d_transpose_grad/Conv2DBackpropFilter)}}' was changed by updating input tensor after it was run by a session. This mutation will have no effect, and will trigger an error in the future. Either don't modify nodes after running them or create a new session.
python3 coco.py train --dataset=/home/qilibj/coco-data/ --model=coco
```

## Appendix: install dependencies

```shell
# resolve dependency #1
# skip opencv-python (already in conda env)
sudo yum install geos-devel
pip install scipy scikit-image numpy six imageio Pillow matplotlib Shapely
pip install imgaug --no-dependencies

# resolve dependency #2
pip install cython
pip install pycocotools

# resolve dependency #3
pip install keras

# dependency #4
pip install -r requirements.txt
```


