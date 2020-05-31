# Face_Detection_Alignment
## modifications xvdp
I had to make a few ajustments to have it run with cudatoolkit 10.1. conda environment with:

`tensorflow=1.3.0, python==3.6.10, numpy==1.19.0rc1, matplotlib==3.2.1, opencv==4.2.0, scipy==1.2.3, menpo==0.8.1, menpofit==0.6.1, menpowidgets==0.3.0`
np.load() and np.expand_dims() in detect_face.py


## original
Face Detection and Alignment Tool
3D projection landmarks (84) and 2D multi-view landmarks(39/68)

Environment:
Tensorflow 1.3, menpo, python 3.5

Train:
CUDA_VISIBLE_DEVICES="1" python train.py --train_dir=ckpt/3D84 --batch_size=8 --initial_learning_rate=0.0001 --dataset_dir=3D84/300W.tfrecords,3D84/afw.tfrecords,3D84/helen_testset.tfrecords,3D84/helen_trainset.tfrecords,3D84/lfpw_testset.tfrecords,3D84/lfpw_trainset.tfrecords,3D84/ibug.tfrecords,3D84/menpo_trainset.tfrecords --n_landmarks=84

Test:
3D model: 84
2D model: frontal68/Union68/Union86(better)

Pretrained Models:
https://drive.google.com/open?id=1DKTeRlJjyo_tD1EluDjYLhtKFPJ9vIVd
