# siamese_person_re_id
Siamese Network for Person Re-ID

## Dependencies
```
TensorFlow version >= 1.5
NumPy 1.14.0
SciPy 1.0.0
PIL 5.0.0
IPython 6.2.1
jupyter 1.0.0
```

## Dataset

Download the MARS dataset from here:
* [MARS](http://www.liangzheng.com.cn/Project/project_mars.html) - The dataset used for training

Next put the data into a directory with the following structure:

```
MARS\
  categories\
    0001\
      ....jpg
      ....jpg
      ....jpg
    0002\
      ....jpg
    0003\
      ....jpg
    0004\
      ....jpg
    ...
```  
You can creade a TFRecord file for TensorFlow consumption. This will create a training and validation set for your dataset:
```
$ python3 create_tf_record.py --tfrecord_filename=mars --dataset_dir=/path/to/dataset/
```

## Training
The training script creates the TFRecord file and then trains the model on the generated TFRecord file.
```
$ python3 train_siamese_network.py --data /path/to/dataset/
```
The updates are stored in `./train.log/` which can be seen in TensorBoard.

## Using the pre-trained model
Download the pre-trained model weights from the following link and put them in a folder named `model_siamese`.
* [Model](https://drive.google.com/drive/folders/1n6JV36gQb9RpYuPcHPXJd2Wled_U_PNu?usp=sharing) - pre-trained model weights

## Inference
To test the model you can run the test script with two test images as such:
```
$ python3 test_siamese_network.py --img1 /path/to/image1/ --img2 /path/to/image2
```

Alternatively you can use the two provided Jupyter notebooks to test the network.
The following notebook runs the test on 20 random validation set image pairs and displays the results.
```
$ jupyter notebook test_siamese_network_jupyter_validation.ipynb
```

The following notebook runs the test on the 7 test images provided in the main directory and displays the results.
```
$ jupyter notebook test_siamese_network_jupyter_images.ipynb
```

