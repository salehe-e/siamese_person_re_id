# siamese_person_re_id
Siamese Network for Person Re-ID

## Dataset

Download the MARS dataset from here:
* [MARS](http://www.liangzheng.com.cn/Project/project_mars.html) - The dataset used for training

Next put the data into a directory with the following structure:

```
MARS\
  bbox_train\
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
Then creade a TFRecord file for TensorFlow consumption. This will create a training and validation set for your dataset:
```
$ python3 create_tf_record.py --tfrecord_filename=mars --dataset_dir=./mars/
```

## Training
Run the training script to train the model on the specified TFRecord file.
```
$ python3 siamese_network.py
```
## Pre-Trained Model
Download thte pre-trained model weights from here:
* [Model](https://drive.google.com/drive/folders/1n6JV36gQb9RpYuPcHPXJd2Wled_U_PNu?usp=sharing) - pre-trained model weights

## Inference
To test the model use the Jupyter notebook provided.
```
$ jupyter notebook jupyter_siamese_test.ipynb
```
The current notebook will run the test on a few unseen examples in the `mars_test` folder. In order to run the test for your own test data, run:
```
$ python produce_test_tfrecord.py /path/to/img1.jpg /path/to/img2.jpg
``` 
and then use the Jupyter notebook with the new test image folder.
