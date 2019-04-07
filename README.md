# tf_object_detection
custom object detection using tensorflow object-detection-api

###### overview


1. Setup
2. How to make your own Dataset
3. Train on your own custom Object
4. predict object on your own trained model  

# Setup
1. Go to [tensorflow/model](https://github.com/tensorflow/models) clone this repository
2. To run your tensorflow-object-detection-api we need to configure add some Dependencies
   ```
   # For CPU
   pip install tensorflow
   # For GPU
   pip install tensorflow-gpu
   ```
       
  The remaining libraries can be installed
  
  ```python
    sudo apt-get install protobuf-compiler python-pil python-lxml python-tk
    pip install --user Cython
    pip install --user contextlib2
    pip install --user jupyter
    pip install --user matplotlib`  
   ```
  
  
## Protobuf Compilation

The Tensorflow Object Detection API uses Protobufs to configure model and
training parameters. Before the framework can be used, the Protobuf libraries
must be compiled. This should be done by running the following command from
the tensorflow/models/research/ directory:


``` bash
# From tensorflow/models/research/
protoc object_detection/protos/*.proto --python_out=.
```

**Note**: If you're getting errors while compiling, you might be using an incompatible protobuf compiler. If that's the case, use the following manual installation

## Manual protobuf-compiler installation and usage

**If you are on linux:**

Download and install the 3.0 release of protoc, then unzip the file.

```bash
# From tensorflow/models/research/
wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
unzip protobuf.zip
```

Run the compilation process again, but use the downloaded version of protoc

```bash
# From tensorflow/models/research/
./bin/protoc object_detection/protos/*.proto --python_out=.
```

## Add Libraries to PYTHONPATH

When running locally, the tensorflow/models/research/ and slim directories
should be appended to PYTHONPATH. This can be done by running the following from
tensorflow/models/research/:

 ```bash
# From tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# run this command to setup in your locally
From tensorflow/models/research/
sudo python3 setup.py install
```


###### For more detail how to configure and run your objection-detection-api on other platform other than LINUX go to [https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)

3. Now once you install and configure `cd models/research/object_detection` then run py typing `Jupyter Notebook` in command line open `object_detection_tutorial.ipynb` and run all the cell.

# Using on you'r own dataset 

## make your own dataset

1. Open terminal make a new dir of name images `mkdir images`
2. Download any images from google into your `images` directory for example I'm using **stapler** images download atleast        250-300 images.
3. Once you Download all the images then we need to convert all the images into `.xml` file for further use to convert it        into `.csv` file so for that go to [Labeling-images](https://github.com/tzutalin/labelImg.git) and clone the repo by using    `git clone https://github.com/tzutalin/labelImg.git` 
4. Once you clone it now you need to setup envoirment to use it run the below command 

```sudo apt-get install pyqt5-dev-tools
sudo pip3 install -r requirements/requirements-linux-python3.txt
cd labelImg
make qt5py3
python3 labelImg.py
python3 labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE]
```
###### For more detail about installation on other platform other than LINUX and how to convert images into .xml file go to [https://github.com/tzutalin/labelImg](https://github.com/tzutalin/labelImg) or refer to this [video](https://www.youtube.com/watch?v=K_mFnvzyLvc&list=PLQVvvaa0QuDcNK5GeCQnxYnSSaar2tpku&index=3)


5. Once you done that make **two dir** in your images directory **test and train** copy **90%** of your images with thier      xml file into **train** directory and **10%** of remaning images with thier xml into **test** directory

6. Make a new dir `mkdir custom_object_detection` move you'r **images** directory itno it. In the same dir i.e                  custom_object_detection make **two** file **xml_to_csv.py and generate_tfrecord.py**  and also make **training** d          directory and **data** directory which we will use later.

## At this point, you should have the following structure, 

```
custom_object_detection
-data/
--test_labels.csv
--train_labels.csv
-images/
--test/
---testingimages.jpg
--train/
---testingimages.jpg
--...yourimages.jpg
-training
-xml_to_csv.py
-generate_tfrecord.py
```

7. Open xml_to_csv.py into you'r favourite editor copy below code into it.

```import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    for directory in ['train','test']: # looping to all your train and test dir 
        image_path = os.path.join(os.getcwd(), 'images/{}'.format(directory)) 
        xml_df = xml_to_csv(image_path)    # converting xml to csv
        xml_df.to_csv('data/{}_labels.csv'.format(directory), index=None) # saving it into two different file train and test 
                                                                          # into data directory
        print('Successfully converted xml to csv.')


main()
``` 
and save the file

8. Now open generate_tfrecord.py and copy the below code into it.

```"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python3 generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=data/train.record

  # Create test data:
  python3 generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=data/test.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('image_dir', '', 'Path to images')
FLAGS = flags.FLAGS


# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'stapler':
        return 1
    else:
        None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(FLAGS.image_dir)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()
```
and save it

**Note**:- If you have more than one class change class_text_to_init function into 

```def class_text_to_int(row_label):
    if row_label == 'stapler':
        return 1  #here add more class if you want to train on multiple class 
    else:
        None
```

9.  Now run You'r xml_to_csv.py by using `python xml_to_csv.py` this will convert all your images into scv file and stored       it into `data/train_labels.csv` and `data/test_labels.csv`

10. Now run generate_tfrecord.py by using
```
python3 generate_tfrecord.py --csv_input=data/train_labels.csv --output_path=data/train.record --image_dir=images/

python3 generate_tfrecord.py --csv_input=data/test_labels.csv --output_path=data/test.record --image_dir=images/

```
this will create test.record and train.record file into `data/` directory

10. Here, we have two options. We can use a pre-trained model, and then use transfer learning to learn a new object, or we could learn new objects entirely from scratch. The benefit of transfer learning is that training can be much quicker, and the required data that you might need is much less. For this reason, we're going to be doing transfer learning here.

TensorFlow has quite a few pre-trained models with checkpoint files available, along with configuration files. You can do all of this yourself if you like by checking out their [configuring jobs documentation](https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/samples/configs/ssd_mobilenet_v1_pets.config). The object API also provides some [sample configurations](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs) to choose from.

I am going to go with mobilenet, using the following checkpoint and configuration file

Run the below code in you'r terminal in `custom_object_detection` folder.

```
wget https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/samples/configs/ssd_mobilenet_v1_pets.config

wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz

```

11. After downloading tar.gz file extrat that file in same directory.

12. Open `ssd_mobilenet_v1_pets.config` in you'r favourite editor change the below code or paste all the given below code into it.

```
# SSD with Mobilenet v1, configured for the stapler dataset.
# Users should configure the fine_tune_checkpoint field in the train config as
# well as the label_map_path and input_path fields in the train_input_reader and
# eval_input_reader. Search for "${YOUR_GCS_BUCKET}" to find the fields that
# should be configured.

model {
  ssd {
    num_classes: 1  # if you have more than one class replace 1 with your number of class
    box_coder {
      faster_rcnn_box_coder {
        y_scale: 10.0
        x_scale: 10.0
        height_scale: 5.0
        width_scale: 5.0
      }
    }
    matcher {
      argmax_matcher {
        matched_threshold: 0.5
        unmatched_threshold: 0.5
        ignore_thresholds: false
        negatives_lower_than_unmatched: true
        force_match_for_each_row: true
      }
    }
    similarity_calculator {
      iou_similarity {
      }
    }
    anchor_generator {
      ssd_anchor_generator {
        num_layers: 6
        min_scale: 0.2
        max_scale: 0.95
        aspect_ratios: 1.0
        aspect_ratios: 2.0
        aspect_ratios: 0.5
        aspect_ratios: 3.0
        aspect_ratios: 0.3333
      }
    }
    image_resizer {
      fixed_shape_resizer {
        height: 300
        width: 300
      }
    }
    box_predictor {
      convolutional_box_predictor {
        min_depth: 0
        max_depth: 0
        num_layers_before_predictor: 0
        use_dropout: false
        dropout_keep_probability: 0.8
        kernel_size: 1
        box_code_size: 4
        apply_sigmoid_to_scores: false
        conv_hyperparams {
          activation: RELU_6,
          regularizer {
            l2_regularizer {
              weight: 0.00004
            }
          }
          initializer {
            truncated_normal_initializer {
              stddev: 0.03
              mean: 0.0
            }
          }
          batch_norm {
            train: true,
            scale: true,
            center: true,
            decay: 0.9997,
            epsilon: 0.001,
          }
        }
      }
    }
    feature_extractor {
      type: 'ssd_mobilenet_v1'
      min_depth: 16
      depth_multiplier: 1.0
      conv_hyperparams {
        activation: RELU_6,
        regularizer {
          l2_regularizer {
            weight: 0.00004
          }
        }
        initializer {
          truncated_normal_initializer {
            stddev: 0.03
            mean: 0.0
          }
        }
        batch_norm {
          train: true,
          scale: true,
          center: true,
          decay: 0.9997,
          epsilon: 0.001,
        }
      }
    }
    loss {
      classification_loss {
        weighted_sigmoid {
        }
      }
      localization_loss {
        weighted_smooth_l1 {
        }
      }
      hard_example_miner {
        num_hard_examples: 3000
        iou_threshold: 0.99
        loss_type: CLASSIFICATION
        max_negatives_per_positive: 3
        min_negatives_per_image: 0
      }
      classification_weight: 1.0
      localization_weight: 1.0
    }
    normalize_loss_by_num_matches: true
    post_processing {
      batch_non_max_suppression {
        score_threshold: 1e-8
        iou_threshold: 0.6
        max_detections_per_class: 100
        max_total_detections: 100
      }
      score_converter: SIGMOID
    }
  }
}

train_config: {
  batch_size: 24    # you can also change the batch size
  optimizer {
    rms_prop_optimizer: {
      learning_rate: {
        exponential_decay_learning_rate {
          initial_learning_rate: 0.004
          decay_steps: 800720
          decay_factor: 0.95
        }
      }
      momentum_optimizer_value: 0.9
      decay: 0.9
      epsilon: 1.0
    }
  }
  fine_tune_checkpoint: "ssd_mobilenet_v1_coco_11_06_2017/model.ckpt"
  from_detection_checkpoint: true
  load_all_detection_checkpoint_vars: true
  # Note: The below line limits the training process to 200K steps, which we
  # empirically found to be sufficient enough to train the pets dataset. This
  # effectively bypasses the learning rate schedule (the learning rate will
  # never decay). Remove the below line to train indefinitely.
  num_steps: 200000
  data_augmentation_options {
    random_horizontal_flip {
    }
  }
  data_augmentation_options {
    ssd_random_crop {
    }
  }
}

train_input_reader: {
  tf_record_input_reader {
    input_path: "data/train.record"   # input training data 
  }
  label_map_path: "data/object-detection.pbtxt"  # path where label_map
}

eval_config: {
  metrics_set: "coco_detection_metrics"
  num_examples: 1100
}

eval_input_reader: {
  tf_record_input_reader {
    input_path: "data/test.record"   # input test data
  }
  label_map_path: "training/object-detection.pbtxt"  # path where label_map
  shuffle: false
  num_readers: 1
}
```

13. Inside training dir, add object-detection.pbtxt:

```
# you can add more item as per you class length in my case there is only one class
item {
  id: 1
  name: 'stapler'
}

```

14. Now copy **data, images, ssd_mobilenet_v1_coco_11_06_2017, training** directory and **ssd_mobilenet_v1_pets.config** file into tensorflow model `models/research/object_detection` .

15. move **ssd_mobilenet_v1_pets.config** into **training** directory .

16.And now, the moment of truth! From within models/object_detection:.

```
python3 train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_pets.config
```

Barring errors, you should see output like:

```
INFO:tensorflow:global step 11788: loss = 0.6717 (0.398 sec/step)
INFO:tensorflow:global step 11789: loss = 0.5310 (0.436 sec/step)
INFO:tensorflow:global step 11790: loss = 0.6614 (0.405 sec/step)
INFO:tensorflow:global step 11791: loss = 0.7758 (0.460 sec/step)
INFO:tensorflow:global step 11792: loss = 0.7164 (0.378 sec/step)
INFO:tensorflow:global step 11793: loss = 0.8096 (0.393 sec/step)
```

Your steps start at 1 and the loss will be much higher. Depending on your GPU and how much training data you have, this process will take varying amounts of time. On something like a 1080ti, it should take only about an hour or so. If you have a lot of training data, it might take much longer. You want to shoot for a loss of about ~1 on average (or lower). I wouldn't stop training until you are for sure under 2. You can check how the model is doing via TensorBoard. Your models/object_detection/training directory will have new event files that can be viewed via TensorBoard.

From models/object_detection, via terminal, you start TensorBoard with:

`tensorboard --logdir='training'`

This runs on 127.0.0.1:6006 (visit in your browser)

Looks good enough, but does it detect stapler?!

In order to use the model to detect things, we need to export the graph

 
 

