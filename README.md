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
   - `# For CPU`
      `pip install tensorflow`
      `# For GPU`
       `pip install tensorflow-gpu` 
       
  The remaining libraries can be installed
  
  ` sudo apt-get install protobuf-compiler python-pil python-lxml python-tk
    pip install --user Cython
    pip install --user contextlib2
    pip install --user jupyter
    pip install --user matplotlib`  
   
  
  
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


5. Once you done that make **two dir** in your images dir **test and train** copy **90%** of your images with thier xml file into **train dir** and **10%** of remaning images with thier xml into **test dir**

6. Make a new dir `mkdir custom_object_detection` move you'r **images** dir itno it. In the same dir i.e                        custom_object_detection make **two** file **xml_to_csv.py and generate_tfrecord.py** 

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

**Note**:- If you have more than one class change class_text_to_init function into ```def class_text_to_int(row_label):
    if row_label == 'stapler':
        return 1
    else:
        None
        ```
        
        

