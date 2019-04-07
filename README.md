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
###### For more detail about installation on other platform other than LINUX go to [https://github.com/tzutalin/labelImg](https://github.com/tzutalin/labelImg)

5. Once you run the above 
