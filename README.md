# try-faces-predict-with-keras
Try faces predict with Keras. And using TensorFlow.



This project is created for learning purposes.<br>

It is not a quality that can be used in production.



## using

push target image (faces) to directory of `detect_images`

Label name is each directory name.

#### Run cnn script

```
python3 faces_cnn.py
```



### Run predict script

```bash
python3 face_predict.py <target_image_file>
```



## Docker

Can execute predict processing in Docker.

(Please execute after learning)

```
# image build
docker build -t face-predict-py .

# connect docker
docker run -it --rm --name face-predict face-predict-py /bin/bash
python face_predict.py <target_image_file>
```



## config

config write constant in `faces_cnn.py`.

You can rewrite as you need.



## Requirement

using Python version `3.6.6`.

And using these libraries

* pillow
* keras
* tensorflow
* sklearn
* h5