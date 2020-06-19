# Deep Learning Image Super Resolution



## <u>What is the Project About</u>

* This is a deep learning project based on the [Image Super-Resolution Using Deep Convolutional Networks - SRCNN](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html) paper using the **PyTorch** deep learning library.



## <u>Framework and Dependencies</u>

* **The project is built on PyTorch 1.4.** 
* You will need MATLAB to execute the `.m` files.



## <u>Directory Structure</u>

* The following is the directory structure to arrange everything for the project.

  ```
  ├───input
  │   ├───bicubic_2x
  │   ├───bicubic_4x
  │   ├───bicubic_rgb_2x
  │   ├───bicubic_rgb_4x
  │   ├───General100
  │   ├───Set14
  │   ├───Set5
  │   ├───T91
  │   ├───T91_G100
  |    train_mscale.h5
  ├───outputs
  └───src
  ```

* `input`: contains the datasets that are used for training and testing. The `train_mscale.h5` is the training datasets that gets generated after running the `generate_train.m` file. 

  * Currently the model has been trained on both `T91` and `General100` image datasets. Both of these datasets are merged into `T91_G100` folder. The same corresponds to in the `generate_train.m` file.
  * The `bicubic_x` folders contain the blurred images that we use for testing. Generate those images using the `bicubic.py` file inside the `src` folder.

* The `outputs` folder will contain all the output files along with the trained model.

* `src` contains the python and MATLAB files.

***Note***: *I have take the MATLAB codes from the [SRCNN-Keras](https://github.com/YapengTian/SRCNN-Keras) repository. The original `generate_train.m` file generate greyscale sub-images. I have formatted the code so as to generate colored (RGB) sub-images. As such, in this project, you will be able to train a neural network model that can carry out super-resolution on RGB images.*  *Please go through the code for more details*.



## <u>Dataset</u>

* You will find the datasets used in this project and more super-resolution datasets [here](https://github.com/xinntao/BasicSR/wiki/Prepare-datasets-in-LMDB-format).



## <u>Execution</u>

* `generate_train.m`: To generate the `train_mscale.h5` sub-images.
* Execute the python scripts while being within the `src` folder in the terminal.
  * `train.py`: For training the SRCNN model.
  * `test.py`: To test on the test images.



## <u>Results</u>





## <u>References</u>

* [Image Super-Resolution Using Deep Convolutional Networks - SRCNN](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html).

* [ SRCNN-Keras](https://github.com/YapengTian/SRCNN-Keras): For the `generate_train.m` file to create the `train_mscale.h5` training data.
* [SRCNN-Tensorflow](https://github.com/jinsuyoo/SRCNN-Tensorflow): For the test images.

