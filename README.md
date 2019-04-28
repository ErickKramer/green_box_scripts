# Green box scripts
This **SOMETHING** contains a set of scripts used to collect images, generate masks of the images, and generate artificial images using any of the cameras in the *HSR robot*

**NOTE:**
* The scripts are themselves robot independent, but the instructions here given are specific for the HSR.
* It is preferred to use any of the stereo cameras rather than the RGB-D because the stereo cameras are HD.

## Content:
* `green_box_image_collector.py`
* `green_box_mask_generator.py`
* `data_augmenter.py`

## Requirements
* If collecting images remotely:
    * Connect to the network **b-it-bots@home**
    * Set up external ros master by executing

        `export ROS_MASTER_URI=http://hsrb.local:11311`

## Setup

### Camera options

### Lighting conditions

### Base



## Steps


## Instructions
### Positioning the camera
1. Open in a web browser the address:
    `http://192.168.50.201/admin/`
2. Access *Servo States* page
3. Introduce a value inside the *command* cell (or dragging the *command bar*) for the joint required to move in order to position the desired camera in the right position relative to the base of the box




### Generating background image
* Execute the command

    `python3 green_box_image_collector.py image_topic_name image_dir_name number_of_images`

    Example:

    `python3 green_box_image_collector.py /hsrb/head_r_stereo_camera/image_raw background_02_04_19 1`

### Collecting images of object
* Execute the command

    `python3 green_box_image_collector.py image_topic_name image_dir_name number_of_images`

    Example:

    `python3 green_box_image_collector.py /hsrb/head_rgbd_sensor/rgb/image_raw sponge_02_04_19 10`

### Generating object masks
* Execute the command

    `python3 green_box_mask_generator.py image_dir_name background_image_name background_threshold`

    Example:

    `python3 green_box_mask_generator.py sponge_02_04_19_front background_02_04_19/background_02_04_19_0.jpg 30`

### Generating artificial images
`./data_augmenter.py [img_dir] [background_img_dir] [number_of_images_per_background]`


## Training process
**Inside models/research**
**It is required to execute `export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim`**
python3 object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr


## Structure

## TODOS:
* Add images of the setup

## Team
