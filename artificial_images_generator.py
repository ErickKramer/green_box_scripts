#!/usr/bin/env python3
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from imageio import imread, imwrite
import pandas as pd
import yaml

class Vec2D(object):
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y

class BoundingBox(object):
    def __init__(self):
        self.nonzero_rows = None
        self.nonzero_cols = None
        self.min_coords = Vec2D()
        self.max_coords = Vec2D()

def get_bb_from_mask(segmentation_mask: np.array) -> BoundingBox:
    '''Returns a BoundingBox object with information extracted
    from the given segmentation mask.

    Keyword arguments:
    segmentation_mask: np.array -- a 2D numpy array representing a grayscale image
                                   with a single object in it, where the assumption
                                   is that non-zero pixels represent the object

    '''
    bb = BoundingBox()
    bb.nonzero_rows, bb.nonzero_cols  = np.where(segmentation_mask)
    bb.min_coords.x, bb.max_coords.x = (np.min(bb.nonzero_cols), np.max(bb.nonzero_cols))
    bb.min_coords.y, bb.max_coords.y = (np.min(bb.nonzero_rows), np.max(bb.nonzero_rows))
    return bb

def get_bb(coords):
    bb = BoundingBox()
    bb.min_coords.x, bb.max_coords.x = (np.min(coords[0]), np.max(coords[0]))
    bb.min_coords.y, bb.max_coords.y = (np.min(coords[1]), np.max(coords[1]))
    return bb

def generate_transformation(bb: BoundingBox, boundaries: tuple) -> np.array:
    '''Generates a homogeneous transformation matrix of type int that translates,
    rotates, and scales the given bounding box, ensuring that the
    transformed points are within the given boundaries.

    Keyword arguments:
    bb: BoundingBox -- a BoundingBox object
    boundaries: tuple -- coordinate boundaries; assumed to represent
                         the (row, column) sizes of an image

    '''
    use_transformation = False
    t = None
    while not use_transformation:
        use_transformation = True
        rectangle_points = np.array([[bb.min_coords.x, bb.min_coords.x, bb.max_coords.x, bb.max_coords.x],
                                     [bb.min_coords.y, bb.max_coords.y, bb.min_coords.y, bb.max_coords.y]])
        rectangle_points = np.vstack((rectangle_points, [1., 1., 1., 1.]))

        # we generate a random rotation angle
        random_rot_angle = np.random.uniform(0, 2*np.pi)
        random_rot_matrix = np.array([[np.cos(random_rot_angle), -np.sin(random_rot_angle)],
                                      [np.sin(random_rot_angle), np.cos(random_rot_angle)]])

        # we generate a random translation within the image boundaries
        random_translation_x = np.random.uniform(-bb.min_coords.x, boundaries[1]-bb.max_coords.x)
        random_translation_y = np.random.uniform(-bb.min_coords.y, boundaries[0]-bb.max_coords.y)
        translation_vector = np.array([[random_translation_x], [random_translation_y]])

        # we generate a random scaling factor between 0.5 and 1.5
        # of the original object size
        random_scaling_factor = np.random.uniform(0.5, 1.0)
        s = np.array([[random_scaling_factor, 0., 0.],
                      [0., random_scaling_factor, 0.],
                      [0., 0., 1.]])

        t = np.hstack((random_rot_matrix, translation_vector))
        t = np.vstack((t, np.array([0., 0., 1.])))
        t = t.dot(s)

        transformed_bb = t.dot(rectangle_points)
        transformed_bb = np.array(transformed_bb, dtype=int)
        for point in transformed_bb.T:
            if point[0] < 0 or point[0] >= boundaries[1] or \
               point[1] < 0 or point[1] >= boundaries[0]:
                use_transformation = False
                break
    return t

def augment_data(img_dir_name: str,
                 background_img_dir: str,
                 images_per_background: int,
                 annotations_file: str,
                 output_dir: str,
                 classes_to_id: dict) -> None:
    '''Given the images in "img_dir_name", each of which is assumed to have a
    single object, generates a new set of images in which the objects are put
    on the backgrounds in "background_img_dir" and are transformed (translated,
    rotated, scaled) in random fashion. For each background and image combination,
    "images_per_background" images are generated.

    Args:
    * img_dir_name: str -- path to a directory with image files and object
                         segmentation masks (img_dir_name is expected to have
                         a directory "object_masks" with the segmentation masks,
                         such that if an image is called "test.jpg", its segmentation
                         mask will have the name "test_mask.jpg")
    * background_img_dir: str -- path to a directory with background images for augmentation
    * images_per_background: int -- number of images to generate per given background
    * annotations_file: str -- yaml file for the annotations of the training images
    * output_dir:str -- directory to store the genenated images
    '''

    max_objects_per_image = 6

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    os.chdir(output_dir)

    if 'train' in annotations_file:
        if not os.path.isdir('training_images'):
            os.mkdir('training_images')
    elif 'val' in annotations_file:
        if not os.path.isdir('validation_images'):
            os.mkdir('validation_images')

    backgrounds = os.listdir(background_img_dir)
    print('Number of backgrounds images ', len(backgrounds))
    objects = os.listdir(img_dir_name)
    print('Number of Objects ', len(objects))

    images = os.listdir(os.path.join(img_dir_name,objects[0]))

    # Generating images paths
    background_paths = []
    for background in backgrounds:
        background_path = os.path.join(background_img_dir, background)
        background_paths.append(background_path)

    objects_paths = []
    for object in objects:
        object_path = os.path.join(img_dir_name, object)
        objects_paths.append(object_path)

    augmented_img_counter = 0

    training_images = []
    val_images = []

    for background_path in background_paths:
        background_img_original = np.array(imread(background_path), dtype=np.uint8)
        # print(background_path)
        for _ in range(images_per_background): # Number of images generated using that backgrounds
            background_img = np.array(background_img_original, dtype=np.uint8)
            augmented_objects = []
            for _ in range(np.random.randint(1,max_objects_per_image)): # Number of objects in the image
                object_path = objects_paths[np.random.randint(len(objects))]
                # print(object_path)
                object_class = object_path.replace(img_dir_name+'/','')
                # print(object_class)
                images = os.listdir(object_path)
                # print(images)
                image_full_name = images[np.random.randint(len(images))]
                # print('Number of images ', len(images))
                while not '.jpg' in image_full_name:
                    image_full_name = images[np.random.randint(len(images))]

                image_name, img_extension = image_full_name.split('.')
                image_path = os.path.join(object_path,image_full_name)

                img = np.array(imread(image_path), dtype=np.uint8)
                segmentation_mask_path = os.path.join(object_path, 'object_masks',
                                                      image_name + '_mask.'+'jpg')
                segmentation_mask = np.array(imread(segmentation_mask_path), dtype=np.uint8)

                # we get the bounding box of the object and generate a transformation matrix
                bb = get_bb_from_mask(segmentation_mask)


                t = generate_transformation(bb, background_img.shape)

                # the object points are transformed with the given transformation matrix
                obj_coords = np.vstack((bb.nonzero_cols[np.newaxis],
                                        bb.nonzero_rows[np.newaxis],
                                        np.ones(len(bb.nonzero_rows), dtype=int)))
                transformed_obj_coords = t.dot(obj_coords)
                transformed_obj_coords = np.array(transformed_obj_coords, dtype=int)

                transformed_bb = get_bb(transformed_obj_coords)

                # the object is added to the background image
                for i, point in enumerate(transformed_obj_coords.T):
                    x = point[0]
                    y = point[1]
                    background_img[y, x] = img[obj_coords[1, i], obj_coords[0, i]]

                id = classes_to_id[object_class]
                augmented_object = {'class_id': id,
                                    'xmin': int(bb.min_coords.x),
                                    'xmax': int(bb.max_coords.x),
                                    'ymin': int(bb.min_coords.y),
                                    'ymax': int(bb.max_coords.y)}

                augmented_objects.append(augmented_object)

            if 'train' in annotations_file:
                output_path = os.path.join('training_images',image_name + '_' + str(augmented_img_counter) \
                 + '.' + img_extension)
            elif 'val' in annotations_file:
                output_path = os.path.join('validation_images',image_name + '_' + str(augmented_img_counter) \
                 + '.' + img_extension)

            # print(output_path)
            imwrite(output_path, background_img)
            training_images.append({'image_name': output_path,'objects': augmented_objects})

            # image_name = output_path
            augmented_img_counter += 1
            print('Augmented {} out of {} images '.format(augmented_img_counter, images_per_background * len(backgrounds)))

    generate_annotation_file(annotations_file, training_images)
    # print(training_images)

def generate_annotation_file(annotations_file, training_images):
    # print(training_images)
    with open(annotations_file, 'w') as annotation_file:
        yaml.dump(training_images, annotation_file,default_flow_style=False,
                      encoding='utf-8')

if __name__ == '__main__':
    img_dir_name = sys.argv[1]
    background_img_dir = sys.argv[2]
    images_per_background = int(sys.argv[3])
    annotations_file = sys.argv[4]
    output_dir = sys.argv[5]

    with open('classes.yml', 'r') as class_file:
        id_to_classes = yaml.load(class_file, Loader=yaml.FullLoader)

    classes_to_id = dict()
    for key,value in id_to_classes.items():
        classes_to_id[value] = key

    print('Generating artificial images...')
    augment_data(img_dir_name, background_img_dir, images_per_background, annotations_file, output_dir, classes_to_id)
    print('Artificial images generation complete')