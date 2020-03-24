import tensorflow as tf
import numpy as np
import align.detect_face
import facenet
import imageio
import cv2
import os


DB_ROOT_DIR = "./Face_Database/"
INPUT_SIZE = 160  # Cropped face image size
MARGIN = 40  # Screen border width

# Create MTCNN
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)


def detect_face(img):
    """
    Face detection based on MTCNN.
    :param img: Image to be detected
    :return: Cropped and prewhitened face image
    """
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = align.detect_face.detect_face(img=img,
                                                      minsize=20,
                                                      pnet=pnet, rnet=rnet, onet=onet,
                                                      threshold=[0.6, 0.7, 0.7],
                                                      factor=0.709)

    if len(bounding_boxes) < 1:
        print("No Faces in Image!")
        return None

    if len(bounding_boxes) > 1:
        print("More than One Face in Image!")
        return None

    det = np.squeeze(bounding_boxes[0, 0:4])
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0] - MARGIN / 2, 0)
    bb[1] = np.maximum(det[1] - MARGIN / 2, 0)
    bb[2] = np.minimum(det[2] + MARGIN / 2, img_size[1])
    bb[3] = np.minimum(det[3] + MARGIN / 2, img_size[0])
    cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]

    aligned = cv2.resize(cropped, (INPUT_SIZE, INPUT_SIZE))
    prewhitened = facenet.prewhiten(aligned)

    return prewhitened


def add_user(img_path, group_name):
    """
    Add a single user, save face in user image as feature vector.
    :param img_path: Path to user image
    :param group_name: Name of user group
    :return: None
    """
    db_path = os.path.join(DB_ROOT_DIR, group_name)
    if not os.path.exists(db_path):
        os.makedirs(db_path)

    if not os.path.exists(img_path):
        print("Invalid Image Path: " + img_path + "!")
        return

    extension = img_path.split(".")[-1]
    if extension != "jpg" and extension != "png" and extension != 'jfif':
        print("Invalid Image Path: " + img_path + "!")
        return
    img = imageio.imread(img_path)
    prewhitened = detect_face(img)

    if prewhitened is not None:
        img_name = str(img_path.split("/")[-1].split(".")[0])
        imageio.imsave(os.path.join(db_path, img_name + '.jpg'), prewhitened)
        print(img_name + " Has Been Added!")


def add_user_batch(img_path, group_name):
    """
    Add users in batches, read all images in the path and save faces as feature vectors
    :param img_path: Path to the folder holding user images
    :param group_name: Name of user group
    :return: None
    """
    db_path = os.path.join(DB_ROOT_DIR, group_name)
    if not os.path.exists(db_path):
        os.makedirs(db_path)

    if not os.path.exists(img_path):
        print("Invalid Image Path: " + img_path + "!")
        return

    for image in os.listdir(img_path):
        extension = image.split(".")[-1]
        if extension == "jpg" or extension == "png" or extension == "jfif":
            img = imageio.imread(os.path.join(img_path, image))
            prewhitened = detect_face(img)

            if prewhitened is not None:
                img_name = str(image.split('.')[0])
                imageio.imsave(os.path.join(db_path, img_name + '.jpg'), prewhitened)
                print(str(image.split('.')[0]) + " Has Been Added!")


def delete_user(user_name, group_name):
    """
    Delete all data of specified user.
    :param user_name: Name of user
    :param group_name: Name of user group
    :return: None
    """
    db_path = os.path.join(DB_ROOT_DIR, group_name)
    if not os.path.exists(db_path):
        print("Group Not Exist: " + group_name + "!")
        return

    delete_flag = False
    for feature in os.listdir(db_path):
        if feature.replace(".", "_").split("_")[0] == user_name:
            delete_flag = True
            try:
                os.remove(os.path.join(db_path, feature))
                print(os.path.join(db_path, feature) + " Has Been Deleted!")
            except OSError:
                print(os.path.join(db_path, feature) + " Cannot be Deleted!")
    if not delete_flag:
        print("User Not Exist: " + user_name + "!")


def delete_user_group(group_name):
    """
    Delete all data in specified user group.
    :param group_name: Name of user group
    :return: None
    """
    db_path = os.path.join(DB_ROOT_DIR, group_name)

    exist = os.path.exists(db_path)
    if not exist:
        print("Group Not Exist: " + group_name + "!")
        return

    for feature in os.listdir(db_path):
        try:
            os.remove(os.path.join(db_path, feature))
        except OSError:
            print(db_path + feature + " Cannot be Deleted!")

    try:
        os.rmdir(db_path)
        print(group_name + " Has Been Deleted!")
    except OSError:
        print(db_path + " Cannot be Deleted!")


if __name__ == '__main__':
    # add_user(img_path="./Face_Image/Hu Ge.jfif", group_name="test_group")
    add_user_batch(img_path="./Face_Image", group_name="test_group")
    # delete_user(user_name="HuGe", group_name="test_group")
    # delete_user_group(group_name="test_group")
