from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import align.detect_face
import facenet
import cv2
import sys
import os
import gc


FPS = 30
INPUT_SIZE = 160  # Cropped face image size
MARGIN = 40  # Screen border width

DB_ROOT_DIR = "./Face_Database/"
MODEL_DIR = './models/'


# Create MTCNN
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    sess_ = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess_.as_default():
        pnet, rnet, onet = align.detect_face.create_mtcnn(sess_, None)


def load_and_align_data(img, face_size):
    """
    Detect and crop faces in frame based on MTCNN.
    :param img: Image to be detected
    :param face_size: Minimum detected face size
    :return: Whether the frame contains faces, face boxes, cropped face image
    """
    img_size = np.asarray(img.shape)[0:2]

    bounding_boxes, _ = align.detect_face.detect_face(img=img,
                                                      minsize=20 * face_size,
                                                      pnet=pnet, rnet=rnet, onet=onet,
                                                      threshold=[0.6, 0.7, 0.7],
                                                      factor=0.709)

    if len(bounding_boxes) < 1:
        return False, np.zeros((0, 0)), []

    det = bounding_boxes
    det[:, 0] = np.maximum(det[:, 0]-MARGIN/2, 0)
    det[:, 1] = np.maximum(det[:, 1]-MARGIN/2, 0)
    det[:, 2] = np.minimum(det[:, 2]+MARGIN/2, img_size[1]-1)
    det[:, 3] = np.minimum(det[:, 3]+MARGIN/2, img_size[0]-1)
    det = det.astype(int)

    # Crop standard-size face images and pre-whiten them
    crop = []
    for i in range(len(bounding_boxes)):
        temp_crop = img[det[i, 1]:det[i, 3], det[i, 0]:det[i, 2], :]
        aligned = cv2.resize(temp_crop, (INPUT_SIZE, INPUT_SIZE))
        prewhitened = facenet.prewhiten(aligned)
        crop.append(prewhitened)
    crop_image = np.stack(crop)

    return True, det, crop_image


def realtime_recogniniton(group_name, face_size=1, track_interval=200, recognition_interval=1000, scale_factor=1, tolerance=0.8):
    """
    Run realtime face recognition.
    :param group_name: Name of user group
    :param face_size: Minimum detected face size
    :param track_interval: Face detect interval/ms
    :param recognition_interval: Face recognize interval/ms
    :param scale_factor: Image processing zoom factor
    :param tolerance: Face recognition threshold
    :return: None
    """
    with tf.Graph().as_default():
        with tf.Session() as sess:     
            # Load FaceNet
            facenet.load_model(MODEL_DIR)

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Load face database
            db_dir = DB_ROOT_DIR + group_name
            known_face_names = []  # Names in database
            image_list = []
            for face in os.listdir(db_dir):
                known_face_names.append(face.replace(".", "_").split("_")[0])
                img = cv2.imread(os.path.join(db_dir, face))
                prewhitened = facenet.prewhiten(img)
                image_list.append(prewhitened)

            known_face_images = np.stack(image_list)  # Face images in database
            num_users = known_face_images.shape[0]  # User number in database
            compare_emb = np.zeros((num_users, 512))  # Features in database

            # Encode faces into 512-dimensional features
            for i in range(num_users):
                sys.stdout.write("\r Loading Data: %.2f%% " % (i / num_users * 100))
                sys.stdout.flush()
                feed_dict = {images_placeholder: known_face_images[i:i + 1, :, :, :], phase_train_placeholder: False}
                compare_emb[i, :] = sess.run(embeddings, feed_dict=feed_dict)
            sys.stdout.write('\n')
            sys.stdout.flush()

            # Clear memory
            del image_list
            del known_face_images
            gc.collect()

            face_flag = False
            bounding_box = np.zeros((0, 0))  # Container for detected face boxes
            crop_image = np.zeros((0, 0))  # Container for cropped face images
            face_names = []  # Container for recognized face names

            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            ret, frame = cap.read()

            timer = 0  # Frame skip timer
            while ret:
                timer += 1
                ret, frame = cap.read()
                small_frame = cv2.resize(frame, (0, 0), fx=1 / scale_factor, fy=1 / scale_factor)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                # Face detection
                if timer % (track_interval * FPS // 1000) == 0:
                    face_flag, bounding_box, crop_image = load_and_align_data(rgb_small_frame, face_size)

                # Face recognition
                if timer % (recognition_interval * FPS // 1000) == 0 and face_flag:
                    feed_dict = {images_placeholder: crop_image, phase_train_placeholder: False}
                    emb = sess.run(embeddings, feed_dict=feed_dict)

                    face_names.clear()

                    for i in range(len(emb)):
                        face_distances = np.sqrt(np.sum(np.square(emb[i, :] - compare_emb[:, :]), axis=1))

                        name = known_face_names[int(np.argmin(face_distances))] if np.min(face_distances) < tolerance else 'Unknown'
                        face_names.append(name)

                # Draw face boxes and names
                for face in range(bounding_box.shape[0]):

                    top = bounding_box[face, 1] * scale_factor
                    right = bounding_box[face, 2] * scale_factor
                    bottom = bounding_box[face, 3] * scale_factor
                    left = bounding_box[face, 0] * scale_factor

                    if len(face_names) > face:
                        name = face_names[face]
                    else:
                        name = 'Unknown'

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)
                    cv2.rectangle(frame, (left, bottom), (right, int(bottom + (bottom - top) * 0.25)), (0, 0, 255), cv2.FILLED)
                    cv2.putText(frame, name, (left, int(bottom + (bottom - top) * 0.24)),
                                cv2.FONT_HERSHEY_DUPLEX, (right - left) / 120, (255, 255, 255), 1)

                cv2.imshow('camera', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyWindow("camera")


if __name__ == '__main__':
    realtime_recogniniton(group_name="test_group", face_size=1, track_interval=200, recognition_interval=1000, scale_factor=1, tolerance=0.8)
