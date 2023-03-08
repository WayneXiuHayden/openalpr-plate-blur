# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import json
import os

from openalpr import Alpr, VehicleClassifier
import sys
import cv2
import numpy as np
from alprstream import AlprStream
import time

# license_key = "SEBKS0xNTkawsbKztLW2sbG8u7i0ura6oKmlpKWjoK+pqK6jrquvqZaZk5KQnZaXAXG620gq4lc9L8QiuNpT8WycadotjMC7NUpJls6reDtyFoBZTCSTrdDGjsQ6dVNHDdf/2c7zMUioI8hXUYJ/F5Nl4uFGJk/h4A9toJX+LZVQZuESps1On1EGt+uvCdaE"
license_key = "SExKS0xOTk+wsbKztLW2t7i5usPN1MrK3s7T2qWjr6GuqqyspaSvqZmXlJCSkp+eATPWXgYuCGHlxS4ea5eQZYbXs4aDy7d7Um0dpAwH8MXg+sQGKkLa9r5FLQ9J8sMJ79zb7nizMwvgFIgO7jQh8uyCpgzCp8m7iFmknofrIh10Eq+IWBtwzYBj0MENz34D"
country = "us"
alpr_conf = "/etc/openalpr/openalpr.conf"
alpr_runtime = "/usr/share/openalpr/runtime_data"

TEST_VIDEO_FILE_PATH = 'video_sample.mp4'


def anonymize_face_simple(image, factor=3.0):
    # automatically determine the size of the blurring kernel based
    # on the spatial dimensions of the input image
    (h, w) = image.shape[:2]
    kW = int(w / factor)
    kH = int(h / factor)
    # ensure the width of the kernel is odd
    if kW % 2 == 0:
        kW -= 1
    # ensure the height of the kernel is odd
    if kH % 2 == 0:
        kH -= 1
    # apply a Gaussian blur to the input image using our computed
    # kernel size
    if kW <= 1:
        kW = 1
    if kH <= 1:
        kH = 1
    return cv2.GaussianBlur(image, (kW, kH), 0)


def put_text(image, text, orgin):
    image = cv2.putText(image, text, org=orgin, fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 255, 0),
                        thickness=3)
    return image


def anonymize_face_pixelate(image, blocks=3):
    # divide the input image into NxN blocks
    (h, w) = image.shape[:2]
    xSteps = np.linspace(0, w, blocks + 1, dtype="int")
    ySteps = np.linspace(0, h, blocks + 1, dtype="int")
    # loop over the blocks in both the x and y direction
    for i in range(1, len(ySteps)):
        for j in range(1, len(xSteps)):
            # compute the starting and ending (x, y)-coordinates
            # for the current block
            startX = xSteps[j - 1]
            startY = ySteps[i - 1]
            endX = xSteps[j]
            endY = ySteps[i]
            # extract the ROI using NumPy array slicing, compute the
            # mean of the ROI, and then draw a rectangle with the
            # mean RGB values over the ROI in the original image
            roi = image[startY:endY, startX:endX]
            (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
            cv2.rectangle(image, (startX, startY), (endX, endY),
                          (B, G, R), -1)
    # return the pixelated blurred image
    return image


def static_image_test():
    image_file = "plate-2.jpg"
    alpr = Alpr(country, alpr_conf, alpr_runtime, license_key)

    if not alpr.is_loaded():
        print("Error loading OpenALPR")
        sys.exit(1)

    # optional detection parameters
    alpr.set_top_n(5)
    # alpr.set_default_region("md")

    results = alpr.recognize_file(image_file)
    image = cv2.imread(image_file)
    orig = image.copy()

    # print(json.dumps(results, indent=4))
    for i, plate in enumerate(results["results"]):
        print('Plate {:-<30}'.format(i))
        coordinates = plate["coordinates"]
        # print(coordinates[0].x)
        startX = min(coordinates[0]['x'], coordinates[3]['x'])
        startY = min(coordinates[0]['y'], coordinates[1]['y'])
        endX = max(coordinates[1]['x'], coordinates[2]['x'])
        endY = max(coordinates[2]['y'], coordinates[3]['y'])

        plate_roi = image[startY:endY, startX:endX]

        # for c in plate["coordinates"]:
        #     print(c)
        # display = '\t{:>7} {}'.format('{:.2f}%'.format(c['confidence']), c['plate'])
        # if c['matches_template']:
        #     display += ' *'
        # print(display)

        # simple
        plate_roi = anonymize_face_simple(plate_roi, factor=3.0)
        image[startY:endY, startX:endX] = plate_roi

        # output = np.hstack([orig, image])
        # cv2.imwrite("Blurred" + str(i) + ".jpg", output)
    output = image
    image_name = image_file.split('.')[0]
    image_extension = image_file.split('.')[-1]
    cv2.imwrite(image_name + "_blurred" + '.' + image_extension, output)
    alpr.unload()


def video_stream_test(video_file=None):
    if video_file is None:
        video_file = TEST_VIDEO_FILE_PATH
    # GPU processing must use a single Alpr object and single thread per GPU.
    # CPU processing may use multiple threads, with one Alpr object/thread per CPU core.
    USE_GPU = False
    GPU_BATCH_SIZE = 10
    GPU_ID = 0

    TRACK_VEHICLES_WITHOUT_PLATES = True
    alpr = Alpr(country, alpr_conf, alpr_runtime, license_key, use_gpu=USE_GPU, gpu_id=GPU_ID,
                gpu_batch_size=GPU_BATCH_SIZE)

    if not alpr.is_loaded():
        print('Error loading Alpr')
        sys.exit(1)
    alpr_stream = AlprStream(frame_queue_size=GPU_BATCH_SIZE, use_motion_detection=True)
    if not alpr_stream.is_loaded():
        print('Error loading AlprStream')
        sys.exit(1)
    vehicle = VehicleClassifier(alpr_conf, alpr_runtime, license_key)
    if not vehicle.is_loaded():
        print('Error loading VehicleClassifier')
        sys.exit(1)

    alpr.set_detect_vehicles(True, TRACK_VEHICLES_WITHOUT_PLATES)

    # Speeds up GPU by copying video data to GPU memory while processing
    if USE_GPU:
        alpr_stream.set_gpu_async(GPU_ID)

    # Connect to stream/video and process results
    alpr_stream.connect_video_file(video_file, 0)
    start = time.time()
    frame_count = 0

    frame_record = {}
    plate_record = {}
    candidate_plates = ['G60RVD', 'N27MKN']  # only show these plates; if empty, show all
    while alpr_stream.video_file_active() or alpr_stream.get_queue_size() > 0:
        if alpr.use_gpu:
            single_frame = alpr_stream.process_batch(alpr)
        else:
            single_frame = alpr_stream.process_frame(alpr)
        if single_frame is None:
            pass
        else:
            if single_frame['results']:
                frame_record[frame_count] = []
                plate_record[frame_count] = []
                for i, plate in enumerate(single_frame["results"]):
                    coordinates = plate['coordinates']
                    start_x = min(coordinates[0]['x'], coordinates[3]['x'])
                    start_y = min(coordinates[0]['y'], coordinates[1]['y'])
                    end_x = max(coordinates[1]['x'], coordinates[2]['x'])
                    end_y = max(coordinates[2]['y'], coordinates[3]['y'])

                    plate_number = plate['plate']
                    # check if plate number is in candidate_plates
                    if candidate_plates and plate_number not in candidate_plates:
                        continue
                    frame_record[frame_count].append((start_x, start_y, end_x, end_y))
                    plate_record[frame_count].append(plate_number)
                print(frame_count, ": ", single_frame['results'])
            frame_count += 1
            # print(frame_count)
        # active_groups = len(alpr_stream.peek_active_groups())
        # # print('Active groups: {:<3} \tQueue size: {}'.format(active_groups, alpr_stream.get_queue_size()))
        # groups = alpr_stream.pop_completed_groups_and_recognize_vehicle(vehicle, alpr)
        #
        # for group in groups:
        #     print('=' * 40)
        #     print('Group from frames {}-{}'.format(group['frame_start'], group['frame_end']))
        #
        #     if group['data_type'] == 'alpr_group':
        #         print('Plate: {} ({:.2f}%)'.format(group['best_plate']['plate'], group['best_plate']['confidence']))

        # if group['data_type'] == 'vehicle' or group['data_type'] == 'alpr_group':
        #     print('Vehicle attributes')
        #     for attribute, candidates in group['vehicle'].items():
        #         print('\t{}: {} ({:.2f}%)'.format(attribute.capitalize(), candidates[0]['name'],
        #                                           candidates[0]['confidence']))
        #     print('=' * 40)

    # Call when completely done to release memory
    end_alpr_process = time.time()

    alpr.unload()
    vehicle.unload()

    print("Alpr Execution time: ", end_alpr_process - start, " seconds")
    print("frame record: ", frame_record)

    cap = cv2.VideoCapture(video_file)
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # print("fps 1: ", fps)
    if not cap.isOpened():
        print("Error opening video stream or file")

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(5)
    print("fps: ", fps)
    frame_count_cv = cap.get(7)
    print("frame_count_cv: ", frame_count_cv)
    print("frame_count: ", frame_count)
    frame_offset = frame_count - frame_count_cv
    print("frame_offset: ", frame_offset)
    frame_ratio = frame_count / frame_count_cv
    print("frame_ratio: ", frame_ratio)

    video_name = video_file.split('.')[0]
    video_extension = video_file.split('.')[-1]
    # out = cv2.VideoWriter(video_name + "_blurred" + '.' + video_extension, cv2.VideoWriter_fourcc(*'XVID'),
    #                       # 'M', 'J', 'P', 'G'
    #                       fps, (frame_width, frame_height))
    out = cv2.VideoWriter(video_name + "_blurred" + '.' + video_extension, cv2.VideoWriter_fourcc(*'X264'),
                          fps, (frame_width, frame_height))
    frame_counting = 1

    start = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # counting = int(frame_counting*frame_ratio) # frame_counting + frame_offset
            counting = frame_counting + frame_offset
            if counting in frame_record:
                for coordinate in frame_record[counting]:
                    start_x = coordinate[0]
                    start_y = coordinate[1]
                    end_x = coordinate[2]
                    end_y = coordinate[3]
                    face = frame[start_y:end_y, start_x:end_x]
                    # blurring the face area
                    # face = anonymize_face_simple(face, factor=5.0)
                    # frame[start_y:end_y, start_x:end_x] = face
                    # adding plate number on top
                    # face = put_text(face, plate_record[frame_count_cv + frame_offset][0])
                    # frame[start_y:end_y, start_x:end_x] = face
                    frame = put_text(frame, plate_record[counting][0], (start_x, start_y))
            frame_counting += 1
            # print(frame_counting)
            out.write(frame)
        else:
            break
    cap.release()
    out.release()

    end_cv_process = time.time()
    print("OpenCV Execution time: ", end_cv_process - start, " seconds")


def plate_blur():
    # directory_path = "face_blurred"
    directory_path = "NYCDemo-03082023-Long/Input"
    for filename in os.listdir(directory_path):
        print("Processing: ", filename)
        video_stream_test(os.path.join(directory_path, filename))


if __name__ == '__main__':
    # static_image_test()
    # video_stream_test()
    plate_blur()
