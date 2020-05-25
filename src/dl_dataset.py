import csv
import youtube_dl
import cv2
import json
import numpy as np
import os
# import moviepy
# from moviepy.editor import *


def read_csv(filename):
    try:
        with open(filename, mode='r') as csvfile:
            csv_reader = csv.reader(csvfile)
            info = []
            for row in csv_reader:
                info.append(row[:])
            csvfile.close()
            return info
    except IOError:
        print('Problem reading: ' + filename)


def read_json(filename):
    with open(filename, 'r') as openfile:
        json_object = json.load(openfile)
    return json_object


def read_label(id_list):
    label_list = []
    for id in id_list:
        json_object = read_json('dataset/' + str(id) + '.json')
        label_list.append([float(x['label']) for x in json_object['points']])
    return np.array(label_list)


def write_label(filename, label):
    info = {'points': []}
    for t in range(len(label)):
        info['points'].append({
            'time': str(t),
            'label': str(label[t])
        })
    json_object = json.dumps(info, indent=2)
    with open(str(filename), 'w') as outfile:
        outfile.write(json_object)


class load_dataset(object):
    def __init__(self, filename):
        datalist = read_csv(filename)
        self.id = [int(x[0]) for x in datalist[1:]]
        self.youtubeid = [x[1][-11:] for x in datalist[1:]]
        self.cuttime = [[x[2], x[3]] for x in datalist[1:]]
        self.actiontime = [[x[4], x[5]] for x in datalist[1:]]
        self.fps = 30
        self.size = (1280, 720)
        for id in self.id:
            # compute the clip start frame and the clip stop frame
            cstart = self.cuttime[id][0].split(':')
            self.cuttime[id][0] = self.fps * (int(cstart[0]) * 60 + int(cstart[1]))
            cstop = self.cuttime[id][1].split(':')
            self.cuttime[id][1] = self.fps * (int(cstop[0]) * 60 + int(cstop[1]))
            # compute the action start frame and action clip stop frame
            astart = self.actiontime[id][0].split(':')
            self.actiontime[id][0] = self.fps * (int(astart[0]) * 60 + int(astart[1]))
            astop = self.actiontime[id][1].split(':')
            self.actiontime[id][1] = self.fps * (int(astop[0]) * 60 + int(astop[1]))

    def read_keypoints(self, id_list):
        keypoints_list = []
        # read each clips' keypoints
        for id in id_list:
            frames = []
            total_frames = self.cuttime[id][1] - self.cuttime[id][0]

            # read keypoints by frames
            for i in range(total_frames):
                filename = str(id) + '_000000000000'[:-len(str(i))] + str(i) + '_keypoints.json'
                try:
                    json_object = read_json('dataset/' + str(id) + '/' + filename)
                    frames.append(json_object['people'][0]['pose_keypoints_2d'])
                except Exception:
                    frames.append([0 for x in range(75)])
            keypoints_list.append(frames)
        return keypoints_list

    def extract_keypoints(self, keypoints_list):
        coordinate_list = []
        for frames in keypoints_list:
            coordinate = []
            # extract coordinate information from pose_keypoints_2d by each frame
            for keypoints in frames:
                coordinate.append([[keypoints[x], keypoints[x + 1]] for x in range(0, len(keypoints), 3)])
            n_fea = len(coordinate[0])
            # reshape to the input format
            coordinate_list.append(np.array(coordinate).reshape((len(frames), 1, n_fea, 2)))
        return np.array(coordinate_list)

    def train_test_split(self, train_set_id, test_set_id):
        if type(train_set_id) is not list:
            train_set_id = [train_set_id]
        if type(test_set_id) is not list:
            test_set_id = [test_set_id]

        # read the label information and keypoints information and assign them to the train set and test set
        x_train = self.read_keypoints(train_set_id)
        x_train = self.extract_keypoints(x_train)
        y_train = read_label(train_set_id)
        x_test = self.read_keypoints(test_set_id)
        x_test = self.extract_keypoints(x_test)
        y_test = read_label(test_set_id)
        return x_train, y_train, x_test, y_test

    # download youtube videos through youtube id
    def download(self, youtubeid):
        url = 'https://www.youtube.com/watch?v=' + str(youtubeid)
        ydl_opts = {
            'format': 'mp4',
            'outtmpl': 'dataset/download_video/%(id)s.%(ext)s',
            'noplaylist': True,
            'continue_dl': True
        }
        try:
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.cache.remove()
                info_dict = ydl.extract_info(url, download=False)
                ydl.prepare_filename(info_dict)
                ydl.download([url])
                return True
        except Exception:
            return False

    # checking whether target file exists
    def find(self, name, path):
        for root, dirs, files in os.walk(path):
            if name in files:
                return os.path.join(root, name)

    # # to capture clips, moviepy need to be installed

    # # capture clips from youtube videos
    # def capture(self, id, youtubeid, cuttime):
    #     # omit if clips already exist
    #     if self.find(str(id) + '.avi', 'dataset/') is None:
    #         # download videos if not exist
    #         if self.find(str(youtubeid) + '.mp4', 'dataset/download_video/') is None:
    #             self.download(youtubeid)
    #
    #         # capture clips
    #         videoCapture = cv2.VideoCapture('dataset/download_video/' + str(youtubeid)+'.mp4')
    #         fps = self.fps
    #         size = self.size
    #         four_cc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    #         videoWriter = cv2.VideoWriter('dataset/' + str(id)+'.avi', four_cc, fps, size)
    #         fps = videoCapture.get(5)
    #
    #         # if fps and size don't match to the settings
    #         if ((fps < 29.97) and (fps > 30)) or ((videoCapture.get(3), videoCapture.get(4)) != (1280, 720)):
    #             if self.find(str(youtubeid)+'_30.mp4', 'dataset/download_video/') is None:
    #                 clip = VideoFileClip('dataset/download_video/' + str(youtubeid)+'.mp4')
    #                 clip = moviepy.video.fx.all.resize(clip, (1280, 720))
    #                 clip.write_videofile('dataset/download_video/' + str(youtubeid)+'_30.mp4', fps=30)
    #             videoCapture = cv2.VideoCapture('dataset/download_video/' + str(youtubeid)+'_30.mp4')
    #
    #         # set up the start frame and the stop frame
    #         start = cuttime[0]
    #         stop = cuttime[1]
    #         i = 0
    #         print('cut start:', id, cuttime)
    #
    #         # begin capturing
    #         while True:
    #             success, frame = videoCapture.read()
    #             if success:
    #                 i += 1
    #                 if (i >= start and i <= stop):
    #                     videoWriter.write(frame)
    #                 elif i > stop:
    #                     print('cut end')
    #                     break
    #             else:
    #                 print('cut end')
    #                 break
    #         videoCapture.release()

    # transform clips to images by frames
    def save_img(self, id):
        videoCapture = cv2.VideoCapture('dataset/' + str(id) + '.avi')
        i = 0
        print('saveimg start:', id)
        while True:
            success, frame = videoCapture.read()
            if success:
                filename = str(id) + '_000000000000'[:-len(str(i))] + str(i) + '.jpg'
                cv2.imwrite('dataset/' + str(id) + '/' + filename, frame)
                i += 1
            else:
                print('saveimg end')
                break
        videoCapture.release()

    # create label json file for each clip
    def create_label_json(self):
        for id in range(len(self.actiontime)):
            # clips start and stop time
            cstart = self.cuttime[id][0]
            cstop = self.cuttime[id][1]
            # actions start and stop time
            astart = self.actiontime[id][0]
            astop = self.actiontime[id][1]
            # if the action occurs in the clip
            if (astart != 0) or (astop != 0):
                astart = astart - cstart if (astart - cstart) >= 0 else 0
                astop = astop - cstart if astop <= cstop else cstop - cstart
            # mark the label by frames
            label = [0 for x in range(cstop - cstart)]
            label[astart:astop] = [1 for x in range(astop - astart)]
            # write labels to the json file
            write_label('dataset/' + str(id) + '.json', label)

    # def main(self):
    #     data = load_dataset('rolling.csv')
    #     for i in data.id:
    #         data.capture(i, data.youtubeid[i], data.cuttime[i])
    #         data.save_img(i)
    #     data.create_label_json()
