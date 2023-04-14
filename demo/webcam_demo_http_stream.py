# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import time
from collections import deque
from operator import itemgetter
from threading import Thread

import datetime
from queue import Queue
from http_stream import start_socket_send

import cv2
import numpy as np
import torch
from mmengine import Config, DictAction
from mmengine.dataset import Compose, pseudo_collate

from mmaction.apis import init_recognizer

FONTFACE = cv2.FONT_HERSHEY_SIMPLEX
FONTSCALE = 1.5
FONTCOLOR = (0, 0, 255)  # BGR, white
MSGCOLOR = (128, 128, 128)  # BGR, gray
THICKNESS = 3
LINETYPE = 1
EXCLUED_STEPS = [
    'OpenCVInit', 'OpenCVDecode', 'DecordInit', 'DecordDecode', 'PyAVInit',
    'PyAVDecode', 'RawFrameDecode'
]


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 webcam demo')
    parser.add_argument('--config', type=str, help='test config file path')
    parser.add_argument('--checkpoint', type=str, help='checkpoint file/url')
    parser.add_argument('--label', type=str, help='label file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--video', type=str, default='', help='video file/url')
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='recognition score threshold')
    parser.add_argument(
        '--average_size',
        type=int,
        default=1,
        help='number of latest clips to be averaged for prediction')
    parser.add_argument(
        '--cfg_options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--port',
        type=int
    )
    parser.add_argument(
        '--interval', default=1,
        type=int
    )
    args = parser.parse_args()

    return args


def show_results():
    # print('Press "Esc", "q" or "Q" to exit')
    global frame_id
    text_info = {}
    objects = []

    while True:
        msg = 'Waiting for action ...'
        _, frame = camera.read()
        frame_id += 1
        if frame_id % interval == 0:
            frame_queue.append(np.array(frame[:, :, ::-1]))

        if len(result_queue) != 0:
            text_info = {}
            results = result_queue.popleft()
            for i, result in enumerate(results):
                selected_label, score = result
                if score < threshold:
                    break
                location = (0, 100 + i * 20)
                text = selected_label + ': ' + str(round(score * 100, 2))
                text_info[location] = text
                cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                            FONTCOLOR, THICKNESS, LINETYPE)
                objects = [{"class_id": label.index(selected_label), "name": selected_label, "confidence": float(score)}]
                break

        elif len(text_info) != 0:
            for location, text in text_info.items():
                cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                            FONTCOLOR, THICKNESS, LINETYPE)

        else:
            cv2.putText(frame, msg, (0, 100), FONTFACE, FONTSCALE, MSGCOLOR,
                        THICKNESS, LINETYPE)

        json = {"frame_id": frame_id,
                "time": datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d %H:%M:%S"),
                "objects": objects
                }
        q_url.put([json, frame])

def inference():
    score_cache = deque()
    scores_sum = 0

    while True:
        cur_windows = []

        while len(cur_windows) == 0:
            if len(frame_queue) == sample_length:
                cur_windows = list(np.array(frame_queue))
                if data['img_shape'] is None:
                    data['img_shape'] = frame_queue.popleft().shape[:2]
                frame_queue.clear()

        cur_data = data.copy()
        cur_data['imgs'] = cur_windows
        cur_data = test_pipeline(cur_data)
        cur_data = pseudo_collate([cur_data])

        # Forward the model
        with torch.no_grad():
            result = model.test_step(cur_data)[0]
        scores = result.pred_scores.item.tolist()
        scores = np.array(scores)
        score_cache.append(scores)
        scores_sum += scores

        if len(score_cache) == average_size:
            scores_avg = scores_sum / average_size
            num_selected_labels = min(len(label), 5)

            score_tuples = tuple(zip(label, scores_avg))
            score_sorted = sorted(
                score_tuples, key=itemgetter(1), reverse=True)
            results = score_sorted[:num_selected_labels]

            result_queue.append(results)
            scores_sum -= score_cache.popleft()


def main():
    global average_size, threshold, interval, \
        device, model, camera, data, label, sample_length, \
        test_pipeline, frame_queue, result_queue, q_url, frame_id

    frame_id = 0
    args = parse_args()
    average_size = args.average_size
    threshold = args.threshold
    interval = args.interval

    device = torch.device(args.device)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Build the recognizer from a config file and checkpoint file/url
    model = init_recognizer(cfg, args.checkpoint, device=args.device)
    camera = cv2.VideoCapture(args.video)
    data = dict(img_shape=None, modality='RGB', label=-1)

    with open(args.label, 'r') as f:
        label = [line.strip() for line in f]
    print(f'label: {label}')

    # prepare test pipeline from non-camera pipeline
    cfg = model.cfg
    sample_length = 0
    pipeline = cfg.test_pipeline
    pipeline_ = pipeline.copy()
    for step in pipeline:
        if 'SampleFrames' in step['type']:
            sample_length = step['clip_len'] * step['num_clips']
            data['num_clips'] = step['num_clips']
            data['clip_len'] = step['clip_len']
            pipeline_.remove(step)
        if step['type'] in EXCLUED_STEPS:
            # remove step to decode frames
            pipeline_.remove(step)
    test_pipeline = Compose(pipeline_)

    assert sample_length > 0

    frame_queue = deque(maxlen=sample_length)
    result_queue = deque(maxlen=1)
    
    q_url = Queue()
    start_socket_send(q_url, args.port)
    pw = Thread(target=show_results, args=(), daemon=True)
    
    pr = Thread(target=inference, args=(), daemon=True)
    
    pw.start()
    
    pr.start()
    pw.join()



if __name__ == '__main__':
    main()
