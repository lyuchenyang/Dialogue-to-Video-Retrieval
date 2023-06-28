from tqdm import tqdm
import json
import codecs
import requests
import pandas as pd
from transformers import BertTokenizer, AutoTokenizer
from os import listdir
from os.path import isfile, join
import torch
import numpy as np
import random

json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)


def draw_samples(lis, ratio):
    samples = ratio if ratio > 1 else int(ratio * len(lis))

    if samples > len(lis):
        new_lis = np.random.choice(len(lis), samples, replace=True)
    else:
        new_lis = np.random.choice(len(lis), samples, replace=False)

    n_lis = [lis[i] for i in new_lis]

    return n_lis


def inspect_avsd():
    dir = 'data/avsd/avsd_val.json'

    js = json_load(dir)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def measure_avg_len(examples, key):
        lens = 0
        overlong = 0
        for e in examples:
            e = examples[e]
            if e[key] is None or len(e[key]) == 0:
                continue
            te = tokenizer.tokenize(e[key])
            if len(te) >= 60:
                overlong += 1
            lens += len(te)
        print(overlong)
        return lens / len(examples)

    avg_len_sum = measure_avg_len(js, 'summary')
    avg_len_script = measure_avg_len(js, 'script')
    return


def extract_audio_from_video():
    import moviepy.editor as mp

    path = 'data/avsd/videos/'
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

    for f in tqdm(onlyfiles):
        dir = path + f
        clip = mp.VideoFileClip(dir)
        clip.audio.write_audiofile('data/avsd/audios/{}.wav'.format(f.split('.')[0]))
    return


def sample_frames_from_video():
    # Importing all necessary libraries
    import cv2

    path = 'data/avsd/videos/'
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

    frames_per_video = 60
    for f in tqdm(onlyfiles):
        # Read the video from specified path
        cam = cv2.VideoCapture(path + f)

        # frame
        currentframe = 0
        all_frames = []
        while (True):

            # reading from frame
            ret, frame = cam.read()

            if ret:
                all_frames.append(frame)
                currentframe += 1
            else:
                break
        lens = len(all_frames)
        if lens >= frames_per_video:
            interval = lens // frames_per_video

            frame_ind = [i * interval for i in range(frames_per_video)]
            for i in range(len(frame_ind)):
                if frame_ind[i] >= lens:
                    frame_ind[i] = lens - 1
            frame_ind[-1] = lens - 1
            sampled_frames = [all_frames[i] for i in frame_ind]
        else:
            sampled_frames = sorted(draw_samples([i for i in range(len(all_frames))], frames_per_video))
            sampled_frames = [all_frames[i] for i in sampled_frames]

        for ind, frame in enumerate(sampled_frames):
            cv2.imwrite('data/avsd/frames/{}_{}.jpg'.format(f.split('.')[0], str(ind)), frame)

        # Release all space and windows once done
        cam.release()
        cv2.destroyAllWindows()


def preprocess_avsd_to_tensor_dataset():
    import clip
    import torch
    from transformers import AutoTokenizer, AutoFeatureExtractor

    import pickle

    image_dir = 'data/avsd/frames/'
    audio_dir = 'data/avsd/audios/'

    train_metadata_dir = 'data/avsd/avsd_train.json'
    val_metadata_dir = 'data/avsd/avsd_val.json'
    test_metadata_dir = 'data/avsd/avsd_test.json'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14@336px", device=device)

    torch.random.manual_seed(0)

    def read_image_and_audio(metadata_dir, split='train'):
        metadata = json_load(metadata_dir)

        all_video_names = []
        all_images, all_audios, all_summaries, all_scripts, all_dialogs, all_ans_in_dialog = [], [], [], [], [], []
        for ind, key in enumerate(tqdm(metadata)):
            md = metadata[key]
            all_video_names.append(key)
            '''
            Abandoned due to significant use of memory
            '''
            # all_frames = []
            # frame_index = sorted(draw_samples([i for i in range(20)], 10))
            # for ind in frame_index:
            #     frame = preprocess(Image.open('{}{}_{}.jpg'.format(image_dir, key, str(ind))))
            #     all_frames.append(frame)
            # all_frames = torch.cat(all_frames, dim=0)
            all_frames = torch.tensor([ind], dtype=torch.int)

            summary = md['summary'] if md['summary'] is not None else md['script']
            script = md['script']

            t_summary = clip.tokenize(summary, context_length=77, truncate=True)
            t_script = clip.tokenize(script, context_length=77, truncate=True)

            all_t_q = []
            for dialog in md['data']:
                q = dialog['question'] + ' ' + dialog['answer']
                t_q = clip.tokenize(q, context_length=77, truncate=True)
                all_t_q.append(t_q)

            all_t_q = torch.cat(all_t_q, dim=0)

            all_t_ans = []
            for dialog in md['data']:
                ans = dialog['answer']
                t_ans = clip.tokenize(ans, context_length=77, truncate=True)
                all_t_ans.append(t_ans)

            all_t_ans = torch.cat(all_t_ans, dim=0)

            all_images.append(all_frames)
            all_audios.append(all_frames)
            all_summaries.append(t_summary)
            all_scripts.append(t_script)
            all_dialogs.append(all_t_q)
            all_ans_in_dialog.append(all_t_ans)

        pickle.dump(
            [all_images, all_audios, all_summaries, all_scripts, all_dialogs, all_ans_in_dialog],
            open('data/avsd/{}.cache'.format(split), "wb"), protocol=4)

        video_names = {'split': split, 'data': all_video_names}
        json_dump(video_names, 'data/avsd/{}_video_names.json'.format(split))

    read_image_and_audio(train_metadata_dir, split='train')
    read_image_and_audio(val_metadata_dir, split='val')
    read_image_and_audio(test_metadata_dir, split='test')


def process_dialogs():
    special_tokens = json_load('../dialog/additional_special_tokens.json')

    def filtering(line):
        for sp in special_tokens['additional_special_tokens'][:10]:
            line = line.replace(sp, '')
        return line

    def output_dialogs_by_task(task_key, split):
        data_dir = '../dialog/{}.csv'.format(split)
        if split == 'dev':
            split = 'val'
        df = pd.read_csv(data_dir)
        hist = list(df['history'])
        inputs = list(df['input'])
        target = list(df['target'])
        tasks = list(df['task'])

        source_lines, target_lines = [], []
        for h, inp, targ, task in zip(hist, inputs, target, tasks):
            if task == task_key:
                if str(h) == 'nan':
                    h = ''
                line = filtering(str(h) + ' ' + str(inp))

                targf = filtering(targ)
                if line.replace(' ', '') == '' or targf.replace(' ', '') == '':
                    continue

                source_lines.append(line)

                target_lines.append(str(targf))

        with open('../dialog/dialog-task/{}/{}.source'.format(task_key, split), 'w') as f:
            for line in source_lines:
                f.writelines(line.replace('\n', '').strip() + '\n')

        with open('../dialog/dialog-task/{}/{}.target'.format(task_key, split), 'w') as f:
            for line in target_lines:
                f.writelines(line.replace('\n', '').strip() + '\n')

    task_keys = ['NLU', 'DST', 'NLG']
    splits = ['train', 'dev', 'test']

    for tk in task_keys:
        for sp in splits:
            output_dialogs_by_task(tk, sp)


def sample_frames_from_video_for_val_set():
    # Importing all necessary libraries
    import cv2

    path = 'data/avsd/videos/Charades_v1/'
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

    video_names = json_load("data/avsd/val_video_names.json")['data']
    video_names = {vn: 0 for vn in video_names}

    frames_per_video = 60
    for ind, f in tqdm(enumerate(onlyfiles)):
        if ind <= 9722:
            continue
        vn = f.split('.')[0]
        if vn in video_names:
            continue
        # Read the video from specified path
        cam = cv2.VideoCapture(path + f)

        # frame
        currentframe = 0
        all_frames = []
        while True:
            # reading from frame
            ret, frame = cam.read()

            if ret:
                all_frames.append(frame)
                currentframe += 1
            else:
                break
        lens = len(all_frames)
        if lens >= frames_per_video:
            interval = lens // frames_per_video

            frame_ind = [i * interval for i in range(frames_per_video)]
            for i in range(len(frame_ind)):
                if frame_ind[i] >= lens:
                    frame_ind[i] = lens - 1
            sampled_frames = [all_frames[i] for i in frame_ind]
        else:
            sampled_frames = sorted(draw_samples([i for i in range(len(all_frames))], frames_per_video))
            sampled_frames = [all_frames[i] for i in sampled_frames]

        for ind, frame in enumerate(sampled_frames):
            cv2.imwrite('data/avsd/videos/frames/{}_{}.jpg'.format(f.split('.')[0], str(ind)), frame)

        # Release all space and windows once done
        cam.release()
        cv2.destroyAllWindows()


def retokenize_avsd_to_tensor_dataset():
    import clip
    import torch
    from transformers import AutoTokenizer

    import pickle

    image_dir = 'data/avsd/videos/frames/'
    audio_dir = 'data/avsd/videos/audios/'

    train_metadata_dir = 'data/avsd/avsd_train.json'
    val_metadata_dir = 'data/avsd/avsd_val.json'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14@336px", device=device)

    tokenizer = AutoTokenizer.from_pretrained('openai/clip-vit-base-patch16')

    def read_image_and_audio(metadata_dir, split='train'):
        metadata = json_load(metadata_dir)

        video_names = json_load('data/avsd/{}_video_names.json'.format(split))

        all_images, all_audios, all_summaries, all_scripts, all_dialogs, all_ans_in_dialog = [], [], [], [], [], []
        for ind, key in enumerate(tqdm(video_names)):
            md = metadata[key]
            '''
            Abandoned due to significant use of memory
            '''
            # all_frames = []
            # frame_index = sorted(draw_samples([i for i in range(20)], 10))
            # for ind in frame_index:
            #     frame = preprocess(Image.open('{}{}_{}.jpg'.format(image_dir, key, str(ind))))
            #     all_frames.append(frame)
            # all_frames = torch.cat(all_frames, dim=0)
            all_frames = torch.tensor([ind], dtype=torch.int)

            summary = md['summary'] if md['summary'] is not None else md['script']
            script = md['script']

            t_summary = clip.tokenize(summary, context_length=77, truncate=True)
            t_script = clip.tokenize(script, context_length=77, truncate=True)

            all_t_q = []
            for dialog in md['data']:
                q = dialog['question'] + ' ' + dialog['answer']
                t_q = clip.tokenize(q, context_length=77, truncate=True)
                all_t_q.append(t_q)

            all_t_q = torch.cat(all_t_q, dim=0)

            all_t_ans = []
            for dialog in md['data']:
                ans = dialog['answer']
                t_ans = clip.tokenize(ans, context_length=77, truncate=True)
                all_t_ans.append(t_ans)

            all_t_ans = torch.cat(all_t_ans, dim=0)

            all_images.append(all_frames)
            all_summaries.append(t_summary)
            all_scripts.append(t_script)
            all_dialogs.append(all_t_q)
            all_ans_in_dialog.append(all_t_ans)

        pickle.dump(
            [all_images, all_audios, all_summaries, all_scripts, all_dialogs, all_ans_in_dialog],
            open('data/avsd/{}.cache'.format(split), "wb"), protocol=4)

    read_image_and_audio(train_metadata_dir, split='train')
    read_image_and_audio(val_metadata_dir, split='val')


def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    import math
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                          -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


def resize_images():
    from PIL import Image

    path = 'data/avsd/videos/frames/'
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

    t = 0

    indices_t = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 51, 59]
    indices_e = [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 170, 199]
    indices = set(indices_e + indices_t)

    for f in tqdm(onlyfiles):
        ind = int(f.replace('.jpg', '').split('_')[1])
        if ind not in indices:
            continue
        image = Image.open(path + f)
        image.thumbnail((336, 336))
        image.save(path.replace('frames', 'frames_resize') + f)


if __name__ == '__main__':
    preprocess_avsd_to_tensor_dataset()
    sample_frames_from_video()
    extract_audio_from_video()
