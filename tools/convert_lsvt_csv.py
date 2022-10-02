# *************************************************************************
# Copyright (2022) Bytedance Inc.
#
# Copyright (2022) oCLIP Authors 
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 
#
#     http://www.apache.org/licenses/LICENSE-2.0 
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
# *************************************************************************


import os.path as osp
import json
import pickle
from tqdm import tqdm
import os
import cv2
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Prepare LSVT Annotation for oCLIP Pre-training")
    parser.add_argument("--data_dir", type=str, default='./data/LSVT')
    parser.add_argument("--save_dir", type=str, default='./data/LSVT')
    args = parser.parse_args()

    with open(os.path.join(args.data_dir, 'train_weak_labels.json'), 'r') as f:
        data = json.load(f)

    i_range = range(len(data.keys()))
    print('num_images: %d'%len(i_range))
    char_list = []
    data_out = ['filepath\ttitle\n']

    for key in tqdm(data.keys()):
        i_name = '{}/train_weak_images/{}.jpg'.format(args.data_dir, key)
        if not os.path.exists(i_name):
            continue
        im = cv2.imread(i_name)
        try:
            s = im.shape
        except:
            continue
        i_txt = []
        for i in range(len(data[key])):
            txt = data[key][i]['transcription'].replace(' ', '')
            if len(txt) < 2:
                continue
            i_txt.append(txt)
            for char in txt:
                if ord(char) not in char_list:
                    char_list.append(ord(char))
        if len(i_txt) == 0:
            continue

        data_out += ['{}\t{}\n'.format(i_name, ' '.join(i_txt))]

    with open(osp.join(args.save_dir, 'train_char.csv'), 'w') as f:
        f.writelines(data_out)

    with open(osp.join(args.save_dir, 'char_dict'), 'wb') as f:
        pickle.dump(char_list, f)
        