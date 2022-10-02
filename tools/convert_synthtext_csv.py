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

import os
import scipy.io, scipy.ndimage
import random
import pickle
from tqdm import tqdm
import argparse



if __name__ == "__main__":

    parser = argparse.ArgumentParser("Prepare SynthText Annotation for oCLIP Pre-training")
    parser.add_argument("--data_dir", type=str, default='./data/SynthText')
    parser.add_argument("--save_dir", type=str, default='./data/SynthText')
    args = parser.parse_args()

    synth_dat = scipy.io.loadmat(os.path.join(args.data_dir, 'gt.mat'))
    imnames = synth_dat['imnames'][0]
    charBB = synth_dat['charBB'][0]
    txt = synth_dat['txt'][0]

    if not os.path.isdir(os.path.join(args.save_dir)):
        os.makedirs(os.path.join(args.save_dir))

    i_range = range(len(txt))
    print('num_images: %d'%len(i_range))

    data_out = ['filepath\ttitle\n']
    for i,im_idx in tqdm(enumerate(i_range)):
        i_name = os.path.join(args.data_dir, 'all_images', str(imnames[im_idx][0]))
        i_txt = txt[im_idx]
        
        word_list = '\n'.join(i_txt)
        word_list = word_list.split()
        random.shuffle(word_list)

        data_out += ['{}\t{}\n'.format(i_name, ' '.join(word_list))]


    with open(os.path.join(args.save_dir, 'train_char.csv'), 'w') as f:
        f.writelines(data_out)

    char_list = list(range(33, 127))
    with open(os.path.join(args.save_dir, 'char_dict'), 'wb') as f:
        pickle.dump(char_list, f)
