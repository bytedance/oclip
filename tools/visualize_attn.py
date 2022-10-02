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

import sys
from PIL import Image
import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
from src.clip.model import oCLIP
import torchvision.transforms as T
import pickle
import json
import argparse

def load_model(model_path, model_info):

    state_dict = torch.load(model_path, map_location="cpu")
    state_dict = state_dict['state_dict']
    state_dict_ = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model = oCLIP(False, **model_info)
    model.eval()
    model.load_state_dict(state_dict_)
    return model


class tokenizer():
    def __init__(self, char_dict_pth):
        with open(char_dict_pth, 'rb') as f:
            self.letters = pickle.load(f)
            self.letters = [chr(x) for x in self.letters]

        self.p2idx = {p: idx+1 for idx, p in enumerate(self.letters)}
        self.idx2p = {idx+1: p for idx, p in enumerate(self.letters)}

        self.idx_mask = len(self.letters) + 1
        self.EOS = len(self.letters) + 2
        self.word_len = 25

    def tokenize(self, text):
        token = torch.zeros(self.word_len)
        for i in range(min(len(text), self.word_len)):
            if text[i] == ' ':
                token[i] = self.idx_mask
            else:
                token[i] = self.p2idx[text[i]]
        if len(text) >= self.word_len:
            token[-1] = self.EOS
        else:
            token[len(text)] = self.EOS

        return token

    def char_token(self, all_texts):
        texts = torch.zeros((1, len(all_texts), self.word_len))
        for i in range(len(all_texts)):
            t = self.tokenize(all_texts[i])  
            texts[0, i] += t

        return texts.long()


def val_transform(im, image_resolution):
    normalize = T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    transform = T.Compose([
        T.ToTensor(),
        normalize,
    ])

    w, h = im.size
    if max(w, h) > image_resolution: 
        ratio = float(image_resolution) / max(w, h)
        w, h = int(w * ratio), int(h * ratio)
    images = transform(im.resize((w, h)))

    images_ = torch.zeros((3, image_resolution, image_resolution))
    mask_ = torch.ones((image_resolution, image_resolution), dtype=torch.bool)
    images_[: images.shape[0], : images.shape[1], : images.shape[2]].copy_(images)
    mask_[: images.shape[1], :images.shape[2]] = False
    mask_ = mask_[::32, ::32]

    return images_.unsqueeze(0).to(device), mask_.unsqueeze(0).to(device)


def visualize(image, image_mask, char_mask, chars_pred, demo_path, text_list):
    att_show = image_mask[0]
    att_show = att_show[:, 1:].view(257, 16, 16)
    att_show = att_show[0].cpu().numpy()

    im = image[0].permute(1, 2, 0).numpy()
    im = cv2.normalize(im, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)
    h, w, _ = im.shape
    mask = cv2.resize(att_show, dsize=(w, h))
    mask = 255 - cv2.normalize(mask, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)
    mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    added_image = cv2.addWeighted(im, 0.8, mask,0.6, 0)
    cv2.imwrite('{}/im_attn_demo.jpg'.format(demo_path), added_image[:, :, ::-1])


    att_show = char_mask[0]
    att_show = att_show.view(len(text_list), 16, 16)
    fig, ax = plt.subplots(ncols=len(text_list), figsize=(15,8))

    for idx, (row, c_p) in enumerate(zip(ax, chars_pred)):
        
        att_show_ = att_show[idx].float().cpu().numpy()

        im = image[0].permute(1, 2, 0).numpy()
        im = cv2.normalize(im, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)
        h, w, _ = im.shape
        mask = cv2.resize(att_show_, dsize=(w, h))
        mask = 255 - cv2.normalize(mask, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)
        mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        added_image = cv2.addWeighted(im, 0.8, mask,0.6, 0)
        cv2.imwrite('{}/char_attn_demo_{}.jpg'.format(demo_path, idx), added_image[:, :, ::-1])

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Visualization of oCLIP Pre-trained Model")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--char_dict_path", type=str)
    parser.add_argument("--model_config_file", type=str, default="src/training/model_configs/RN50.json")
    parser.add_argument("--im_fn", type=str, default="demo/sample.jpg")
    parser.add_argument("--text_list", nargs='+', default=["ST LING", "STRLIN ", "A GYLL'S", " ODGINGS"])
    parser.add_argument("--demo_path", type=str, default="demo/")
    args = parser.parse_args()

    device = "cpu"

    with open(args.model_config_file, 'r') as f:
        model_info = json.load(f)

    model = load_model(args.model_path, model_info)

    image, im_mask = val_transform(Image.open(args.im_fn), model_info["image_resolution"])
    
    char_tokenizer = tokenizer(args.char_dict_path)
    text = char_tokenizer.char_token(args.text_list)
    text = text.to(device)

    with torch.no_grad():

        image_features, text_features, chars, image_mask, char_mask, logit_scale = model(image, text, im_mask)

        chars = chars.argmax(dim=-1)[0].numpy()
        chars_pred = [char_tokenizer.idx2p[x] for x in chars]


    visualize(image, image_mask, char_mask, chars_pred, args.demo_path, args.text_list)