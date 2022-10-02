# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2022) B-
# ytedance Inc..  
# *************************************************************************

import os
import time
import json
import numpy as np

import torch
import torch.nn as nn

from torch.cuda.amp import autocast
import torch.distributed as dist

import sys
import pdb

import logging
import random

def is_master(args):
    return (not args.distributed) or args.gpu == 0


def get_loss(model, images, texts, masked_chars, image_masks, loss_img, loss_txt, loss_char, args):
    image_features, text_features, text_logits, logit_scale = model(images, texts, image_masks)
    logit_scale = logit_scale.mean()

    indices = torch.where(masked_chars != 0)
    char_gts = masked_chars[indices]
    char_logits = text_logits[indices]
    char_loss = loss_char(char_logits, char_gts)

    if args.distributed and args.aggregate:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # We gather tensors from all gpus to get more negatives to contrast with.
        gathered_image_features = [
            torch.zeros_like(image_features) for _ in range(world_size)
        ]
        gathered_text_features = [
            torch.zeros_like(text_features) for _ in range(world_size)
        ]
        gattered_char_loss = [
            torch.zeros_like(char_loss) for _ in range(world_size)
        ]

        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)
        dist.all_gather(gattered_char_loss, char_loss)

        all_image_features = torch.cat(
            [image_features]
            + gathered_image_features[:rank]
            + gathered_image_features[rank + 1 :]
        )
        all_text_features = torch.cat(
            [text_features]
            + gathered_text_features[:rank]
            + gathered_text_features[rank + 1 :]
        )
        all_char_loss = char_loss + \
            torch.sum(torch.tensor(gattered_char_loss[:rank])).to(images.device) + \
            torch.sum(torch.tensor(gattered_char_loss[rank + 1 :])).to(images.device)

        # this is needed to send gradients back everywhere.
        logits_per_image = logit_scale * all_image_features @ all_text_features.t()
        logits_per_text = logits_per_image.t()
    else:
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()
        all_char_loss = char_loss

    ground_truth = torch.arange(len(logits_per_image)).long()

    if args.gpu is not None:
        ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)

    char_loss = all_char_loss * args.char_loss_weight
    # return char_loss
    total_loss = (
        loss_img(logits_per_image, ground_truth)
        + loss_txt(logits_per_text, ground_truth)
    ) / 2

    return char_loss, total_loss


def get_loss_first_stage(model, images, texts, image_masks, loss_img, loss_txt, args):
    image_features, text_features, logit_scale = model(images, texts, image_masks)
    logit_scale = logit_scale.mean()

    if args.distributed and args.aggregate:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # We gather tensors from all gpus to get more negatives to contrast with.
        gathered_image_features = [
            torch.zeros_like(image_features) for _ in range(world_size)
        ]
        gathered_text_features = [
            torch.zeros_like(text_features) for _ in range(world_size)
        ]

        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)

        all_image_features = torch.cat(
            [image_features]
            + gathered_image_features[:rank]
            + gathered_image_features[rank + 1 :]
        )
        all_text_features = torch.cat(
            [text_features]
            + gathered_text_features[:rank]
            + gathered_text_features[rank + 1 :]
        )

        # this is needed to send gradients back everywhere.
        logits_per_image = logit_scale * all_image_features @ all_text_features.t()
        logits_per_text = logits_per_image.t()
    else:
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

    ground_truth = torch.arange(len(logits_per_image)).long()

    if args.gpu is not None:
        ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)

    # return char_loss
    total_loss = (
        loss_img(logits_per_image, ground_truth)
        + loss_txt(logits_per_text, ground_truth)
    ) / 2

    return total_loss

def train(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    os.environ["WDS_EPOCH"] = str(epoch)
    model.train()

    dataloader, sampler = data['train'].dataloader,  data['train'].sampler

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    loss_char = nn.CrossEntropyLoss()
    if args.gpu is not None:
        loss_img = loss_img.cuda(args.gpu)
        loss_txt = loss_txt.cuda(args.gpu)
        if not args.first:
            loss_char = loss_char.cuda(args.gpu)

    if args.distributed and sampler is not None:
        sampler.set_epoch(epoch)

    num_batches_per_epoch = dataloader.num_batches

    end = time.time()
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        optimizer.zero_grad()

        if args.first:
            images, texts, image_masks = batch
        else:
            images, texts, chars, image_masks = batch

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            texts = texts.cuda(args.gpu, non_blocking=True)
            if not args.first:
                chars = chars.cuda(args.gpu, non_blocking=True)


        data_time = time.time() - end

        m = model.module if args.distributed or args.dp else model

        # with automatic mixed precision.
        if args.precision == "amp":
            with autocast():

                if args.first:
                    total_loss = get_loss_first_stage(model, images, texts, image_masks, loss_img, loss_txt, args)
                else:
                    char_loss, clip_loss = get_loss(model, images, texts, chars, image_masks, loss_img, loss_txt, loss_char, args)
                    total_loss = char_loss + clip_loss


                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
            scaler.update()

        else:
            if args.first:
                total_loss = get_loss_first_stage(model, images, texts, image_masks, loss_img, loss_txt, args)
            else:
                char_loss, clip_loss = get_loss(model, images, texts, chars, image_masks, loss_img, loss_txt, loss_char, args)
                total_loss = char_loss + clip_loss

            total_loss.backward()
            optimizer.step()
        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        # m.logit_scale.data = torch.clamp(m.logit_scale.data, 0, 4.6052)

        batch_time = time.time() - end
        end = time.time()
        if is_master(args) and (i % 100) == 0:
            num_samples = i * len(images) * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * i / num_batches_per_epoch
            if args.first:
                logging.info(
                    f"Train Epoch: {epoch} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)]\t"
                    f"Loss: total loss: {total_loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
                    f"\tLR: {optimizer.param_groups[0]['lr']:5f}"
                )
            else:
                logging.info(
                    f"Train Epoch: {epoch} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)]\t"
                    f"Loss: char loss: {char_loss.item():.6f}\tclip loss: {clip_loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
                    f"\tLR: {optimizer.param_groups[0]['lr']:5f}"
                )
            # save train loss / etc.

            timestep = epoch * num_batches_per_epoch + i
            log_data = {
                "loss": total_loss.item(),
                "data_time": data_time,
                "batch_time": batch_time,
                # "scale":  m.logit_scale.data.item(),
                "lr": optimizer.param_groups[0]["lr"]
            }

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, timestep)
