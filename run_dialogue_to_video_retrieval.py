""" running training and evaluation code for dialogue-to-video retrieval

    Created by Chenyang Lyu
"""

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import TensorDataset
from transformers import CLIPProcessor, CLIPModel, CLIPConfig
import torch.distributed as dist
from torch.nn import CrossEntropyLoss

import argparse
import sklearn.metrics as metric
import glob
import logging
import os
import random
import numpy as np
import json
import pickle
import codecs
from PIL import Image
from tqdm import tqdm, trange
from sklearn.metrics import top_k_accuracy_score
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    get_linear_schedule_with_warmup,
)

from modeling import AADV
import clip

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)
json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def draw_samples(lis, ratio):
    samples = ratio if ratio > 1 else int(ratio * len(lis))

    if samples > len(lis):
        new_lis = np.random.choice(len(lis), samples, replace=True)
    else:
        new_lis = np.random.choice(len(lis), samples, replace=False)

    n_lis = [lis[i] for i in new_lis]

    return n_lis


def train(args, model, train_dataset, preprocess, val_set=None):
    """ Training the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    train_dataset, train_video_names = train_dataset

    args.train_batch_size = args.train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    t_total = (len(train_dataloader) * args.num_train_epochs) // args.gradient_accumulation_steps

    # Prepare optimizer for training
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_group_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        }
    ]

    optimizer = AdamW(optimizer_group_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_steps * t_total),
                                                num_training_steps=t_total)
    loss_fct = CrossEntropyLoss()
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    # Train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    steps_trained_in_current_epoch = 0

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            # Skip past any already trained steps
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1

            batch = tuple(t.to(args.device) for t in batch)

            train_video_ind = list(batch[0].cpu().numpy())

            all_image_frames = []
            for vid in train_video_ind:
                _all_image_frames = []
                for vfi in args.train_frame_ind:
                    frame = preprocess(
                        Image.open('{}{}_{}.jpg'.format(args.image_dir, train_video_names['data'][vid], str(vfi))))
                    _all_image_frames.append(frame.unsqueeze(0))
                _all_image_frames = torch.cat(_all_image_frames, dim=0).unsqueeze(0)
                all_image_frames.append(_all_image_frames)

            all_image_frames = torch.cat(all_image_frames, dim=0).to(args.device)

            inputs = {'image_frames': all_image_frames,
                      'audio': batch[1],
                      'summary': batch[2],
                      'script': batch[3],
                      'dialog': batch[4],
                      'ans_in_dialog': batch[5],
                      }

            image_features, n_video_features, n_text_features, logit_scale = model(inputs)

            logit_scale = logit_scale[0] if args.n_gpu > 1 else logit_scale

            logits = torch.mm(logit_scale * n_video_features, n_text_features.t())

            labels = torch.tensor([i for i in range(n_text_features.size(0))], dtype=torch.long,
                                  device=args.device)

            loss_i = loss_fct(logits, labels)
            loss_e = loss_fct(logits.t(), labels)

            loss = (loss_i + loss_e) / 2
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if args.local_rank == -1 and global_step % args.eval_steps == 0 and val_set is not None:  # Only evaluate when single GPU otherwise metrics may not average well
                        evaluate(args, model, preprocess, val_set)
                        model_to_save = model.module if hasattr(model,
                                                                'module') else model  # Take care of distributed/parallel training
                        logger.info("Saving model checkpoint to %s", args.output_dir)
                        torch.save(model_to_save.state_dict(), args.output_dir + 'model.pt')
                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_last_lr()[0]
                    logs['learning_rate'] = learning_rate_scalar
                    logs['loss'] = loss_scalar
                    logging_loss = tr_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{'step': global_step}}))

                # if args.save_steps > 0 and global_step % args.save_steps == 0:
                #     # Save model checkpoint
                #     output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                #     if not os.path.exists(output_dir):
                #         os.makedirs(output_dir)
                #     model_to_save = model if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                #     model_to_save.save_pretrained(output_dir)
                #     torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                #     logger.info("Saving model checkpoint to %s", output_dir)
    if args.local_rank in [-1, 0]:
        tb_writer.close()
    global_step = 1 if global_step == 0 else global_step

    return global_step, tr_loss / global_step


def evaluate(args, model, preprocess, eval_dataset, prefix=""):
    eval_dataset, eval_video_names = eval_dataset

    args._eval_batch_size = args.eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args._eval_batch_size)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        model_module = model.module
    else:
        model_module = model
    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_image_features = []
    all_text_features = []
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model_module.eval()
            batch = tuple(t.to(args.device) for t in batch)

            eval_video_ind = list(batch[0].cpu().numpy())

            all_image_frames = []
            for vid in eval_video_ind:
                _all_image_frames = []
                for vfi in args.eval_frame_ind:
                    frame = preprocess(
                        Image.open('{}{}_{}.jpg'.format(args.image_dir, eval_video_names['data'][vid], str(vfi))))
                    _all_image_frames.append(frame.unsqueeze(0))
                _all_image_frames = torch.cat(_all_image_frames, dim=0).unsqueeze(0)
                all_image_frames.append(_all_image_frames)
            all_image_frames = torch.cat(all_image_frames, dim=0).to(args.device)

            inputs = {'image_frames': all_image_frames,
                      'audio': batch[1],
                      'summary': batch[2],
                      'script': batch[3],
                      'dialog': batch[4],
                      'ans_in_dialog': batch[5],
                      }

            # image encoding without self-attention
            all_image_features.append(model_module.encode_image(inputs['image_frames']).transpose(0, 1))

            if args.search_key in ['script', 'summary']:
                all_text_features.append(model_module.clip.get_text_features(inputs[args.search_key]))
            else:
                all_text_features.append(model_module.encode_dialogue_query(inputs[args.search_key],
                                                                            inputs[args.dialog_feature_key]))
            # print(all_image_features[-1].size(), all_text_features[-1].size())
        all_image_features = torch.cat(all_image_features, dim=0)
        all_video_features = torch.sum(all_image_features, dim=1) / all_image_features.size(1)
        all_text_features = torch.cat(all_text_features, dim=0)

        # r_text_features = all_text_features.unsqueeze(0).repeat(all_text_features.size(0), 1, 1)  # added repeat

        # r_text_features = r_text_features.transpose(0, 1)
        # all_image_features = all_image_features.transpose(0, 1)

        # all_image_features = model_module.query_multi_attention(r_text_features,
        #                                                         all_image_features,
        #                                                         all_image_features)[0].transpose(0, 1).to('cuda')
        # model.to('cuda')

        # # with l2 norm
        t_video_features = torch.nn.functional.normalize(model_module.video_to_multimodal(all_video_features), p=2,
                                                         dim=-1)
        t_text_features = torch.nn.functional.normalize(model_module.text_to_multimodal(all_text_features), p=2, dim=-1)

        # # without l2 norm
        # t_video_features = model.video_to_multimodal(all_image_features)
        # t_text_features = model.text_to_multimodal(all_text_features)

        logit_scale = model_module.logit_scale.exp()

        # original multiply
        logits = torch.mm(logit_scale * t_video_features, t_text_features.t())

        # text weighted multiply
        # t_text_features = t_text_features.unsqueeze(1)
        # logits = torch.bmm(logit_scale * t_video_features.transpose(0, 1),
        #                    t_text_features.transpose(1, 2)).squeeze(-1)

        logits = logits.cpu().numpy()
    labels = [i for i in range(t_video_features.size(0))]

    top_1 = top_k_accuracy_score(labels, logits, k=1)
    top_5 = top_k_accuracy_score(labels, logits, k=5)
    top_10 = top_k_accuracy_score(labels, logits, k=10)
    print('Metrics: top-1: {}, top-5: {}, top-10: {}'.format(str(round(100 * top_1, 2)),
                                                             str(round(100 * top_5, 2)),
                                                             str(round(100 * top_10, 2))))
    evaluate_rank(logits, labels)

    return


def evaluate_rank(sim_matrix, labels):
    ranks = []
    for logits, label in zip(sim_matrix, labels):
        logits_w_ind = {ind: logit for ind, logit in enumerate(logits)}
        rank_list = [key for key, value in sorted(logits_w_ind.items(), key=lambda item: item[1], reverse=True)]
        ranks.append(rank_list.index(label) + 1)

    print('Metrics: median rank: {}, mean rank: {}'.format(str(np.median(ranks)), str(np.mean(ranks))))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task_name", type=str, default="avsd",
                        help="the name of the training task (the dataset name)")
    parser.add_argument("--model_size", type=str, default="16",
                        help="the size of pre-trained CLIP model (ViT-16 or ViT-32)")
    parser.add_argument("--num_train_epochs", type=int, default=10,
                        help="the numebr of training epochs")
    parser.add_argument("--do_train", action="store_true",
                        help="whether to train the model or not")
    parser.add_argument("--do_eval", action="store_true",
                        help="whether to evaluate the model or not")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="the weight decay rate")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="the learning rate used to train the model")
    parser.add_argument("--warmup_steps", type=float, default=0.0,
                        help="the warm_up step rate")
    parser.add_argument("--seed", type=int, default=0,
                        help="the random seed used in model initialization and dataloader")
    parser.add_argument("--train_batch_size", type=int, default=16,
                        help="the batch size used in training")
    parser.add_argument("--eval_batch_size", type=int, default=16,
                        help="the batch size used in evaluation")
    parser.add_argument("--logging_steps", type=int, default=50,
                        help="the logging steps")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="conduct evaluation every eval_steps")
    parser.add_argument("--device", type=int, default=0,
                        help="the device id used for training and evaluation")
    parser.add_argument("--n_gpu", type=int, default=1,
                        help="number of gpus being used")
    parser.add_argument("--attention_heads", type=int, default=8,
                        help="the attention heads used in multi head attention function")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--search_key", type=str, default="dialog",
                        help="the key used in retrieval")
    parser.add_argument("--dialog_feature_key", type=str, default="summary",
                        help="the key used in dialog feature fusion")
    parser.add_argument("--n_frames", type=int, default=6,
                        help="the frames sampled from each video in training")
    parser.add_argument("--eval_n_frames", type=int, default=6,
                        help="the frames sampled from each video in evaluation")
    parser.add_argument("--all_frame_feature_ratio", type=float, default=1.0,
                        help="the coefficient used to multiply with all frame features in training")
    parser.add_argument("--eval_all_frame_feature_ratio", type=float, default=1.0,
                        help="the coefficient used to multiply with all frame features in evaluation")
    parser.add_argument("--dialog_runs", type=int, default=10,
                        help="the runs of dialog query used in training and evaluation")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--image_dir", type=str, default="data/avsd/frames/",
                        help="the directory used to store video frames.")
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-base-patch16",
                        help="the name for the CLIP model used in training.")
    parser.add_argument("--clip_processor_name", type=str, default="ViT-B/16",
                        help="the name for the CLIP processor used in training.")
    args, _ = parser.parse_known_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1

    args.device = device
    args.adam_epsilon = 1e-8
    args.max_grad_norm = 5.0

    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    clip_model_name = args.clip_model_name  # openai/clip-vit-large-patch14-336
    clip_processor_name = args.clip_processor_name  # "ViT-L/14@336px"
    clip_config = CLIPConfig.from_pretrained(clip_model_name)

    args.transformer_width = clip_config.projection_dim
    args.audio_feature_dim = 20000

    args.t_frames = 60
    args.e_frames = 60
    interval = args.t_frames // args.n_frames

    frame_ind = [i * interval for i in range(args.n_frames)]
    for i in range(len(frame_ind)):
        if frame_ind[i] >= args.t_frames:
            frame_ind[i] = args.t_frames - 1
    frame_ind[-1] = args.t_frames - 1
    args.train_frame_ind = frame_ind

    # randomly sampled index
    # args.frame_ind = draw_samples([i for i in range(args.t_frames)], args.n_frames)
    # args.eval_n_frames = 30
    interval = args.e_frames // args.n_frames

    frame_ind = [i * interval for i in range(args.eval_n_frames)]
    for i in range(len(frame_ind)):
        if frame_ind[i] >= args.e_frames:
            frame_ind[i] = args.e_frames - 1
    frame_ind[-1] = args.e_frames - 1
    args.eval_frame_ind = frame_ind

    # args.eval_frame_ind = args.train_frame_ind

    if 'large' in args.image_dir:
        args.train_frame_ind = [i for i in range(args.n_frames)]
        args.eval_frame_ind = [i for i in range(args.n_frames)]

    model_prefix = 'video_retrieval'

    args.output_dir = 'trained_models/dialog_to_video_retrieval/' \
                      '{}_{}_epochs-{}_lr-{}'.format(model_prefix,
                                                     args.search_key,
                                                     str(args.num_train_epochs),
                                                     str(args.learning_rate))
    if args.local_rank in [-1, 0]:
        print(args.output_dir)
    data_dirs = ["data/avsd/train.cache", "data/avsd/val.cache", "data/avsd/test.cache"]
    video_names = ["data/avsd/train_video_names.json", "data/avsd/val_video_names.json", "data/avsd/test_video_names.json"]

    all_images, all_audios, all_summaries, all_scripts, all_dialogs, all_ans_in_dialog = pickle.load(
        open(data_dirs[0], 'rb'))
    train_dataset = TensorDataset(torch.cat(all_images, dim=0),
                                  torch.cat([audio.unsqueeze(0) for audio in all_audios], dim=0),
                                  torch.cat(all_summaries, dim=0), torch.cat(all_scripts, dim=0),
                                  torch.cat([dialog.unsqueeze(0) for dialog in all_dialogs], dim=0),
                                  torch.cat([ans.unsqueeze(0) for ans in all_ans_in_dialog], dim=0))
    train_dataset = (train_dataset, json_load(video_names[0]))

    all_images, all_audios, all_summaries, all_scripts, all_dialogs, all_ans_in_dialog = pickle.load(
        open(data_dirs[1], 'rb'))

    val_video_names = json_load(video_names[1])
    val_dataset = TensorDataset(torch.cat(all_images, dim=0),
                                torch.cat([audio.unsqueeze(0) for audio in all_audios], dim=0),
                                torch.cat(all_summaries, dim=0),
                                torch.cat(all_scripts, dim=0),
                                torch.cat([dialog.unsqueeze(0) for dialog in all_dialogs], dim=0),
                                torch.cat([ans.unsqueeze(0) for ans in all_ans_in_dialog], dim=0))
    val_dataset = (val_dataset, {'data': val_video_names['data']})

    all_images, all_audios, all_summaries, all_scripts, all_dialogs, all_ans_in_dialog = pickle.load(
        open(data_dirs[2], 'rb'))

    test_video_names = json_load(video_names[2])

    test_dataset = TensorDataset(torch.cat(all_images, dim=0),
                                 torch.cat([audio.unsqueeze(0)
                                            for audio in all_audios], dim=0),
                                 torch.cat(all_summaries, dim=0),
                                 torch.cat(all_scripts, dim=0),
                                 torch.cat([dialog.unsqueeze(0)
                                            for dialog in all_dialogs], dim=0),
                                 torch.cat([ans.unsqueeze(0)
                                            for ans in all_ans_in_dialog], dim=0))
    test_dataset = (test_dataset, {'data': test_video_names['data']})

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load(clip_processor_name, device=device)
    del clip_model

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        model = AADV(args, clip_config)
        model.clip = CLIPModel.from_pretrained(clip_model_name)
        model.to(args.device)
        global_step, tr_loss = train(args, model, train_dataset, preprocess, val_set=val_dataset)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training

        logger.info("Saving model checkpoint to %s", args.output_dir)
        torch.save(model_to_save.state_dict(), args.output_dir + 'model.pt')

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

    # Evaluation
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            model = AADV(args, clip_config)
            model.load_state_dict(torch.load(checkpoint + 'model.pt'))
            model.to(args.device)
            evaluate(args, model, preprocess, val_dataset)
            evaluate(args, model, preprocess, test_dataset)
    return


if __name__ == '__main__':
    main()
