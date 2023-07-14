
<div align="center">
  
# Dialogue-to-Video Retrieval 🎥💬

<!-- **Authors:** -->

**[Chenyang Lyu](https://lyuchenyang.github.io), [Manh-Duy Nguyen](mailto:manh.nguyen5@mail.dcu.ie), [Van-Tu Ninh](mailto:van.ninh2@mail.dcu.ie)**<sup>†</sup>, **[Liting Zhou](mailto:liting.zhou@dcu.ie), [Cathal Gurrin](mailto:cathal.gurrin@dcu.ie), [Jennifer Foster](mailto:jennifer.foster@dcu.ie)**

<!-- **Affiliations:** -->

School of Computing, Dublin City University, Dublin, Ireland 🏫

† The first three authors contributed equally. 🤝

</div>

This repository contains the code for ECIR paper _Dialogue-to-Video Retrieval_, which proposed a novel approach for retrieving videos based on dialogue queries. The system incorporates structured conversational information to improve retrieval performance. 💡



## Table of Contents 📑

- [1. Introduction](#1-introduction)
- [2. Dataset](#2-dataset)
- [3. Pre-processing](#3-pre-processing)
- [4. Training](#4-training)
- [5. Usage](#5-usage)
- [6. Dependencies](#6-dependencies)

## 1. Introduction 📝

Recent years have witnessed an increasing amount of dialogue/conversation on the web, especially on social media. This has inspired the development of dialogue-based retrieval systems. In the case of dialogue-to-video retrieval, videos are retrieved based on user-generated dialogue queries. This approach utilizes structured conversational information to improve the accuracy of video recommendations. 🌐

This repository presents a novel dialogue-to-video retrieval system that incorporates structured conversational information. Experimental results on the AVSD dataset demonstrate the superiority of our approach over previous models, achieving significant improvements in retrieval performance. 📈

## 2. Dataset 📚

To run the system, you need to download the AVSD dataset. The dataset is available at the following links:

- [avsd dataset](https://drive.google.com/file/d/1P9gmqJFsxuimp_BENLy0Tp98MdGqelIL/view?usp=sharing)

In addition, you also need to download the original videos from the Charades dataset. The videos can be downloaded from [this link](https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip).

Please put the downloaded videos and dataset into the directory "data/avsd/". 📂

## 3. Pre-processing 🔄

Before training the dialogue-to-video retrieval model, you need to pre-process the AVSD dataset. To do this, run the following command:

```shell script
python data_preprocess.py
```

This script will extract video frames and audios from the AVSD videos and process the dataset into a tensor dataset. 🎞️🔉

## 4. Training 🚀

To train the dialogue-to-video retrieval model, you can use the provided script. Note that the script is expected to run on a server with at least 4 GPUs (ideally NVIDIA A100). The ideal batch size is 16 in total, so if running on a 4-GPU server, the batch size for each GPU should be 4.

```shell script
python run_dialogue_to_video_retrieval.py --do_train --do_eval --num_train_epochs 5 --n_frames 12 --learning_rate 1e-5 --train_batch_size 4 --eval_batch_size 16 --attention_heads 8 --eval_steps 100000 --n_gpu 4 --image_dir data/avsd/frames/ --clip_model_name openai/clip-vit-base-patch16 --clip_processor_name ViT-B/16
```

## 5. Usage 💻

Once the model is trained, you can use it for dialogue-to-video retrieval. Provide a dialogue query, and the system will retrieve the most relevant videos based on the query. 🔎

## 6. Dependencies 🛠️

- Python (>=3.8) 🐍
- Pytorch (>=2.0) 🔥
- NumPy 🧮
- Pandas 🐼

Please make sure to install the required dependencies before running the code. ⚙️

## Citation 📄

Please cite our paper using the bibtex below if you found that our paper is useful to you:

```bibtex
@inproceedings{lyu2023dialogue,
  title={Dialogue-to-Video Retrieval},
  author={Lyu, Chenyang and Nguyen, Manh-Duy and Ninh, Van-Tu and Zhou, Liting and Gurrin, Cathal and Foster, Jennifer},
  booktitle={Advances in Information Retrieval: 45th European Conference on Information Retrieval, ECIR 2023, Dublin, Ireland, April 2--6, 2023, Proceedings, Part II},
  pages={493--501},
  year={2023},
  organization={Springer}
}
```
