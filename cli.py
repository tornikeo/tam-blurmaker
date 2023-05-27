#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('reload_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
# In[2]:
from tqdm import tqdm

# from cli_tools import *
from cli_tools import (TrackingAnything, download_checkpoint,
                       download_checkpoint_from_google_drive,
                       get_frames_from_video, sam_refine, select_template,
                       vos_tracking_video)


def redirect_stdout():
    sys.stdout = open("stdout.txt", "w")
    sys.stderr = open("stderr.txt", "w")


# Desired Args structure
# --track_object "[]"
# --track_object "[202, 264, 4, 294, 74]" --frame_start 202 --frame_end 410 --input /Users/maciejkrupowies/.bmcache/pexels-free-videos-853889-1920x1080-25fps
args = argparse.ArgumentParser()
# args.add_argument('--input', type=str, default='test_sample/frames_family_144')

# {
#     "points": [
#         {
#             "frame": 444,
#             "pos": [360,110],
#             "label": 1
#         }
#     ],
#     "track_end_number": 1924


## Sample commands (mall_480):
# python cli.py --input test_sample/frames_mall_480 --frame_start 0 --frame_end 700 --track_object "[[0, 164, 155, 185, 209]]"
# python cli.py --input test_sample/frames_family_480 --frame_start 1031 --frame_end 1500 --track_object "[[1031, 216, 56, 295, 235]]"

args.add_argument("--input", type=str, default="test_sample/frames_family_480")
args.add_argument(
    "--frame_start",
    type=int,
    default=444,
)  # default='test_sample/family_480/blue_dress_lady_face.json')
args.add_argument(
    "--frame_end", type=int, default=1924
)  # default='test_sample/family_480/blue_dress_lady_face.json')
# {firstFrame, x_min, y_min, x_max, y_max}
# args.add_argument('--track_object', type=str, default='[[1008, 85, 23, 86, 24]]') #default='test_sample/family_144/blue_dress_lady_face.json')
args.add_argument(
    "--track_object", type=str, default="[[444, 350, 100, 370, 120]]"
)  # default='test_sample/family_144/blue_dress_lady_face.json')
args.add_argument(
    "--debug",
    action="store_true",
    help="Print debug info to screen",
)  # default='test_sample/family_144/blue_dress_lady_face.json')

args.add_argument(
    '--debug-video',
    action='store_true',
    help='Create debug video at the end of the run',
)
args.add_argument(
    "--cap_frame_size",
    type=int,
    help="If set, will reduce the frame size to AT MOST this value, e.g. if your video has 4k resolution, and you set this."
    "to 1080, the frames will be reduced to 1080x1080. This is useful for reducing the compute requirements.",
    default=None,
)  # default='test_sample/family_144/blue_dress_lady_face.json')

args = args.parse_args()

if not args.debug:
    redirect_stdout()

if args.cap_frame_size is not None:
    print(
        f"WARNINIG: cap_frame_size is set to {args.cap_frame_size}. This will reduce the frame size to AT MOST {args.cap_frame_size}x{args.cap_frame_size}."
    )

args.sam_model_type = "vit_b"
import time

# args.debug = False
args.output_video = Path(f"debug-video-output-{time.time_ns()}.mp4")
args.mask_save = False
args.output = Path("output.json")

try:
    args.device = "cuda"
    torch.nn.functional.conv2d(
        torch.randn(8, 4, 3, 3).to("cuda"),
        torch.randn(1, 4, 5, 5).to("cuda"),
        padding=1,
    )
except Exception as e:
    print(f"GPU init failed with: {e}")
    print("Using CPU, this will be slow.")
    args.device = "cpu"

print("Woo hoo. Let's go!")

# args, defined in track_anything.py
# args = parse_argument()
# args = default_args()
# args = argparse.Namespace()
# args.input = Path("test_sample/family_480.mp4")
# args.track_data = Path("test_sample/family_480/blue_dress_lady_face.json")
# try:
#     args.device = "cuda" if torch.cuda.is_available() else "cpu"
#     torch.nn.functional.conv2d(torch.zeros(1, 1).to(args.device), torch.zeros(1, 1).to(args.device))
# except RuntimeError as e:
#     print(e)
#     args.device = "cpu"
# args.sam_model_type = "vit_b"
# args.output = Path("output.json")
# args.debug = False
# args.mask_save = False
# args.output_video = Path("result.mp4")
# args.track_data = json.load(open(args.track_data, "r"))


# {
#     "points": [
#         {
#             "frame": 1008,
#             "pos": [286,79],
#             "label": 1
#         }
#     ],
#     "track_end_number": 1226
# }

# if args.track_data is not None:
#     args.track_data = json.load(open(args.track_data, "r"))
# else:
#     args.track_data = json.loads(args.track_data_raw)
#     start_frame, x, y, label, end_frame = args.track_data[0]
#     args.track_data = {
#         "points": [
#             {
#                 "frame": start_frame,
#                 "pos": [x, y],
#                 "label": label
#             }
#         ],
#         "track_end_number": end_frame
#     }
# print(f'args.track_data: {args.track_data}')

args.track_data = json.loads(args.track_object)
start_frame, x_min, y_min, x_max, y_max = args.track_data[0]
assert (
    start_frame == args.frame_start
), "--track_object [start_frame,...] must be equal to --frame_start"
print("WARNING: Currently only one label (label 0) is supported.")
# print('WARNING: Currently the BBOX is reduced to a single middle point at the center (x,y).')
print(
    "WARNING: Currently --frame_start is forced to be equal to the first --track_object BBOX frameNum."
)
args.track_data = dict(
    frame_end=args.frame_end,
    bboxes=[
        dict(
            frame=start_frame,
            label=0,
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
        )
    ],
)
print(f"args.track_data: {args.track_data}")
# args.track_data = {
#     "points": [
#         {
#             "frame": start_frame,
#             "pos": [x, y],
#             "label": label
#         }
#     ],
#     "track_end_number": end_frame
# }

# return args


# In[3]:


args.track_data


# In[4]:


# check and download checkpoints if needed
SAM_checkpoint_dict = {
    "vit_h": "sam_vit_h_4b8939.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_b": "sam_vit_b_01ec64.pth",
}
SAM_checkpoint_url_dict = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}
sam_checkpoint = SAM_checkpoint_dict[args.sam_model_type]
sam_checkpoint_url = SAM_checkpoint_url_dict[args.sam_model_type]
xmem_checkpoint = "XMem-s012.pth"
xmem_checkpoint_url = (
    "https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s012.pth"
)
e2fgvi_checkpoint = "E2FGVI-HQ-CVPR22.pth"
e2fgvi_checkpoint_id = "10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3"

folder = "./checkpoints"
SAM_checkpoint = download_checkpoint(sam_checkpoint_url, folder, sam_checkpoint)
xmem_checkpoint = download_checkpoint(xmem_checkpoint_url, folder, xmem_checkpoint)
e2fgvi_checkpoint = download_checkpoint_from_google_drive(
    e2fgvi_checkpoint_id, folder, e2fgvi_checkpoint
)
# args.port = 12212
# args.device = "cuda:1"
# args.mask_save = True

# initialize sam, xmem, e2fgvi models
model = TrackingAnything(SAM_checkpoint, xmem_checkpoint, None, args)
# video_input: /tmp/182f5d11c044d7004053ecf4b9f0678894a151ab/mall_480.mp4
# video_state: {'user_name': '', 'video_name': '', 'origin_images': None, 'painted_images': None, 'masks': None, 'inpaint_masks': None, 'logits': None, 'select_frame_number': 0, 'fps': 30}
# interactive_state = {
#     "inference_times": 0,
#     "negative_click_times": 0,
#     "positive_click_times": 0,
#     "mask_save": args.mask_save,
#     "multi_mask": {"mask_names": [], "masks": []},
#     "track_end_number": None if args.track_data['end_frame'] == -1 else args.track_data['end_frame'],
#     "resize_ratio": 1,
# }
# video_state = {
#     "user_name": "",
#     "video_name": "",
#     "origin_images": None,
#     "painted_images": None,
#     "masks": None,
#     "inpaint_masks": None,
#     "logits": None,
#     "select_frame_number": 0,
#     "fps": 30,
# }


# In[5]:


# video_state, video_info, origin_image = get_frames_from_video(
#     model,
#     args.input,
#     video_state,
# )
import cv2

files = sorted(list(Path(args.input).glob("*.jpg")))
print(f"Opening files: {files[:3]}...{files[-3:]}, len: {len(files)}")

frames = [
    cv2.cvtColor(
        cv2.imread(str(p)),
        cv2.COLOR_BGR2RGB,
    )
    for p in files
]
masks = [np.zeros((frames[0].shape[0], frames[0].shape[1]), np.uint8)]
logits = [None] * len(frames)
fps = 25

if args.cap_frame_size is not None and (max(frames[0].shape) > args.cap_frame_size):
    original_frame_dims = (frames[0].shape[0], frames[0].shape[1])
    print(f"Reducing frame size to {args.cap_frame_size}x{args.cap_frame_size}...")
    frames = [
        cv2.resize(frame, (args.cap_frame_size, args.cap_frame_size))
        for frame in frames
    ]
    masks = [
        cv2.resize(mask, (args.cap_frame_size, args.cap_frame_size)) for mask in masks
    ]
    resized_frame_dims = (frames[0].shape[0], frames[0].shape[1])
    for key in ["x_min", "x_max"]:
        args.track_data["bboxes"][0][key] = round(
            args.track_data["bboxes"][0][key]
            * resized_frame_dims[1]
            / original_frame_dims[1]
        )
    for key in ["y_min", "y_max"]:
        args.track_data["bboxes"][0][key] = round(
            args.track_data["bboxes"][0][key]
            * resized_frame_dims[0]
            / original_frame_dims[0]
        )
    print(f"After rescaling the bbox is now: {args.track_data['bboxes'][0]}")
model.samcontroler.sam_controler.reset_image()
# model.samcontroler.sam_controler.set_image(frames[0])
# In[6]:


# In[7]:


# points = args.track_data['points']
bbox = args.track_data["bboxes"][0]

# template_frame, video_state, interactive_state, run_status=select_template(
#     model,
#     bbox['frame'],
#     video_state,
#     interactive_state
# )
model.samcontroler.sam_controler.reset_image()
model.samcontroler.sam_controler.set_image(frames[bbox["frame"]])

# In[8]:


# evt = argparse.Namespace()
# evt.index = [0, 0]

# x_mid = round((bbox['x_min'] + bbox['x_max']) / 2)
# y_mid = round((bbox['y_min'] + bbox['y_max']) / 2)
# print(f"Rounding bbox to center point: ({x_mid}, {y_mid})")

# template_frame, video_state, interactive_state, run_status = sam_refine(
#     model=model,
#     video_state=video_state,
#     # point_prompt=sam_refine_args['point_prompt'],
#     point_prompt=None,#"Positive",
#     click_state=None,#[[180,176],[1]],
#     # prompt={
#     #     "prompt_type": ["click"],
#     #     "input_point": [points[0]['pos']],#[[180,176]],
#     #     "input_label": [points[0]['label']],
#     #     "multimask_output": "False",
#     # },
#     prompt=dict(
#         prompt_type=["click"],
#         input_point=[[x_mid, y_mid]],
#         input_label=[bbox['label']],
#         multimask_output="False",
#     ),
#     interactive_state=interactive_state,
#     evt=evt,
# )

# model.samcontroler.sam_controler.reset_image()
# model.samcontroler.sam_controler.set_image(
#     video_state["origin_images"][video_state["select_frame_number"]]
# )
# prompt=dict(
#     prompt_type=["click"],
#     input_point=[[x_mid, y_mid]],
#     input_label=[bbox['label']],
#     multimask_output="False",
# )

# template_mask, logit, painted_image = model.first_frame_click(
#     image=frames[bbox['frame']],
#     points=np.array(prompt["input_point"]),
#     labels=np.array(prompt["input_label"]),
#     multimask=prompt["multimask_output"],
# )

template_mask, logit, painted_image = model.samcontroler.first_frame_click(
    image=frames[bbox["frame"]],
    box=np.array([bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]]),
    # labels=np.array(prompt["input_label"]),
    # multimask=prompt["multimask_output"],
    multimask=False,
)

# template_mask = painted_image
# In[9]:
from typing import List


def bbox2(img):
    # rows = np.any(img, axis=1)
    # cols = np.any(img, axis=0)
    # try:
    #     rmin, rmax = np.where(rows)[0][[0, -1]].tolist()
    #     cmin, cmax = np.where(cols)[0][[0, -1]].tolist()
    #     return [rmin, rmax, cmin, cmax]
    # except IndexError:
    #     return None
    img_binary = img.astype(bool)
    rows = np.any(img_binary, axis=1)
    cols = np.any(img_binary, axis=0)
    try:
        y_min, y_max = np.where(rows)[0][[0, -1]].tolist()
        x_min, x_max = np.where(cols)[0][[0, -1]].tolist()
        return [x_min, y_min, x_max, y_max]
    except IndexError:
        return None


# video_output, video_state, interactive_state, run_status = vos_tracking_video(
#     model=model,
#     video_output=args.output_video,
#     video_state=video_state,
#     interactive_state=interactive_state,
#     mask_dropdown=[],
#     generate_callback=on_each_prediction,
# )
# outputs = []


model.xmem.clear_memory()

# if interactive_state["multi_mask"]["masks"]:
#     mask_dropdown = ["mask_001"]
#     template_mask = interactive_state["multi_mask"]["masks"][
#         int(mask_dropdown[0].split("_")[1]) - 1
#     ] * (int(mask_dropdown[0].split("_")[1]))
#     breakpoint()
#     for i in range(1, len(mask_dropdown)):
#         mask_number = int(mask_dropdown[i].split("_")[1]) - 1
#         template_mask = np.clip(
#             template_mask
#             + interactive_state["multi_mask"]["masks"][mask_number]
#             * (mask_number + 1),
#             0,
#             mask_number + 1,
#         )
#     video_state["masks"][video_state["select_frame_number"]] = template_mask
# else:

# template_mask = masks[video_state["select_frame_number"]]
# fps = video_state["fps"]

# masks = []
# logits = []
# painted_images = []
# images = video_state["origin_images"]

sys.stdout = sys.__stdout__
label = 0

all_painted_images: List[np.ndarray] = []

for i, frame in enumerate(frames):
    if i >= bbox["frame"] and i < args.frame_end:
        # if i == 0:
        mask, logit, painted_image = model.xmem.track(
            frame,
            first_frame_annotation=template_mask if i == bbox["frame"] else None,
        )
        det_bbox = bbox2(mask > 0)
        if det_bbox is None:
            det_bbox = [0, 0, 0, 0]

        all_painted_images.append(painted_image)

        if args.cap_frame_size is not None and (
            max(frames[0].shape) > args.cap_frame_size
        ):
            # det_bbox = [round(v * original_frame_dims[0] / resized_frame_dims[0]) for v in det_bbox]
            det_bbox[0] = round(
                det_bbox[0] * original_frame_dims[0] / resized_frame_dims[0]
            )
            det_bbox[1] = round(
                det_bbox[1] * original_frame_dims[1] / resized_frame_dims[1]
            )
            det_bbox[2] = round(
                det_bbox[2] * original_frame_dims[0] / resized_frame_dims[0]
            )
            det_bbox[3] = round(
                det_bbox[3] * original_frame_dims[1] / resized_frame_dims[1]
            )
        # FIXES: Remove output for frames out of range (eg. don't show bbox 0,0,0,0 when time_start was 300 and end 700 for all other frames)
        # print(
        #     i,
        #     ": {",
        #     label,
        #     ": {'class': 0, 'bbox': ",
        #     tuple(det_bbox),
        #     ", 'score': ''}}",
        # )
        print(f'{i}: {{"1": {{"class": 0, "bbox": {det_bbox}, "score": ""}}}}')
        # masks.append(mask)
        # logits.append(logit)
        # painted_images.append(painted_image)
        # if bbox is not None:
        #     bbox = [0, 0, 0, 0]
        # print(json.dumps({i: {label: {'class': 0, 'bbox': bbox, 'score': ''}}}))
    # elif i > bbox['frame'] and i < args.frame_end:
    # mask, logit, painted_image = model.xmem.track(frame)
    # masks.append(mask)
    # logits.append(logit)
    # painted_images.append(painted_image)
    # breakpoint()
    # det_bbox = bbox2(mask > 0)
    # Write outputs in {1: {'class': 0, 'bbox': [0, 0, 0, 0], 'score': ''}} format
    # print(json.dumps({i: {label: {'class': 0, 'bbox': bbox, 'score': ''}}}))

    # print(json.dumps({i: {label: {'class': 0, 'bbox': det_bbox, 'score': ''}}}))
    # sample output `197: {1: {'class': 0, 'bbox': [105, 98, 149, 126], 'score': ''}}`

    # if i >= bbox['frame'] and i < args.frame_end:
    #     print(i, ": {", label, ": {'class': 0, 'bbox': ", tuple(det_bbox), ", 'score': ''}}")

# Fixes: Add debug video
from cli_tools import generate_video_from_frames

if args.debug_video:
    # print("Generating video from frames...")
    video_output = generate_video_from_frames(
        all_painted_images,
        output_path=args.output_video,
        fps=fps,
    )  # import video_input to name the output video

# In[10]:

# video_state["masks"][video_state["select_frame_number"]] = mask
# for frame_num, mask in enumerate(video_state["masks"]):
#     # print(mask)
#     # print(i)
#     # mask = np.load(mask)
#     # Get bounding box [x,y,x,y] from binary mask
#     bbox = bbox2(mask > 0)
#     if bbox is not None:
#         # Write outputs in {1: {'class': 0, 'bbox': [0, 0, 0, 0], 'score': ''}} format
#         outputs.append({frame_num: {'class': 0, 'bbox': bbox, 'score': ''}})

# sys.stdout = save_stdout
# if not args.debug:
#     return_stdout()

# # Write outputs to json
# Path(args.output).open('w').write(json.dumps({
#     'results': outputs
# }, indent=4))

# print(json.dumps({
#     'results': outputs
#     }, indent=4)
# )
