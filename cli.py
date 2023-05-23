#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('reload_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[2]:

import sys
import io
import json
# from cli_tools import *
from cli_tools import download_checkpoint, download_checkpoint_from_google_drive, get_frames_from_video, TrackingAnything, \
    select_template, sam_refine, vos_tracking_video
import numpy as np
from pathlib import Path
import torch
from matplotlib import pyplot as plt
import argparse

save_stdout = sys.stdout
save_stderr = sys.stderr

def redirect_stdout():
    sys.stdout = open('stdout.txt', 'w')
    sys.stderr = open('stderr.txt', 'w')

def return_stdout():
    sys.stdout = save_stdout
    sys.stderr = save_stderr

# Desired Args structure
# --track_object "[]"
# --track_object "[202, 264, 4, 294, 74]" --frame_start 202 --frame_end 410 --input /Users/maciejkrupowies/.bmcache/pexels-free-videos-853889-1920x1080-25fps
args = argparse.ArgumentParser()
args.add_argument('--input', type=str, default='test_sample/frames_family_144')
args.add_argument('--frame_start', type=int, default=0,) # default='test_sample/family_480/blue_dress_lady_face.json')
args.add_argument('--frame_end', type=int, default=3000) # default='test_sample/family_480/blue_dress_lady_face.json')
# {firstFrame, x_min, y_min, x_max, y_max}
args.add_argument('--track_object', type=str, default='[[1008, 85, 23, 86, 24]]') #default='test_sample/family_144/blue_dress_lady_face.json')
args.add_argument('--silent', type=int, default=1) #default='test_sample/family_144/blue_dress_lady_face.json')
args = args.parse_args()

if args.silent:
    redirect_stdout()

args.sam_model_type = "vit_b"
args.debug = False
args.mask_save = False
args.output = Path("output.json")
args.output_video = Path("result.mp4")

try:
    args.device = 'cuda'
    torch.nn.functional.conv2d(torch.randn(8, 4, 3, 3).to('cuda'),
                                torch.randn(1, 4, 5, 5).to('cuda'), padding=1)
except Exception as e:
    print(f"GPU init failed with: {e}")
    print("Using CPU, this will be slow.")   
    args.device = 'cpu'    

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
print('WARNING: Currently only one label (label 0) is supported.')
print('WARNING: Currently the BBOX is reduced to a single middle point at the center (x,y).')
print('WARNING: Currently --frame_start is forced to be equal to the first --track_object BBOX frameNum.')
args.track_data = dict(
    start_frame = 0,
    end_frame = args.frame_end,
    bboxes = [
        dict(
            label = 0,
            frame = start_frame,
            x_min = x_min,
            y_min = y_min,
            x_max = x_max,
            y_max = y_max,
        )
    ]

)
print(f'args.track_data: {args.track_data}')
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
interactive_state = {
    "inference_times": 0,
    "negative_click_times": 0,
    "positive_click_times": 0,
    "mask_save": args.mask_save,
    "multi_mask": {"mask_names": [], "masks": []},
    "track_end_number": None if args.track_data['end_frame'] == -1 else args.track_data['end_frame'],
    "resize_ratio": 1,
}
video_state = {
    "user_name": "",
    "video_name": "",
    "origin_images": None,
    "painted_images": None,
    "masks": None,
    "inpaint_masks": None,
    "logits": None,
    "select_frame_number": 0,
    "fps": 30,
}


# In[5]:


video_state, video_info, origin_image = get_frames_from_video(
    model,
    args.input,
    video_state,
)


# In[6]:




# In[7]:


# points = args.track_data['points']
bbox = args.track_data['bboxes'][0]

template_frame, video_state, interactive_state, run_status=select_template(
    model,
    bbox['frame'], 
    video_state, 
    interactive_state
)


# In[8]:


evt = argparse.Namespace()
evt.index = [0, 0]

x_mid = round((bbox['x_min'] + bbox['x_max']) / 2)
y_mid = round((bbox['y_min'] + bbox['y_max']) / 2)
print(f"Rounding bbox to center point: ({x_mid}, {y_mid})")

template_frame, video_state, interactive_state, run_status = sam_refine(
    model=model,
    video_state=video_state,
    # point_prompt=sam_refine_args['point_prompt'],
    point_prompt=None,#"Positive",
    click_state=None,#[[180,176],[1]],
    # prompt={
    #     "prompt_type": ["click"],
    #     "input_point": [points[0]['pos']],#[[180,176]],
    #     "input_label": [points[0]['label']],
    #     "multimask_output": "False",
    # },
    prompt=dict(
        prompt_type=["click"],
        input_point=[[x_mid, y_mid]],
        input_label=[bbox['label']],
        multimask_output="False",
    ),
    interactive_state=interactive_state,
    evt=evt,
)


# In[9]:
from typing import List


def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    try:
        rmin, rmax = np.where(rows)[0][[0, -1]].tolist()
        cmin, cmax = np.where(cols)[0][[0, -1]].tolist()
        return rmin, rmax, cmin, cmax
    except IndexError:
        return None

def on_each_prediction(i: int, 
                       masks: List[np.ndarray], 
                       logits: List[np.ndarray], 
                       painted_images: List[np.ndarray]):
# for frame_num, mask in enumerate(video_state["masks"]):
    # print(mask)
    # print(i)
    # mask = np.load(mask)
    # Get bounding box [x,y,x,y] from binary mask
    saved = sys.stdout
    sys.stdout = sys.__stdout__
    bbox = bbox2(masks[-1] > 0)
    if bbox is None:
        bbox = [0, 0, 0, 0]
    else:
        assert False, f"i={i}, bbox={bbox}"
    label = 0
    # Write outputs in {1: {'class': 0, 'bbox': [0, 0, 0, 0], 'score': ''}} format
    print(json.dumps({i: {label: {'class': 0, 'bbox': bbox, 'score': ''}}}))
    sys.stdout = saved


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
from tqdm import tqdm

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
    
template_mask = video_state["masks"][video_state["select_frame_number"]]
fps = video_state["fps"]

masks = []
logits = []
painted_images = []
images = video_state["origin_images"]
sys.stdout = sys.__stdout__
label = 0
for i in range(len(images)):
    if i == video_state["select_frame_number"]:
    # if i == 0:
        mask, logit, painted_image = model.xmem.track(images[i], template_mask)
        masks.append(mask)
        logits.append(logit)
        painted_images.append(painted_image)  

        bbox = bbox2(masks[-1] > 0)
        if bbox is None:
            bbox = [0, 0, 0, 0]
        print(json.dumps({i: {label: {'class': 0, 'bbox': bbox, 'score': ''}}}))
    elif i > video_state["select_frame_number"] and i < video_state.get("track_end_number", len(images) - 1):
        mask, logit, painted_image = model.xmem.track(images[i])
        masks.append(mask)
        logits.append(logit)
        painted_images.append(painted_image)
        breakpoint()
        bbox = bbox2(masks[-1] > 0)
        if bbox is None:
            bbox = [0, 0, 0, 0]
        # Write outputs in {1: {'class': 0, 'bbox': [0, 0, 0, 0], 'score': ''}} format
        print(json.dumps({i: {label: {'class': 0, 'bbox': bbox, 'score': ''}}}))
    else:
        print(json.dumps({i: {label: {'class': 0, 'bbox': [0,0,0,0], 'score': ''}}}))

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
# if args.silent:
#     return_stdout()

# # Write outputs to json
# Path(args.output).open('w').write(json.dumps({
#     'results': outputs
# }, indent=4))

# print(json.dumps({
#     'results': outputs
#     }, indent=4)
# )
