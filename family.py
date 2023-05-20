#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('reload_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import torch
if torch.cuda.is_available():
    print("Using GPU")
else:
    print("Using CPU, this will be slow.")

from cli import *

print("Woo hoo. Let's go!")

# args, defined in track_anything.py
# args = parse_argument()
# args = default_args()
args = argparse.Namespace()
args.input = Path("test_sample/family_480.mp4")
args.track_data = Path("test_sample/family_480/blue_dress_lady_face.json")
# args.device = "cuda" if torch.cuda.is_available() else "cpu"
args.device = "cpu"
args.sam_model_type = "vit_b"
args.output = Path("output.json")
args.debug = False
args.mask_save = False
args.output_video = Path("result.mp4")
args.track_data = json.load(open(args.track_data, "r"))
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
    "track_end_number": args.track_data["track_end_number"],
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


from matplotlib import pyplot as plt


# In[7]:


points = args.track_data['points']

template_frame, video_state, interactive_state, run_status=select_template(
    model,
    points[0]['frame'], 
    video_state, 
    interactive_state
)


# In[8]:


evt = argparse.Namespace()
evt.index = [0, 0]
template_frame, video_state, interactive_state, run_status = sam_refine(
    model=model,
    video_state=video_state,
    # point_prompt=sam_refine_args['point_prompt'],
    point_prompt=None,#"Positive",
    click_state=None,#[[180,176],[1]],
    prompt={
        "prompt_type": ["click"],
        "input_point": [points[0]['pos']],#[[180,176]],
        "input_label": [points[0]['label']],
        "multimask_output": "False",
    },
    interactive_state=interactive_state,
    evt=evt,
)


# In[9]:


video_output, video_state, interactive_state, run_status = vos_tracking_video(
    model=model,
    video_output=args.output_video,
    video_state=video_state,
    interactive_state=interactive_state,
    mask_dropdown=[],
)
outputs = []


# In[10]:


def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    try:
        rmin, rmax = np.where(rows)[0][[0, -1]].tolist()
        cmin, cmax = np.where(cols)[0][[0, -1]].tolist()
        return rmin, rmax, cmin, cmax
    except IndexError:
        return None
# video_state["masks"][video_state["select_frame_number"]] = mask
for frame_num, mask in enumerate(video_state["masks"]):
    # print(mask)
    # print(i)
    # mask = np.load(mask)
    # Get bounding box [x,y,x,y] from binary mask
    bbox = bbox2(mask > 0)
    if bbox is not None:
        # Write outputs in {1: {'class': 0, 'bbox': [0, 0, 0, 0], 'score': ''}} format
        outputs.append({frame_num: {'class': 0, 'bbox': bbox, 'score': ''}})

# Write outputs to json
Path(args.output).open('w').write(json.dumps({
    'results': outputs
}, indent=4))

