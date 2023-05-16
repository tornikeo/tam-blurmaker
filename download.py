from PIL import Image
import gdown
import numpy as np
import os
from pathlib import Path
import requests

# try:
#     from mmcv.cnn import ConvModule
# except:
#     os.system("mim install mmcv")


# download checkpoints
def download_checkpoint(url, folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    if not os.path.exists(filepath):
        print("download checkpoints ......")
        response = requests.get(url, stream=True)
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print("download successfully!")

    return filepath


def download_checkpoint_from_google_drive(file_id, folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    if not os.path.exists(filepath):
        print(
            "Downloading checkpoints from Google Drive... tips: If you cannot see the progress bar, please try to download it manuall \
              and put it in the checkpointes directory. E2FGVI-HQ-CVPR22.pth: https://github.com/MCG-NKU/E2FGVI(E2FGVI-HQ model)"
        )
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filepath, quiet=False)
        print("Downloaded successfully!")

    return filepath


if __name__ == "__main__":
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
    sam_checkpoint = SAM_checkpoint_dict['vit_b']
    sam_checkpoint_url = SAM_checkpoint_url_dict["vit_b"]
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