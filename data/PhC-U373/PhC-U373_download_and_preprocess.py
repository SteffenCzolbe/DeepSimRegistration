import urllib.request
from zipfile import ZipFile
from PIL import Image
import numpy as np
import os
import shutil


def folder_to_tif_stack(folder, tif_out, crop_at_one=False):
    os.makedirs(os.path.dirname(tif_out), exist_ok=True)
    files = sorted(os.listdir(folder))
    images = [Image.open(os.path.join(folder, f)) for f in files]
    # crop to size
    images = list(map(lambda x: x.crop((4, 4, 692, 516)), images))
    # clean up seg masks
    if crop_at_one:
        # map the individual segmentation masks to one class for all cells
        images = list(
            map(lambda img: Image.fromarray(np.clip(np.array(img), 0, 1)), images)
        )
    images[0].save(tif_out, save_all=True, append_images=images[1:])


# download
url = "http://data.celltrackingchallenge.net/training-datasets/PhC-C2DH-U373.zip"
tmp_file = "./PhC-C2DH-U373.zip"
urllib.request.urlretrieve(url, tmp_file)

# unzip
with ZipFile(tmp_file, "r") as zipObj:
    zipObj.extractall(".")

# create tiff of intensity images
folder_to_tif_stack("./PhC-C2DH-U373/01/", "./images/01.tif")
folder_to_tif_stack("./PhC-C2DH-U373/02/", "./images/02.tif")
folder_to_tif_stack(
    "./PhC-C2DH-U373/01_ST/SEG/", "./labels-class/01.tif", crop_at_one=True
)
folder_to_tif_stack(
    "./PhC-C2DH-U373/02_ST/SEG/", "./labels-class/02.tif", crop_at_one=True
)

# cleanup
os.remove(tmp_file)
shutil.rmtree("./PhC-C2DH-U373")

