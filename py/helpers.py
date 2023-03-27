import os
import numpy as np
from wholeslidedata.annotation.structures import Polygon

ROOT = r'L:\\basic\\divg\\PATH-COMPAI\\datasets\\Barrett\\Barrett ASL 21-11-22\\'

def get_all_annotated_cases(root=ROOT):
    dir = os.path.join(root, "annotated")

    filepaths = {nm.split('_')[0]:{"HE":{},"p53":{}} for nm in os.listdir(dir)}
    for nm in os.listdir(dir):
        case = nm.split('_')[0]
        if "he" in nm.lower():
            coupe = "HE"
        elif "p53" in nm.lower():
            coupe = "p53"
        else:
            print("coupe not found")
            return

        if ".tiff" in nm:
            typ = "wsi"
        elif ".xml" in nm:
            typ = "wsa"
        else:
            print("filetype wrong")
            return

        filepaths[case][coupe][typ] = os.path.join(dir, nm)
    return filepaths


def get_outlines(wsa):
    return [a.coordinates for a in wsa.annotations if isinstance(a, Polygon)]


def get_area(outline, spacing, factor=0.25):
    min = outline.min(axis=0) # min (x, y)
    max = outline.max(axis=0) # max (x, y)
    x, y = tuple((min+max)/2)
    w, h = tuple((max-min)*factor/spacing)
    return x, y, w, h


def get_patch(wsi, biopsy, spacing):
    return wsi.get_patch(*get_area(biopsy, spacing), spacing)


def get_sub_areas(area, size=256, padding=0, factor=0.25, spacing=2.0):
    x, y, w, h = area
    factored_size = size/factor*spacing
    x = x - (w/factor*spacing)/2 + factored_size/2
    y = y - (h/factor*spacing)/2 + factored_size/2
    sub_areas = []
    for i in range(round(h//size)+1):
        sub_areas.append([])
        for j in range(round(w//size)+1):
            sub_areas[i].append((x+j*factored_size, y+i*factored_size, size+padding, size+padding))
    return sub_areas


def patch_empty(patch):
    return patch.mean() > 223


def concat_one(points):
    return np.concatenate((points,np.ones((len(points),1))),axis=1)


