
from wholeslidedata.annotation.structures import Point
import numpy as np
import cv2
from matplotlib import pyplot as plt

from .helpers import get_patch


def get_3p_transform(annotations1, annotations2, spacing, indices=[0,1,2]):
    scale = 1/spacing*0.25

    # We'll only add control points and not rectangles to the HE slides but just exclude rectangles anyway
    control_points1 = np.float32([a.coordinates for a in annotations1 if isinstance(a, Point)]) * scale
    control_points2 = np.float32([a.coordinates for a in annotations2 if isinstance(a, Point)]) * scale
    
    # Point set registration with opencv
    return cv2.getAffineTransform(control_points1[indices], control_points2[indices])


default = {
    "affine": np.array([
        [1.,0.,0.],
        [0.,1.,0.]
    ]),
    "perspective": np.array([
        [1.,0.,0.],
        [0.,1.,0.],
        [0.,0.,1.]
    ])
}

def get_transformation_matrix(patch1, patch2, n_features=5000, warp="affine", plotting=False):
    # Convert to grayscale.
    img1 = cv2.cvtColor(patch1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(patch2, cv2.COLOR_BGR2GRAY)

    # make p53 harsher colored
    img1 = ((img1/img1.max())**6 * 256).astype(np.uint8)
    img2 = ((img2/img2.max())**4 * 256).astype(np.uint8)

    if plotting:
        fig, ax = plt.subplots(1,2,figsize=(20,20))
        ax[0].imshow(img1, cmap="gray")
        ax[1].imshow(img2, cmap="gray")
    
    # Create ORB detector with 5000 features.
    orb_detector = cv2.ORB_create(n_features)
    
    # Find keypoints and descriptors.
    # The first arg is the image, second arg is the mask
    #  (which is not required in this case).
    kp1, d1 = orb_detector.detectAndCompute(img1, None)
    kp2, d2 = orb_detector.detectAndCompute(img2, None)

    if type(d1)==type(None) or type(d2)==type(None):
        print(f"Couldn't find keypoint descriptors ({warp})")
        return default[warp], "no descriptors"
    
    # Match features between the two images.
    # We create a Brute Force matcher with 
    # Hamming distance as measurement mode.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    
    # Match the two sets of descriptors.
    matches = matcher.match(d1, d2)
    
    # Sort matches on the basis of their Hamming distance.
    list(matches).sort(key = lambda x: x.distance)
    
    # Take the top 90 % matches forward.
    matches = matches[:int(len(matches)*0.9)]
    no_of_matches = len(matches)
    
    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))
    
    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt

    if len(p1) == 0 or len(p1)!=len(p2):
        print(f"lengths of points not okay (p1: {len(p1)}, p2: {len(p2)})")
        return default[warp], "no keypoint matches"
    
    # Find the homography matrix.
    message = "success"
    if warp == "perspective":
        transform, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
    elif warp == "affine":
        transform, mask = cv2.estimateAffine2D(p1, p2, cv2.RANSAC)
        if type(transform)==type(None):
            print(f"couldn't estimate affine matrix")
            transform = default[warp]
            message = "couldn't estimate affine matrix"
    else:
        print(f"warp {warp} not found")
        return None, f"warp {warp} not found"
    
    return transform, message


def align(patch1, patch2, n_features=5000, warp="affine"):
    """Return patch1 transformed
    https://www.geeksforgeeks.org/image-registration-using-opencv-python/"""  
    height, width = patch2.shape[:2]
    transform, message = get_transformation_matrix(patch1, patch2, n_features, warp=warp)
    determinant = np.linalg.det(transform[:2,:2])

    # Use this matrix to transform the
    # colored image wrt the reference image.
    if warp == "perspective":
        return cv2.warpPerspective(patch1,
                            transform, (width, height)), message, determinant
    elif warp == "affine":
        return cv2.warpAffine(patch1,
                            transform, (width, height)), message, determinant


def get_align_transform(wsi1, wsi2, outline1, outline2, spacing=2.0, n_features=5000, warp="affine", plotting=False):
    patch1 = get_patch(wsi1, outline1, spacing)
    patch2 = get_patch(wsi2, outline2, spacing)

    transform, message = get_transformation_matrix(patch1, patch2, n_features, warp, plotting=plotting)

    if transform.shape[0] == 2:
        transform = np.concatenate([transform, np.array([[0,0,1]])])

    xy1 = outline1.min(axis=0)
    xy2 = outline2.min(axis=0)
    scale = 1/spacing*0.25

    correction_post = np.array([
        [1,0,xy2[0]],
        [0,1,xy2[1]],
        [0,0,1]
    ]) @ np.array([
        [1/scale,0,0],
        [0,1/scale,0],
        [0,0,1]
    ])
    correction_pre = np.array([
        [scale,0,0],
        [0,scale,0],
        [0,0,1]
    ]) @ np.array([
        [1,0,-xy1[0]],
        [0,1,-xy1[1]],
        [0,0,1]
    ])
    return (correction_post @ transform @ correction_pre)[:2], message, np.linalg.det(transform[:2,:2])
    

"""Detect whether the bounding box is still covered by the warped image"""

def get_warped_corners(img, homography_matrix):
    # Get the dimensions of the image
    height, width = img.shape[:2]

    # Get the corners of the original image
    corners = np.array([[0, 0], [0, height], [width, height], [width, 0]], dtype=np.float32)

    # Transform the corners of the original image to the warped image
    warped_corners = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), homography_matrix).reshape(-1, 2)

    return warped_corners


def covers_bounding_box(img, homography_matrix, bounding_box_size):
    # Get the dimensions of the image
    height, width = img.shape[:2]
    
    # Check if the bounding box is still covered by the warped image
    # Get the corners of the warped image
    warped_corners = get_warped_corners(img, homography_matrix)

    # Get the corners of the bounding box
    central_w, central_h = bounding_box_size
    central_x = (width - central_w) // 2
    central_y = (height - central_h) // 2
    bb_corners = [(central_x, central_y), (central_x, central_y + central_h),
                (central_x + central_w, central_y + central_h), (central_x + central_w, central_y)]
    
    # Get the edges of the bounding box and warped image
    bb_edges = [(bb_corners[i], bb_corners[(i + 1) % 4]) for i in range(4)]
    wc_edges = [(warped_corners[i], warped_corners[(i + 1) % 4]) for i in range(4)]

    # Check if any of the bounding box edges intersect with any of the warped image edges
    if all(not intersect(e1, e2) for e1 in bb_edges for e2 in wc_edges):
        # Check if the center of the bounding box is inside the warped image
        return cv2.pointPolygonTest(warped_corners, (central_x + central_w // 2, central_y + central_h // 2), False) >= 0
    return False


def intersect(l1, l2):
    x1, y1 = l1[0]
    x2, y2 = l1[1]
    x3, y3 = l2[0]
    x4, y4 = l2[1]

    # Check if the lines are parallel
    # den is the denominator of ua and ub
    den = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if den == 0:
        return False
    
    # Check if the lines intersect
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / den
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / den

    # Check if the intersection is within the line segments
    return 0 <= ua <= 1 and 0 <= ub <= 1
