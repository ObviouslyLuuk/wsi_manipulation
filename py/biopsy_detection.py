import numpy as np
import cv2



"""
0/445 has a good example of a biopsy that should be split
1/445 has a nice positive control with edge artifacts, and also good biopsy examples
5/445 has good positive control example with lots of issues, 
    also artifacts that are seen as biopsies
12/445 has a lot of the same biopsies in different areas
15/445 has an air bubble artifact
16/445 has example of positive control that's small enough to be a biopsy
    also an artifact on top of biopsies
27/445 another example of artifact seen as biopsy
28/445 good example of too small correctly ignored
30/445 slide without white background, lots of double biopsies
31/445 pink artifact seen as biopsy
44/445 another example of positive control that's small enough to be a biopsy
45/445 slide that has limited spacing
46/445 star artifacts correctly ignored
48/445 one biopsy ignored because it's too small
61/445 air bubble artifact seen as biopsy
73/445 biopsy ignored because it's within another's area

    
"""



###############################################################################
# MASKING
###############################################################################
def get_edge_pixel_indices(img):
    '''
    Returns an array of edge pixel coordinates for a given image.

    Parameters:
        img: np.ndarray of shape (n, m, 3)
            The image to get the edge pixels from.

    Returns:
        edge_pixels: np.ndarray of shape (n, 2)
    '''
    edge_pixels = np.array([np.array([i, 0]) for i in range(img.shape[0])] + \
                       [np.array([img.shape[0]-1, i]) for i in range(img.shape[1]-1, 0, -1)] + \
                       [np.array([i, img.shape[1]-1]) for i in range(img.shape[0]-1, 0, -1)] + \
                       [np.array([0, i]) for i in range(img.shape[1]-1, 0, -1)]).astype(np.uint16)
    return edge_pixels


def get_img_edges(img):
    '''
    Returns the image with only the edges.

    Parameters:
        img: np.ndarray of shape (n, m, 3)
            The image to get the edges from.

    Returns:
        edges: np.ndarray of shape (n, m, 3)
    '''
    indices = get_edge_pixel_indices(img)
    return img[indices[:, 0], indices[:, 1]]


def get_most_common_color(img):
    '''
    Returns the most common color in a given image.

    Parameters:
        img: np.ndarray of shape (n, m, 3)
            The image to get the most common color from.

    Returns:
        color: np.ndarray of shape (3,)
    '''
    colors, counts = np.unique(img.reshape(-1, 3), axis=0, return_counts=True)
    return colors[np.argmax(counts)]


def get_similar_color_mask(img, color, threshold=0.1):
    '''
    Returns a mask of the pixels in the image that are similar to the given color.

    Parameters:
        img: np.ndarray of shape (..., 3)
            The image to get the similar color mask from.
        color: np.ndarray of shape (3,)
            The color to compare the pixels to.
        threshold: int or float
            The threshold for the color similarity.

    Returns:
        mask: np.ndarray of shape (n, m)
    '''
    # Assert that img, color and threshold are of the correct and compatible type (int VS float)
    if type(threshold) == int:
        assert img.dtype == color.dtype == np.uint8, "Image and color must be of type np.uint8 when threshold is of type int, but are of type {} and {}.".format(img.dtype, color.dtype)
    elif type(threshold) == float:
        assert img.dtype == color.dtype == np.float32, "Image and color must be of type np.float32 when threshold is of type float, but are of type {} and {}.".format(img.dtype, color.dtype)

    return np.linalg.norm(img - color, axis=-1) < threshold


def get_background_mask(img, threshold=0.1, not_background_color=None, only_edges=False):
    '''
    Returns a mask of the pixels in the image that are not background.
    Whatever is the most common color in the image is considered the background color.

    Parameters:
        img: np.ndarray of shape (n, m, 3)
            The image to get the background mask from.
        threshold: int or float
            The threshold for the color similarity.
                default: 0.1 (float) or 26 (int)
        not_background_color: np.ndarray of shape (3,)
            If given, this will be excluded as a contender for the background color.
        only_edges: bool
            If True, only the edges will be considered for the most common color.
    
    Returns:
        mask: np.ndarray of shape (n, m)
            The background mask (True for foreground, False for background)
        background_color: np.ndarray of shape (3,)
            The background color.
    '''
    # Assert that img, color and threshold are of the correct and compatible type (int VS float)
    if type(threshold) == int:
        assert img.dtype == np.uint8, "Image must be of type np.uint8 when threshold is of type int."
    elif type(threshold) == float:
        assert img.dtype == np.float32, "Image must be of type np.float32 when threshold is of type float."

    # Get the most common color in the image, from the pixels we are considering
    contender_img = get_img_edges(img) if only_edges else img
    if not_background_color is not None:
        contender_img = contender_img[~get_similar_color_mask(contender_img, not_background_color, threshold)]
    background_color = get_most_common_color(contender_img)

    return ~get_similar_color_mask(img, background_color, threshold), \
        background_color


def get_contours(mask, closing_iterations=1):
    '''
    Returns the contours of a given mask.

    Parameters:
        mask: np.ndarray of shape (n, m)
            The mask to get the contours from.
        closing_iterations: int
            The number of closing iterations to do on the mask.
    
    Returns:
        contours: np.ndarray of shape (NUM_CONTOURS, NUM_POINTS, 2)
            The contours of the mask.
        hierarchy: np.ndarray of shape (NUM_CONTOURS, 4)
            The hierarchy of the contours. 
            (Next, Previous, First_Child, Parent)
        bounding_rects: list of len NUM_CONTOURS of tuples of shape (4,)
            The bounding rectangles of the contours.
            (x, y, w, h) where x and y are the top left corner coordinates
        mask_closed: np.ndarray of shape (n, m)
            The mask after closing.
    '''
    # Do closing on the mask
    kernel = np.ones((5, 5), np.uint8)
    mask_closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=closing_iterations)

    # Get the contours of the mask
    contours, hierarchy = cv2.findContours(mask_closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = [c[:,0,:] for c in contours]
    hierarchy = hierarchy[0]
    return contours, hierarchy, [cv2.boundingRect(c) for c in contours], mask_closed




###############################################################################
# CONTOURS
###############################################################################
def get_interpolations(p1, p2, weight):    
    '''
    Returns point in between two points, or an array of points in between 
    two arrays of points. If weight is 0, the first point is returned, if
    weight is 1, the second point is returned, and if weight is 0.5, the
    point halfway in between the two points is returned.

    Parameters:
        p1: np.ndarray of shape (2,) or (n, 2)
            The first point or array of points.
        p2: np.ndarray of shape (2,) or (n, 2)
            The second point or array of points.
        weight: float
            The weight of the interpolation.

    Returns:
        interpolation: np.ndarray of shape (2,) or (n, 2)
            The interpolated point or array of points.
    '''
    return p1 + (p2 - p1) * weight


def sample_points_in_contour_area(contour, n, distance_from_edge):
    '''
    Returns n points sampled from the contour area, 
    with a relative distance from the edge.

    Parameters:
        contour: np.ndarray of shape (n, 2)
            The contour surrounding the area to sample points from.
        n: int
            The number of points to sample.
        distance_from_edge: int or float
            The relative distance from the edge to sample the points from.
            0 is the edge, 1 is the center of the contour.

    Returns:
        points: np.ndarray of shape (n, 2)
            The points sampled from the contour area.
    '''
    # Get the center of the contour
    center = np.mean(contour, axis=0)

    # Get random set of contour points of size n
    n = min(n, len(contour))
    contour = np.random.choice(contour, size=n, replace=False)

    # Get the points in between the contour points and the center
    return get_interpolations(contour, center, distance_from_edge).astype(np.uint32)
    

def contour_area_is_masked(mask, contour, distance_from_edge=0.5, validate=True):
    '''
    Returns True if the contour area is masked, False otherwise.

    Parameters:
        mask: np.ndarray of shape (n, m)
            The image to get the contour area color from.
        contour: np.ndarray of shape (n, 2)
            The contour surrounding the area to sample points from.
            A contour is a list of points, where each point is a list of two coordinates.
    
    Returns:
        masked: bool
            True if the contour area is masked, False otherwise.
    '''
    # Assert that the mask is of the correct type
    assert mask.dtype == np.uint8, "Mask must be of type np.uint8."
    assert mask.max() == 1, "Mask must be binary."

    # Get the points in the contour area
    contour_area_points = sample_points_in_contour_area(contour, len(contour), distance_from_edge)

    # Filter out points outside the contour area
    if validate:
        contour_area_points = contour_area_points[np.where(cv2.pointPolygonTest(contour, contour_area_points, False) >= 0)]

    # Get the color of the contour area
    contour_area_color = np.mean(mask[contour_area_points[:, 0], contour_area_points[:, 1]])

    # Return True if the contour area is masked, False otherwise
    # It is masked if the color is 0, which means it is black
    return contour_area_color < 0.5




###############################################################################
# CONTOUR SPLITTING
###############################################################################
def split_contour(contour, split_idx1, split_idx2):
    '''
    Splits a contour into two closed contours at the given
    split index pair. The split index pair is a pair of indices
    that are the indices of the points in the contour that
    should be connected to form the split.

    Parameters:
        contour: np.ndarray of shape (n, 2)
            The contour to split.
        split_idx1: int
            The index of the first point in the contour to split.
        split_idx2: int
            The index of the second point in the contour to split.

    Returns:
        contour1: np.ndarray of shape (n, 2)
            The first contour.
        contour2: np.ndarray of shape (n, 2)
            The second contour.
    '''
    assert type(split_idx1) == type(split_idx2) == int, "Split index pair must be of type int."
    assert split_idx1 < split_idx2, "Split index 1 must be smaller than split index 2."

    indices_between = np.arange(split_idx1, split_idx2)
    indices_outside = np.concatenate([np.arange(0, split_idx1), np.arange(split_idx2, len(contour))])

    # Indices between are the first contour
    contour1 = contour[indices_between]

    # Indices outside are the second contour
    contour2 = contour[indices_outside]

    return contour1, contour2



# TODO: Try the idea where for each point pair you calculate euclidian distance
    # but also contour index distance, one should be low, one high
    # normalize them, and look at distribution to see if a split is likely
    # Then if it is you take the smallest euclidian distance and split there