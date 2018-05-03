import time
from random import random

import numpy as np
import matplotlib.image as mpimg
import cv2
from scipy.ndimage.filters import gaussian_filter

from imaging.graph import vertex_id, build_graph
from imaging.disjoint_set import DisjointSet


def weight_diff_rgb(img, x1, y1, x2, y2):
    r = (img[y1, x1, 0] - img[y2, x2, 0]) ** 2
    g = (img[y1, x1, 1] - img[y2, x2, 1]) ** 2
    b = (img[y1, x1, 2] - img[y2, x2, 1]) ** 2
    return np.sqrt(r + g + b)


def weight_diff_gray(img, x1, y1, x2, y2):
    v = (img[y1, x1] - img[y2, x2]) ** 2
    return np.sqrt(v)


def default_thresh_meth(component_size, k):
    return k / component_size


"""
Efficient Graph-Based Image Segmentation - see http://cs.brown.edu/~pff/papers/seg-ijcv.pdf
The input is a graph G = (V, E), with n vertices and m edges.
The output is a segmentation of V into components S = (C1, ..., Cr).
"""
def segment_graph(graph, num_vertices, k, threshold_meth):
    # 0. Sort E into π = (o1, ..., om), by non-decreasing edge weight.
    sorted_edges = sorted(graph.edges, key=lambda e: e.w)

    # 1. Start with a segmentation S0 , where each vertex v i is in its own component.
    disjoint_set = DisjointSet(num_vertices)

    threshold = [threshold_meth(1, k)] * num_vertices
    # 2. Repeat step 3 for q = 1, . . . , m.
    for edge in sorted_edges:
        # 3. Construct S q given S q−1 (for more details see the original paper)
        s1 = disjoint_set.find(edge.v1)
        s2 = disjoint_set.find(edge.v2)

        if s1 != s2 and edge.w <= threshold[s1] and edge.w <= threshold[s2]:
            disjoint_set.union(s1, s2)
            s1 = disjoint_set.find(s1)
            threshold[s1] = edge.w + threshold_meth(disjoint_set.nodes[s1].size, k)

    # 4. Return S = S m .
    return disjoint_set, sorted_edges


def remove_small_components(disjoint_set, sorted_edges, min_size):
    for edge in sorted_edges:
        s1 = disjoint_set.find(edge.v1)
        s2 = disjoint_set.find(edge.v2)

        if s1 != s2 and (disjoint_set.subset_size(s1) < min_size or disjoint_set.subset_size(s2) < min_size):
            disjoint_set.union(s1, s2)

    return disjoint_set


def segment_image(image, weight_meth, k, threshold_meth, component_min_size):
    w, h = image.shape[1], image.shape[0]

    start_time = time.time()
    graph = build_graph(image, weight_meth)
    spent_time = time.time() - start_time
    print('build_graph spent time: {0}'.format(spent_time))

    start_time = time.time()
    disjoint_set, sorted_edges = segment_graph(graph, w*h, k, threshold_meth)
    spent_time = time.time() - start_time
    print('segment_graph spent time: {0}'.format(spent_time))

    start_time = time.time()
    #disjoint_set = remove_small_components(disjoint_set, sorted_edges, component_min_size)
    spent_time = time.time() - start_time
    print('remove_small_components spent time: {0}'.format(spent_time))

    return disjoint_set


def generate_disjointset_image(disjoint_set, width, height):
    colors = [(random(), random(), random()) for i in range(width*height)]

    img = np.zeros((height, width, 3))
    for y in range(height):
        for x in range(width):
            comp = disjoint_set.find(vertex_id(img, x, y))
            img[y, x] = colors[comp]

    return img


if __name__ == '__main__':
    #img = mpimg.imread('../test_images/test1.jpg')
    img = mpimg.imread('../test_images/beach.gif')
    #img = mpimg.imread('../test_images/grain.gif')

    img = img / 255.

    sigma = 0.5
    #img = gaussian_filter(img, sigma)
    img = cv2.GaussianBlur(img, (5, 5), sigma)
    #mpimg.imsave('../result-gaussian.png', img)

    k = 500
    component_min_size = 50

    disjoint_set = segment_image(img, weight_diff_rgb, k, default_thresh_meth, component_min_size)

    result_img = generate_disjointset_image(disjoint_set, img.shape[1], img.shape[0])

    mpimg.imsave('../result1.png', result_img)
    #mpimg.imsave('../result2.png', result_img)

