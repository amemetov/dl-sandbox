class Edge(object):
    def __init__(self, v1, v2, w):
        self.v1 = v1
        self.v2 = v2
        self.w = w


class Graph(object):
    def __init__(self):
        self.edges = []

    def add_edge(self, v1, v2, edge_weight):
        self.edges.append(Edge(v1, v2, edge_weight))

    def num_edges(self):
        return len(self.edges)


class ImageGraph(Graph):
    def __init__(self, image, weight_meth):
        super().__init__()
        self.image = image
        self.width = image.shape[1]
        self.weight_meth = weight_meth

    def add_pixels_edge(self, x1, y1, x2, y2):
        v1 = vertex_id(self.image, x1, y1)
        v2 = vertex_id(self.image, x2, y2)
        weight = self.weight_meth(self.image, x1, y1, x2, y2)
        self.add_edge(v1, v2, weight)


def vertex_id(img, x, y):
    width = img.shape[1]
    return x + y * width


"""
weight_meth is a method with the signature (image, x1, y1, x2, y2):float
"""
def build_graph(image, weight_meth):
    w, h = image.shape[1], image.shape[0]

    graph = ImageGraph(image, weight_meth)

    for y in range(h):
        for x in range(w):
            if x > 0:
                graph.add_pixels_edge(x, y, x - 1, y)

            if y > 0:
                graph.add_pixels_edge(x, y, x, y - 1)

            # if x > 0 and y > 0:
            #     graph.add_pixels_edge(x, y, x - 1, y - 1)
            #
            # if x > 0 and y < h - 1:
            #     graph.add_pixels_edge(x, y, x - 1, y + 1)

    return graph