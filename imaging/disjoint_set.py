class Node(object):
    def __init__(self, parent, rank=0, size=1):
        self.parent = parent
        self.rank = rank
        self.size = size


class DisjointSet(object):
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.nodes = [Node(s) for s in range(num_nodes)]

    def find(self, vertex_id):
        origin_vertex_id = vertex_id
        while vertex_id != self.nodes[vertex_id].parent:
            vertex_id = self.nodes[vertex_id].parent
        self.nodes[origin_vertex_id].parent = vertex_id
        return vertex_id

    def union(self, s1, s2):
        if self.nodes[s1].rank > self.nodes[s2].rank:
            self.nodes[s2].parent = s1
            self.nodes[s1].size += self.nodes[s2].size
        else:
            self.nodes[s1].parent = s2
            self.nodes[s2].size += self.nodes[s1].size

            if self.nodes[s1].rank == self.nodes[s2].rank:
                self.nodes[s2].rank += 1

        self.num_nodes -= 1

    def subset_size(self, vertex_id):
        return self.nodes[vertex_id].size

