import random

from anytree import Node, LevelOrderIter, PreOrderIter
import json

TelAviv_json_dir = 'C:\\Users\\USER\\PycharmProjects\\privacy_parking\\tel_aviv.json'


class Location:
    def __init__(self, index, parent=None, name=None):
        self.index = index
        self.name = name
        self.node = Node(name, parent=parent)


def build_sub_tree(dictionary, parent=None, count_till_now=0):
    node = Node(dictionary['Name'], parent, index=count_till_now)
    count_till_now += 1
    # print(count_till_now)
    for sub_dict in dictionary['sub_areas']:
        count_till_now, _ = build_sub_tree(sub_dict, node, count_till_now)
    return count_till_now, node


class LocationsTaxonomy:
    def __init__(self, json_file):
        with open(json_file, 'r') as f:
            self.json_dict = json.load(f)
        self.max_index, self.root_node = build_sub_tree(self.json_dict)
        # for pre, fill, node in RenderTree(self.root_node):
        #     print("%s%s" % (pre, node.name))
        leaves_nodes = [node for node in LevelOrderIter(self.root_node) if len(node.children) == 0]
        self.leaves_enumeration = {node.index: i for (i, node) in enumerate(leaves_nodes)}

    def get_random_node(self, height=-1):
        height_nodes = [node for node in LevelOrderIter(self.root_node) if (height == -1 or height == node.height)]
        if len(height_nodes) == 0:
            print('not enough nodes in this height')
            return None
        return random.choice(height_nodes)

    def get_paths_from_root(self):
        return [list(leaf.path) for leaf in PreOrderIter(self.root_node, filter_=lambda node: node.is_leaf)]

    def get_leaves_enumerated(self, root_node: Node):
        leaves_nodes = self.get_leaves(root_node)
        return [(node.index, node) for node in leaves_nodes]

    def get_random_leaf(self, root_node: Node):
        leaves_nodes = self.get_leaves_enumerated(root_node)
        index, rand_node = random.choice(leaves_nodes)
        return index, rand_node

    @staticmethod
    def get_leaves(root_node: Node):
        leaves_nodes = [node for node in LevelOrderIter(root_node) if len(node.children) == 0]
        return leaves_nodes

    @staticmethod
    def get_number_leaves(root_node: Node):
        leaves_nodes = LocationsTaxonomy.get_leaves(root_node)
        return len(leaves_nodes)

if __name__ == "__main__":
    random.seed(1)
    location_taxonomy = LocationsTaxonomy(TelAviv_json_dir)
    print(location_taxonomy.get_random_node())
