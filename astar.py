"""
A* Path Finding Algorithm implementation.

Project created for a final assignment on Mathematical Modelling & Big Data (AM6020) module
@ UCC - MSc in Bioinformatics & Computational Biology.

Copyright shared between: Iulian Ichim & Diarmaid Nagle
"""


from __future__ import annotations
from typing import Generator

import numpy
import numpy as np
import sys
import time
import math
import matplotlib.pyplot as plt


class Graph:
    def __init__(self, graph_array: [np.ndarray | list[list]] = None, num_nodes: int = None, saturation: float = None,
                 edge_max: int = 10, from_file: bool = False, debug: bool = False):
        """
        Except 'graph_array' all arguments are passed only if the user wants the graph to be randomly generated.
        If no arguments besides 'from_file' are passed, the class expects that `read_from_edge_list` will be called.
        :param graph_array: If not provided, it will create a randomly generated array of nodes. Any 2D array ca be
        passed. The Class accepts NumPy Array or a Nested List.
        :param num_nodes: Number of nodes to randomly generate
        :param saturation: saturation factor to determine how many edges a node has
        :param edge_max: Maximum edge weight, default is 10
        :param debug: If set to True, prints analysis information
        :param from_file: Read graph from given file.
        """
        self.debug = debug
        self.num_nodes = num_nodes
        self.edge_max = edge_max

        if isinstance(graph_array, list):
            self.origin_array = np.array(graph_array)
        elif isinstance(graph_array, np.ndarray):
            self.origin_array = graph_array

        elif not from_file:
            self.origin_array = self.random_array(num_nodes, saturation, edge_max)

        if not from_file:
            self.graph = self._parse_array()
        self.a_star_path = None

    def _parse_array(self):
        """
        Parses given original array and converts it to a Graph - Array of Nodes and Edges.
        :return: Graph Array
        """
        if self.debug:
            print('Converting array to graph...')
            start = time.time()

        node_dict = dict()

        # For each row in the array, create Node objects at that index
        for node in range(len(self.origin_array)):
            node_obj = Node()
            node_obj.value = node
            node_dict[node] = node_obj

        # each row is a node
        for node in range(len(self.origin_array)):
            # each column is that current index other node neighbours
            for neighbour in range(len(self.origin_array[node])):
                if self.origin_array[node, neighbour] != 0:
                    # Create an Edge object
                    edge = Edge()
                    edge.weight = self.origin_array[node, neighbour]

                    # the value at i, j in the matrix is the Edge weight
                    node_dict[node].neighbours[node_dict[neighbour]] = edge

        nodes = numpy.array([node for node in node_dict.values()], dtype=object)

        if self.debug:
            end = time.time()
            print(f"Converting the array took: {end - start}\n")

        return nodes

    def array_a_star(self, start_node: int, stop_node: int):
        """
        Array implementation of A* algorithm. Parses a given array as an actual array of integers and find the shortest
        path from start_node to end_node
        :param start_node: Start Node
        :param stop_node: End Node
        :return: Shortest path from Start Node to End Node
        """

        if stop_node > self.origin_array.shape[0]:
            raise ValueError('Stop node is larger than the number of nodes.')
        elif start_node < 0:
            raise ValueError('Start node is less than zero')

        # List of not visited nodes
        open_list: list[int | Node] = []

        # List of visited neighbours
        close_list: list[int | Node] = []

        # add the starting node to the open_list
        open_list.append(start_node)

        # keep a record of all g_scores
        g_scores = dict()
        g_scores[start_node] = 0

        # this stores our solution
        valid_path = dict()
        valid_path[start_node] = start_node

        while len(open_list) > 0:

            current_node = None

            # Since a "graph-like" network does not have a proper structure, such as a grid-like network,
            # there is no proper estimation of the heuristic. Therefore, the best practice is to underestimate
            # the heuristic, thus setting it to 1.
            h_score = 1

            # find the node in open_list with the smallest f-score
            for node in open_list:
                if current_node is None or g_scores[node] + h_score < g_scores[current_node] + h_score:
                    # current_node = min(open_list, key=lambda x: g_scores[x] + h_score)
                    current_node = node

            if self.debug:
                sys.stdout.write(
                    f'\rAnalyzing node {current_node}. h:{h_score},'
                    f'g:{g_scores[current_node]}, f:{g_scores[current_node] + h_score}')
                sys.stdout.flush()

            if current_node == stop_node:
                path = []

                # Parse all nodes added to the valid_path and recreate the final path
                while valid_path[current_node] != current_node:
                    path.append(current_node)
                    current_node = valid_path[current_node]

                path.append(start_node)

                path.reverse()

                self.a_star_path = path

                if self.debug:
                    print(f"\nFound final path {len(path)} steps long:\n{path}")

                return path

            current_neighbours = self._get_neighbours(current_node)

            for neighbour in current_neighbours:

                # If a neighbour is in neither list, add it to possible nodes ad possible path then recalculate its
                # g_score
                if neighbour not in open_list and neighbour not in close_list:
                    open_list.append(neighbour)
                    valid_path[neighbour] = current_node

                    g_scores[neighbour] = g_scores[current_node] + self.origin_array[current_node][neighbour]

                else:
                    # otherwise, set the next smallest neighbour as a valid path
                    if g_scores[neighbour] > g_scores[current_node] + self.origin_array[current_node][neighbour]:
                        g_scores[neighbour] = g_scores[current_node] + self.origin_array[current_node][neighbour]
                        valid_path[neighbour] = current_node

                        if neighbour in close_list:
                            close_list.remove(neighbour)
                            open_list.append(neighbour)

            open_list.remove(current_node)
            close_list.append(current_node)

    def graph_a_star(self, start_node: int, stop_node: int):
        """
        Node object implementation of A* algorithm. The difference from the previous algorithm is that this parses
        a collection of Node objects.
        :param start_node: start node
        :param stop_node: end node
        :return: Return shortest path from start_node to end_node
        """

        # find the Start Node in the graph and set it as start node
        start = self.graph[start_node]
        start.g_score = 0
        start.is_start = True

        # find the Stop Node and set it as stop node
        self.graph[stop_node].is_stop = True

        open_list = list()
        close_list = list()

        open_list.append(start)

        valid_path = dict()
        valid_path[start.value] = start

        standard_heuristic = 1

        start.h_score = standard_heuristic

        while len(open_list) > 0:

            # get the Node with the smallest f_score
            current_node = min(open_list, key=lambda x: x.g_score + x.h_score)

            if self.debug:
                sys.stdout.write(
                    f'\rAnalyzing {current_node}. h:{current_node.h_score},'
                    f'g:{current_node.g_score}, f:{current_node.h_score + current_node.g_score}')
                sys.stdout.flush()

            if current_node.is_stop:
                path = []

                while valid_path[current_node.value] != current_node:
                    path.append(current_node.value)
                    current_node = valid_path[current_node.value]

                path.append(start.value)

                path.reverse()

                self.a_star_path = path

                if self.debug:
                    print(f"\nFound final path {len(path)} steps long:\n{path}")

                return path

            for neighbour, edge in current_node.neighbours.items():

                if neighbour not in open_list and neighbour not in close_list:
                    open_list.append(neighbour)
                    valid_path[neighbour.value] = current_node

                    neighbour.h_score = standard_heuristic
                    neighbour.g_score = current_node.g_score + edge.weight

                else:
                    if neighbour.g_score > current_node.g_score + neighbour.g_score:
                        neighbour.h_score = standard_heuristic
                        neighbour.g_score = current_node.g_score + edge.weight
                        valid_path[neighbour.value] = current_node

                        if neighbour in close_list:
                            close_list.remove(neighbour)
                            open_list.append(neighbour)

            open_list.remove(current_node)
            close_list.append(current_node)

    def _get_neighbours(self, node: int) -> Generator:
        """
        Helper function to get the neighbours of the given node in the array.
        :param node: Node whose neighbours to get
        :return: Generator of neighbours of given node
        """
        current_node = self.origin_array[node]

        neighbours = []

        for x in range(0, len(current_node)):
            if current_node[x] != 0:
                neighbours.append(x)

        neighbours = (n for n in neighbours)

        return neighbours

    def read_edge_list(self, file, sep: str = '\t', invalid_node_char: str = '-',
                       default_edge: int = None, generator: bool = True):
        """
        Parse a file of edge lists i.e. node-to-node connections
        :param generator: If set to False it will convert the results to a list
        :param invalid_node_char: Char used when node is not present
        :param sep: Separator to use when splitting nodes, defaults to tab
        :param file: name of file or path to file
        :param default_edge: specify a edge for each connection. Default = 1
        :return: generator of edges
        """
        nodes_dict = dict()

        # keep track of a list of orphan nodes
        orphans = []

        with open(file, 'r') as ppi_file:

            # keep track of line number
            n = 0

            while True:

                line = ppi_file.readline()

                if line == '':
                    break

                # strip lines of any whitespace chars
                clean_line = line.strip()

                line_values = clean_line.split(sep)

                node = line_values[0]
                connection = line_values[1]

                if node == invalid_node_char:
                    orphans.append((connection, n))
                    n += 1
                    continue

                elif connection == invalid_node_char:
                    orphans.append((node, n))
                    n += 1
                    continue

                # Instantiate the Edge class
                if default_edge is not None:
                    edge = Edge()
                    edge.weight = default_edge
                else:
                    edge = Edge()
                    edge.weight = 1

                # if a node or connected node is not in the dictionary, instantiate a new node and add it
                #   doing this avoid making duplicate objects for the same nodes
                if node not in nodes_dict:
                    node_obj = Node()
                    node_obj.value = node
                    nodes_dict[node] = node_obj

                if connection not in nodes_dict:
                    connection_obj = Node()
                    connection_obj.value = connection
                    nodes_dict[connection] = connection_obj

                # add the nodes and the connection to each-others neighbours
                # this avoid having cases where connections would have 0 neighbours
                # since they are not found in the first column, but still have a neighbour
                nodes_dict[node].neighbours[nodes_dict[connection]] = edge
                nodes_dict[connection].neighbours[nodes_dict[node]] = edge

                n += 1

        # for memory purposes, if this method of parsing is chosen, the graph is a generator since
        # PPI files can have millions of values which translate to tens of thousands of nodes
        if generator:
            self.graph = (n for n in nodes_dict.values())
        else:
            self.graph = [n for n in nodes_dict.values()]

    def random_array(self, num_nodes, saturation, edge_max):
        """
        The number of nodes is given by 'num_nodes'. Each node is connected to between 1 and 'num_nodes' - 1 other
        nodes. The 'saturation' controls the proportion of other nodes that each node connects to: 0 indicates
        connection to only 1 other node (to prevent orphan nodes) and 1 indicates connection to all other nodes. The
        maximum edge weight is defined by 'edge_max', default = 10. Returns an array where each row represents a node
        and each column represents another node that it could be connected to. 0 indicates no connection,
        any other number indicates the weight of the edge between the specified nodes
        """

        if self.debug:
            start = time.time()
            print('Generating array...')

        if saturation < 0 or saturation > 1:
            raise ValueError(f"Saturation must be between 0 and 1. Value provided:{saturation}")

        sat_per_mil = round(10000 * saturation)
        new_array = np.zeros((num_nodes, num_nodes))  # Initiate a numpy array with values of 0
        for row in range(len(new_array)):
            for col in range(len(new_array)):
                edge_value = np.random.randint(1, edge_max)  # Generate random number between 1 and maximum edge
                # length
                rand_per_mil = np.random.randint(1,
                                                 10000)  # Random int used to decide whether to connect 2 nodes or not
                if col - row == 1:
                    new_array[row, col] = edge_value
                    new_array[col, row] = edge_value  # Needed so edge A->B is the same distance as B->A
                elif row < col and sat_per_mil > rand_per_mil:
                    new_array[row, col] = edge_value
                    new_array[col, row] = edge_value  # Needed so edge A->B is the same distance as B->A
        # Prevent direct connection between 1st and last node
        new_array[0, -1] = 0
        new_array[-1, 0] = 0

        if self.debug:
            end = time.time()
            print(f'Time to generate random array took: {end - start}\n')

        return new_array

    def visualise_graph(self, circle_mode=True):
        """
        Method to visualise graph, for upto 50 nodes and a maximum edge weight of 6.
        When circle_mode == True, the nodes are placed at equal separations around a circle.
        For a number of nodes <= 9, circle_mode can be set to False, to display the graph as a
        staggered grid.
        Return None
        """

        # Checking suitability for visualisation
        if self.num_nodes > 50:
            raise ValueError("Too many nodes to display (50 Max)")
        elif self.edge_max > 6:
            raise ValueError("Edge maximum too high for display (6 Max)")

        # Check if array_a_star has been run, if not run it
        if self.a_star_path is None:
            self.array_a_star(0, self.num_nodes - 1)  # Set start and end node
        node_coords = {}  # Node label as key, values are list of x and y coords

        # Setting some constants for the plot
        RADIUS = 180  # Radius of circle that nodes are on
        COLOUR_DICT = {1: "r", 2: "y", 3: "g", 4: "b", 5: "m"}
        LABEL_OFFSET = 1.2  # This sets how far the labels are from nodes (circle mode)

        # Generating node locations on circle
        if circle_mode:
            for i in range(0, self.num_nodes):
                x_value = RADIUS * math.cos(i * 2 * math.pi / self.num_nodes)
                y_value = RADIUS * math.sin(i * 2 * math.pi / self.num_nodes)
                node_coords[i] = [x_value, y_value]
        elif self.num_nodes > 9:
            raise ValueError("Circle mode must be used for nodes > 9")
        else:
            for i in range(0, self.num_nodes):
                # Staggering the nodes to reduce edge overlap
                x_value = i % 3
                y_value = i // 3
                shim_y = 0
                shim_x = 0
                if y_value % 2 == 1:
                    shim_x = 0.3
                if x_value % 2 == 1:
                    shim_y = 0.1
                x_value += shim_x
                y_value += shim_y
                # Scaling
                x_value *= 100
                y_value *= 100

                node_coords[i] = [x_value, y_value]

        # Creating lists of x and y coords for plotting
        all_x = []
        all_y = []
        for key in node_coords:
            all_x.append(node_coords[key][0])
            all_y.append(node_coords[key][1])

        # Plotting the lines connecting the nodes
        for i in range(len(self.origin_array)):
            for j in range(len(self.origin_array[i])):
                edge_weight = self.origin_array[i, j]
                if edge_weight != 0:
                    xs = []
                    ys = []
                    xs.append(node_coords[i][0])
                    xs.append(node_coords[j][0])
                    ys.append(node_coords[i][1])
                    ys.append(node_coords[j][1])
                    plt.plot(xs, ys, COLOUR_DICT[edge_weight], marker=None)

        # Plotting nodes and their labels
        for key in node_coords:
            # Plots nodes
            plt.plot(node_coords[key][0], node_coords[key][1], "k", marker="o")
            # Plots labels
            if circle_mode:
                if key == 0:
                    plt.text(node_coords[key][0] * LABEL_OFFSET, node_coords[key][1] * LABEL_OFFSET,
                             s="Start")
                elif key == (self.num_nodes - 1):
                    plt.text(node_coords[key][0] * LABEL_OFFSET, node_coords[key][1] * LABEL_OFFSET, s="End")
                else:
                    plt.text(node_coords[key][0] * LABEL_OFFSET, node_coords[key][1] * LABEL_OFFSET,
                             s=str(key))
            else:
                if key == 0:
                    plt.text(node_coords[key][0] - 10, node_coords[key][1] - 10, s="Start")
                elif key == (self.num_nodes - 1):
                    plt.text(node_coords[key][0] - 10, node_coords[key][1] - 10, s="End")
                else:
                    plt.text(node_coords[key][0] - 10, node_coords[key][1] - 10, s=str(key))

                    # Formatting the graph
        plt.title("""Example Network with Edge Colour indicating Connection Time
                (Dashed Line is A* Route)""")
        plt.legend([1, 2, 3, 4, 5], bbox_to_anchor=(1, 1), loc="upper left", title="Edge Weight")
        ctrl = plt.gca()  # Needed to edit the legend handles
        leg_ctrl = ctrl.get_legend()
        for i in range(1, len(leg_ctrl.legend_handles) + 1):
            leg_ctrl.legend_handles[i - 1].set_color(COLOUR_DICT[i])
        ctrl.get_xaxis().set_visible(False)  # 2D position is arbitrary - don't show
        ctrl.get_yaxis().set_visible(False)  # 2D position is arbitrary - don't show

        # Adjusting plot x and y limits
        if circle_mode:
            plt.xlim(-240, 240)
            plt.ylim(-240, 240)

            # Adding the A* path to the graph
        soln_trace_x = []  # Store x-coords of trace
        soln_trace_y = []  # Store y-coords of trace
        for i in self.a_star_path:
            soln_trace_x.append(node_coords[i][0])
            soln_trace_y.append(node_coords[i][1])
        plt.plot(soln_trace_x, soln_trace_y, "k:", lw="5")

        # Display the finished graph
        plt.show()


class Node:
    def __init__(self):
        self.value: int = 0
        self.is_start: bool = False
        self.is_stop: bool = False
        self.neighbours: dict[Node: int] = dict()
        self.g_score = 0
        self.h_score = 0

    def __str__(self):
        if self.is_start:
            return f'Start Node {self.value} ({len(self.neighbours)}n)'
        elif self.is_stop:
            return f'Stop Node {self.value} ({len(self.neighbours)}n)'
        else:
            return f'Node {self.value} ({len(self.neighbours)}n)'


class Edge:
    def __init__(self):
        self.weight = None

    def __str__(self):
        return f'Edge of weight {self.weight}'


if __name__ == '__main__':
    # graph = np.array(([0, 4, 4, 0, 0], [4, 0, 2, 1, 1], [4, 2, 0, 1, 0], [0, 1, 1, 0, 4], [0, 1, 0, 4, 0]))

    graph = Graph(num_nodes=8, saturation=0.2, edge_max=5, debug=True)

    graph.array_a_star(0, 6)
    graph.visualise_graph(circle_mode=True)
    graph.visualise_graph(circle_mode=False)

    # test = Graph(from_file=True, debug=True)
    #
    # test.read_edge_list('yeast.edgelist', generator=False)
    #
    # test.graph_a_star(start_node=0, stop_node=5400)
