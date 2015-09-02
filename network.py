#!/usr/bin/env python
"""Provides a library for importing and exporting networks and demands
"""
__author__ = 'Christian Kreuzberger'
__email__ = 'christian.kreuzberger@itec.aau.at'
__license__ = 'Not decided yet'
__maintainer__ = 'Christian Kreuzberger'
__status__= 'Development'


from igraph import *
import random
import numpy as np


class Network:
    """ Memory storage for a bi-directional network graph with demands, including importing and exporting
    :type graph: Graph
    :type clients: list[str]
    :type servers: list[str]
    :type routers: list[str]
    :type content: list[str]
    """

    def __init__(self):
        """ Initialize the bi-directional graph
        :return:
        """
        self.clear()

    def clear(self):
        """ Clear the graph, demands, etc...
        :return:
        """
        self.graph = Graph().as_directed()
        self.clients = {}
        self.servers = {}
        self.routers = {}
        self.content = {}
        self.demands = {}


    def print_stats(self):
        """ Print network statistics
        :return:
        """
        print "Network with: ", len(self.demands), "demands,", self.graph.vcount(), "nodes and", self.graph.ecount(), "edges"


    def import_network(self, filename):
        """ import a network with demands from a file
        :param filename: the filename to read
        :rtype Boolean
        :return: True on success, else False
        """

        # open file for reading the network
        fp = open(filename, "r")
        print "Opening", filename, "and reading network and demands ..."

        # (0 -> clients, 1 -> server, 2 -> router, 3 -> content (with representations, SSIM, PSNR), 4 -> edges)
        current_status = 0

        for line in fp:
            line = line.strip()
            if line == "Client:":
                print "Processing client"
                current_status = 0
            elif line == "Server:":
                print "processing server"
                current_status = 1
            elif line == "Router:":
                print "processing router"
                current_status = 2
            elif line == "Content:":
                print "processing content"
                current_status = 3
            elif line == "Edge:":
                print "processing edge"
                current_status = 4
            elif line == "Demand:":
                print "processing demand"
                current_status = 5
            else:
                if line != "" and not line.startswith("#"):
                    # this is a regular line, need to parse it and do according to current_status
                    if current_status == 0:
                        # create a new client in graph
                        vertex_id = self.graph.vcount()
                        self.graph.add_vertex(line)
                        self.clients[line] = vertex_id
                    elif current_status == 1:
                        # create a new server in graph
                        vertex_id = self.graph.vcount()
                        self.graph.add_vertex(line)
                        self.servers[line] = vertex_id
                    elif current_status == 2:
                        # create a new router in graph
                        vertex_id = self.graph.vcount()
                        self.graph.add_vertex(line)
                        self.routers[line] = vertex_id
                    elif current_status == 3:
                        # create a new content node in graph
                        contentline = line.split(',')
                        vertex_id = self.graph.vcount()
                        self.graph.add_vertex(contentline[0])

                        representations = []
                        for rep in contentline[1:]:
                            representations.append(float(rep.strip()))

                        self.content[contentline[0]] = {'vertex_id': vertex_id, 'representations': representations}
                    elif current_status == 4:  # edges
                        # each line is an edge with a certain capacity
                        edge = line.split(',')

                        capacity = float(edge[2].strip())
                        self.graph.add_edge(edge[0].strip(), edge[1].strip())
                        edge_id = self.graph.ecount() - 1
                        self.graph.es[edge_id]["capacity"] = capacity
                    elif current_status == 5:  # demand
                        # each demand is a line with client, content and a value
                        demandline = line.split(',')
                        clientvalue = demandline[0].strip()

                        self.demands[clientvalue] = {'from': clientvalue, 'to': demandline[1].strip()}
                    # end if
                # end if correct line
            # end if line type
        # end for line

        print "Imported", len(self.demands), "demands,", self.graph.vcount(), "nodes and", self.graph.ecount(), "edges"
        self.graph.es["originalCapacity"] = self.graph.es["capacity"]
        self.graph.vs["label"] = self.graph.vs["name"]

        return True

    def export_network(self, filename):
        """ export the current network with demands to a file
        :param filename: the filename to write
        :return: True on success, else False
        """
        return False

    def create_random_network(self, numRouters, numClients, numServers, randomGraphTool="Barabasi",
                              routerDefaultCapacity=100000, clientDefaultCapacity=10000, serverDefaultCapacity=100000):
        # we are using igraph Barabasi to generate the network (should be configureable though)
        if randomGraphTool == "Barabasi":
            self.graph = Graph.Barabasi(numRouters, 3)
        elif randomGraphTool == "??":
            self.graph = None  # todo


        # rename all nodes to Router
        k = 0
        for v in self.graph.vs:
            routerName = "Router" + str(k)
            v["name"] = routerName
            self.routers[routerName] = v.index
            k += 1

        # give all links belonging to routers (so essentially all edges for now) the same capacity
        self.graph.es["capacity"] = routerDefaultCapacity

        # figure out the minimum degree of the nodes in the graph
        minDegree = min(self.graph.degree())

        # get all nodes with minimum degree - these will be used as clients
        nodesWithMinDegree = []

        for v in self.graph.vs:
            if self.graph.degree(v) == minDegree:
                nodesWithMinDegree.append(v)


        # add clients to nodes that have minimum degree
        for k in range(0, numClients):
            # create client node
            clientName = "Client" + str(k)
            vertice_id = self.graph.vcount()
            self.graph.add_vertex(clientName)
            # connect this client node to one node with minimal degree
            randomNodeId = random.randint(0, len(nodesWithMinDegree) - 1)
            self.graph.add_edge(clientName, nodesWithMinDegree[randomNodeId])

            max_edge_id = self.graph.ecount()
            self.graph.es[max_edge_id - 1]["capacity"] = clientDefaultCapacity

            self.clients[clientName] = vertice_id

        # add servers also to nodes that have minimum degree
        for k in range(0, numServers):
            # create client node
            serverName = "Server" + str(k)
            vertice_id = self.graph.vcount()
            self.graph.add_vertex(serverName)
            # connect this client node to one node with minimal degree
            randomNodeId = random.randint(0, len(nodesWithMinDegree) - 1)
            self.graph.add_edge(serverName, nodesWithMinDegree[randomNodeId])

            max_edge_id = self.graph.ecount()
            self.graph.es[max_edge_id - 1]["capacity"] = serverDefaultCapacity

            self.servers[serverName] = vertice_id

        # TODO: Move coloring of nodes into another method
        for v in self.graph.vs:
            if "Router" in v["name"]:
                v["color"] = "green"
            elif "Client" in v["name"]:
                v["color"] = "yellow"
            elif "Server" in v["name"]:
                v["color"] = "blue"

        # convert into directed graph (means: each undirected edge becomes two directed edges)
        self.graph = self.graph.as_directed() # TODO: Check if this is still needed?

        # remove edges from the network to the server
        # the reason for this is simple: we do not want data to flow TO the server
        for serverName in self.servers:
            neighbours = self.graph.incident(serverName, mode=IN)

            for v in neighbours:
                self.graph.delete_edges(v)

        # remove edges from client to the network
        # the reason for this is simple: we do not want data to flow from the client to the network
        for clientName in self.clients:
            neighbours = self.graph.incident(clientName, mode=OUT)
            for v in neighbours:
                self.graph.delete_edges(v)

        # store original capacity
        self.graph.es["originalCapacity"] = self.graph.es["capacity"]
        self.graph.vs["label"] = self.graph.vs["name"]


        print "Created", len(self.demands), "demands,", self.graph.vcount(), "nodes and", self.graph.ecount(), "edges"


    def shortest_path(self, fromNodeName, toNodeName):
        """ calls igraph.get_shortest_paths with the parameters v=fromNodeName and to=toNodeName
        :param fromNodeName: the originating node
        :param toNodeName:  the destination node
        :rtype: list[list[int]]
        :return: a list of shortest paths (can be more than one), consisting of edge IDs
        """
        shortestPaths = self.graph.get_shortest_paths(v=fromNodeName, to=toNodeName)
        return shortestPaths


    def export_demand_paths_matrix(self):
        """ export path-demand incidence matrix A with a_ji = 1 if demand i uses edge j
        :rtype: np.array
        :return: path-demand incidence matrix A
        """
        num_edges = self.graph.ecount()
        cols = []

        # iterate over each client/demand
        for client in self.demands:
            # the column contains all zeros initially
            column = np.zeros(num_edges)
            d = self.demands[client]
            # calculate shortest path from the server to the client
            shortest_paths = self.shortest_path(d['to'], d['from'])
            # for each edge in this path, set the respective entry in the column to 1
            if len(shortest_paths) > 0:
                p = shortest_paths[0]
                for eid in p:
                    column[eid] = 1
            else:
                print "ERROR: Could not find path for demand ", d
            # end if

            # append this column to the rest of the matrix
            cols.append(column)
        # convert the list to a numpy array
        cols = np.array(cols)
        # need to transpose this
        return cols.transpose()

