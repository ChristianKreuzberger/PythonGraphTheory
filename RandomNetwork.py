__author__ = 'ckreuz'

from igraph import *
import random


def read_from_file(fileName):
    clients = {}
    servers = {}
    routers = {}
    content = {}
    demand = []

    # create new graph
    g = Graph()
    g = g.as_directed()


    # open file
    fp = open(fileName, "r")

    current_status = 0 # (0 -> clients, 1 -> server, 2 -> router, 3 -> content (with representations, SSIM, PSNR), 4 -> edges)

    for line in fp:
        line = line.strip()
        if line == "Client:":
            # do something
            print "Processing client"
            current_status = 0
        elif line == "Server:":
            # do something
            print "processing server"
            current_status = 1
        elif line == "Router:":
            # do somethingl
            print "processing router"
            current_status = 2
        elif line == "Content:":
            # do something
            print "processing content"
            current_status = 3
        elif line == "Edge:":
            # do something
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
                    vertex_id = g.vcount()
                    g.add_vertex(line)
                    clients[line] = vertex_id
                elif current_status == 1:
                    # create a new server in graph
                    vertex_id = g.vcount()
                    g.add_vertex(line)
                    servers[line] = vertex_id
                elif current_status == 2:
                    # create a new router in graph
                    vertex_id = g.vcount()
                    g.add_vertex(line)
                    routers[line] = vertex_id
                elif current_status == 3:
                    # create a new content node in graph
                    contentline = line.split(',')
                    vertex_id = g.vcount()
                    g.add_vertex(contentline[0])

                    representations = []
                    for rep in contentline[1:]:
                        representations.append(float(rep.strip()))

                    content[contentline[0]] = {'vertex_id': vertex_id, 'representations': representations}
                elif current_status == 4: # edges
                    # each line is an edge with a certain capacity
                    edge = line.split(',')

                    capacity = float(edge[2].strip())
                    g.add_edge(edge[0].strip(), edge[1].strip())
                    edge_id = g.ecount()-1
                    g.es[edge_id]["capacity"] = capacity
                elif current_status == 5: # demand
                    # each demand is a line with client, content and a value
                    demandline = line.split(',')

                    demand.append({'from': demandline[0].strip(), 'to': demandline[1].strip()})

                # end if
    print "Imported ", g.vcount() , " nodes and ", g.ecount(), " edges"
    g.es["originalCapacity"] = g.es["capacity"]
    g.vs["label"] = g.vs["name"]
    return {"graph": g, "clients": clients, "servers": servers, "routers": routers, "content": content, "demand": demand}






def generate_random_network(numRouters, numClients, numServers,
                          routerDefaultCapacity=100000, clientDefaultCapacity=10000, serverDefaultCapacity=100000):
    clients = {}
    servers = {}
    routers = []

    # generate a network
    g = Graph.Barabasi(numRouters, 3)

    # rename all nodes to Router
    k = 0
    for v in g.vs:
        routerName = "Router" + str(k)
        v["name"] = routerName
        routers.append(routerName)
        k+=1

    # give all routers capacity
    g.es["capacity"] = routerDefaultCapacity

    # get min degree
    minDegree = min(g.degree())
    # get all nodes with mindegree

    nodesWithMinDegree = []

    for v in g.vs:
        if g.degree(v) == minDegree:
            nodesWithMinDegree.append(v)


    # add clients
    for k in range(0,numClients):
        # create client node
        clientName = "Client" + str(k)
        vertice_id = g.vcount()
        g.add_vertex(clientName)
        # connect this client node to one node with minimal degree
        randomNodeId = random.randint(0,len(nodesWithMinDegree)-1)
        g.add_edge(clientName, nodesWithMinDegree[randomNodeId])

        max_edge_id = g.ecount()
        g.es[max_edge_id-1]["capacity"] = clientDefaultCapacity

        clients[clientName] = vertice_id

    # add servers
    for k in range(0,numServers):
        # create client node
        serverName = "Server" + str(k)
        vertice_id= g.vcount()
        g.add_vertex(serverName)
        # connect this client node to one node with minimal degree
        randomNodeId = random.randint(0,len(nodesWithMinDegree)-1)
        g.add_edge(serverName, nodesWithMinDegree[randomNodeId])

        max_edge_id = g.ecount()
        g.es[max_edge_id-1]["capacity"] = serverDefaultCapacity

        servers[serverName] = vertice_id


    g.vs["label"] = g.vs["name"]
    for v in g.vs:
        if "Router" in v["name"]:
            v["color"] = "green"
        elif "Client" in v["name"]:
            v["color"] = "yellow"
        elif "Server" in v["name"]:
            v["color"] = "blue"

    # convert into directed graph
    g = g.as_directed()
    # remove edges from network to the server
    for serverName in servers:
        neighbours = g.incident(serverName,mode=IN)

        for v in neighbours:
            g.delete_edges(v)
            #g.delete_edges([g.vs.select(v), g.vs.select(serverName)])

    # remove edges from client to the network
    for clientName in clients:
        neighbours = g.incident(clientName,mode=OUT)
        for v in neighbours:
            g.delete_edges(v)

    g.es["originalCapacity"] = g.es["capacity"]

    return {"graph": g, "clients": clients, "servers": servers, "routers": routers}
