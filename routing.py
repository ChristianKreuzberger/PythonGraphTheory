__author__ = 'ckreuz'
from igraph import *
import random
import RandomNetwork

def nodeColor(nodeName):
    if "Client" in nodeName:
        return "green";
    elif "Server" in nodeName:
        return "orange";
    elif "Router" in nodeName:
        return "cyan";
    else:
        return "white";

def nodeSize(nodeName):
    if "Client" in nodeName:
        return 35
    elif "Server" in nodeName:
        return 25
    elif "Router" in nodeName:
        return 25
    else:
        return 30


def nodeShape(nodeName):
    if "Client" in nodeName:
        return "triangle-up"
    elif "Server" in nodeName:
        return "triangle-up"
    elif "Router" in nodeName:
        return "rectangle"
    else:
        return "circle"

network = RandomNetwork.read_from_file('network1.dat')


def shortLabel(nodeName):
    if "Client" in nodeName:
        return nodeName.replace("Client", "C")
    elif "Server" in nodeName:
        return nodeName.replace("Server", "S")
    elif "Router" in nodeName:
        return nodeName.replace("Router", "R")
    else:
        return nodeName.replace("Content", "V")



def edgeColor(originalCapacity, currentCapacity):
    if currentCapacity <= 0.01: # nothing left
        return "red"
    elif currentCapacity/originalCapacity <= 0.1: # less than 10%
        return "orange"
    elif currentCapacity / originalCapacity <= 0.3: # less than 30%
        return "dark orange"
    elif currentCapacity / originalCapacity <= 0.6: # less than 60%
        return "cyan"
    else:
        return "green"

def edgeWidth(originalCapacity, currentCapacity):
    if currentCapacity != originalCapacity:
        return 3
    else:
        return 1



def checkStaticPathForDemand(graph, FIB, client, server, demandValue):
    shortestPath = []
    find_route(graph, FIB, client, server)
    try:
        shortestPaths = [ find_route(graph, FIB, client, server) ]
    except:
        return False

    if len(shortestPaths) == 0:
        return False

    # check capacity
    for path in shortestPaths:
        lastNodeId = -1
        newPath = list(reversed(path))
        for nodeId in newPath:
            if lastNodeId != -1:
                edgeId =  g.get_eid(lastNodeId, nodeId)
                #print "Edge(" + str(lastNodeId) + "," + str(nodeId) + ") has id", edgeId
                #graph.es[edgeId]["color"] = "red"
                if graph.es[edgeId]["capacity"] < demandValue:
                    return False

            lastNodeId = nodeId


    for path in shortestPaths:
        lastNodeId = -1
        newPath = list(reversed(path))
        for nodeId in newPath:
            if lastNodeId != -1:
                #print "Edge(" + str(lastNodeId) + "," + str(nodeId) + ")"
                edgeId =  g.get_eid(lastNodeId, nodeId)
                #graph.es[edgeId]["color"] = "red"
                graph.es[edgeId]["capacity"] -= demandValue

            lastNodeId = nodeId

    return True





def satisfyDemandsStaticRoute(g, FIB, clients, content, activeDemands, representations):
     # check if demands can be satisfied with heuristic
    for rep in representations:
        for demand in activeDemands:
            if demand["enabled"]:
                lastRep = demand["value"]
                neededBandwidth = rep - lastRep
                satisfied = checkStaticPathForDemand(g, FIB, clients[demand["from"]], content[demand["to"]]['vertex_id'], neededBandwidth)
                if not satisfied:
                    print "Could not satisfy demand of ", rep, " for " + demand["from"] + " to " + demand["to"]
                    demand["enabled"] = False
                else:
                    #print "Demand of ", rep, " for " + demand["from"] + " to " + demand["to"] + " satisfied"
                    demand["value"] = rep
    return activeDemands


g = network["graph"]
clients = network["clients"]
servers = network["servers"]
content = network["content"]

g.vs["color"] = [nodeColor(nodeName) for nodeName in g.vs["label"]]

g.vs["size"] = [nodeSize(nodeName) for nodeName in g.vs["label"]]
g.vs["shape"] = [nodeShape(nodeName) for nodeName in g.vs["label"]]


random.seed(4) # set seed for plotting the network!!!
layout = g.layout("fr")
#plot(g, layout = layout, bbox = (1500, 900), margin=50, vertex_label = [shortLabel(nodeName) for nodeName in g.vs["label"]])


# now, start doing demands
demands = network['demand']



# create and populate a forward information base
def create_forward_information_base(g, content, servers):
    FIB = {}
    # for each vertice v in g.vs, calculate the shortest path + hop count to all contents
    for v in g.vs:
        if v['name'] in content:
            continue

        # create Forward Information Base for v
        v['FIB'] = {}


        for c in content:
            cid = content[c]['vertex_id']
            # create entry for c in FIB
            v['FIB'][cid] = {}
            paths = g.get_all_shortest_paths(v=c, to=v)
            if len(paths) == 0:
                print "no path found... check please...", v, c
            else:
                for path in paths:
                    actual_path = list(reversed(path)) # reverse path
                    next_hop = actual_path[1]
                    hop_count = len(actual_path)
                    if next_hop in v['FIB'][cid]:
                        if hop_count < v['FIB'][cid][next_hop]:
                            v['FIB'][cid][next_hop] = hop_count
                    else:
                        v['FIB'][cid][next_hop] = hop_count
        FIB[v.index] = v['FIB']
    return FIB

def get_next_hop(FIB, cur_node, content_node):
    if content_node in FIB[cur_node]:
        return FIB[cur_node][content_node]
    else:
        print "Could not find content FIB information for ", content_node, " in node ", cur_node
        return -1

def find_route(g, FIB, start_node, target_node):
    cur_node = start_node
    path = [ start_node ]

    while cur_node != target_node:
        #print "cur_node=", cur_node
        if target_node in FIB[cur_node]:
            next_hops = FIB[cur_node][target_node]

            for next_hop in next_hops.keys():
                hop_count = next_hops[next_hop]
                #print next_hop, hop_count
                #print "Next_hop = ", g.vs[next_hop]
            # choose next_hop
            cur_node = next_hop
            path.append(next_hop)
    return path

FIB = create_forward_information_base(g, content, servers)





for demand in demands:
    demand["value"] = 0.0
    demand["enabled"] = True

cnt = 0

activeDemands = []

representations = content['Content0']['representations']

for demand in demands:
    print "Satisfying New Demand: client", demand['from'], 'demands content', demand['to']
    activeDemands.append(demand)
    thisContent = demand['to']

    # reset edge capacities
    g.es['capacity'] = g.es['originalCapacity']

    # reset demand values
    for demand in activeDemands:
        demand["value"] = 0.0
        demand["enabled"] = True

    # check if activeDemands can be satisfied with heuristic
    activeDemands = satisfyDemandsStaticRoute(g, FIB, clients, content, activeDemands, representations)


    print "Current Values for ", len(activeDemands), " demands:"

    for d in activeDemands:
        print d['from'] + ',' + d['to'] + ',' + str(d['value'])

    plot(g, "actual_demands_" + str(cnt) + ".pdf", layout = layout, bbox = (1400, 750), margin=50,
         vertex_label = [shortLabel(nodeName) for nodeName in g.vs["label"]],
         edge_color = [edgeColor(a, b)  for a,b in zip (g.es["originalCapacity"], g.es["capacity"]) ] ,
         edge_width = [edgeWidth(a, b)  for a,b in zip (g.es["originalCapacity"], g.es["capacity"]) ] )
    cnt += 1


