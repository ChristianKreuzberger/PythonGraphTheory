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

def edgeWeight(originalCapacity, currentCapacity, currentDemand):
    if currentCapacity <= 0.01: # literally no capacity remaining
        #print "link dead, remaining capacity = ", currentCapacity, " demand=", currentDemand
        return 10000 # very high weight, do not use this link
    if currentDemand > currentCapacity:
        #print "not enough cap, but probably still satisfiable"
        return 10000 # cant satisfy this either

    usedCapacity = originalCapacity - currentCapacity

    if usedCapacity > originalCapacity * 0.9: # this is really not optimal anymore, so we allow detours of length 8 now
        return 8

    if usedCapacity > originalCapacity * 0.8: # if more than 80 % o the capacity is used, we should try to find alternative links with 4 times the length
        return 4

    if usedCapacity > originalCapacity * 0.6: # if more than 60 % of the capacity is used, we should try to find alternative links with twice the length
        return 2

    return 1


def findPathForDemand(graph, client, server, demandValue, edgeWeightFunction=edgeWeight):
    # find edges with no demand remaining and delete them

    weights = [edgeWeightFunction(a,b, demandValue) for a,b in zip (graph.es["originalCapacity"], graph.es["capacity"])]


    # todo: output="epath"
    shortestPaths = []
    try:
        shortestPaths = graph.get_shortest_paths(v=server, to=client, weights=weights)
    except:
        return False

    if len(shortestPaths) == 0:
        return False


    # check capacity
    for path in shortestPaths:
        lastNodeId = -1
        for nodeId in path:
            if lastNodeId != -1:
                #print "Edge(" + str(lastNodeId) + "," + str(nodeId) + ")"
                edgeId =  g.get_eid(lastNodeId, nodeId)
                #graph.es[edgeId]["color"] = "red"
                if graph.es[edgeId]["capacity"] < demandValue:
                    #print "not satisfiable!"
                    return False

            lastNodeId = nodeId


    for path in shortestPaths:
        lastNodeId = -1
        for nodeId in path:
            if lastNodeId != -1:
                #print "Edge(" + str(lastNodeId) + "," + str(nodeId) + ")"
                edgeId =  g.get_eid(lastNodeId, nodeId)
                #graph.es[edgeId]["color"] = "red"
                graph.es[edgeId]["capacity"] -= demandValue

            lastNodeId = nodeId


    return True


def satisfy(g, clients, content, demand, representations, edgeWeight=edgeWeight):
    for rep in representations:
        if demand["enabled"]:
            lastRep = demand["value"]
            neededBandwidth = rep - lastRep
            satisfied = findPathForDemand(g, clients[demand["from"]], content[demand["to"]]['vertex_id'], neededBandwidth, edgeWeight)
            if not satisfied:
                #print "Could not satisfy demand of ", rep, " for " + demand["from"] + " to " + demand["to"]
                demand["enabled"] = False
            else:
                #print "Demand of ", rep, " for " + demand["from"] + " to " + demand["to"] + " satisfied"
                demand["value"] = rep
    return demand["value"]


def satisfyDemands(g, clients, content, activeDemands, representations, edgeWeight=edgeWeight):
     # check if demands can be satisfied with heuristic
    for rep in representations:
        for demand in activeDemands:
            if demand["enabled"]:
                lastRep = demand["value"]
                neededBandwidth = rep - lastRep
                satisfied = findPathForDemand(g, clients[demand["from"]], content[demand["to"]]['vertex_id'], neededBandwidth, edgeWeight)
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
plot(g, layout = layout, bbox = (1500, 900), margin=50, vertex_label = [shortLabel(nodeName) for nodeName in g.vs["label"]])


# now, start doing demands
demands = network['demand']

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
    activeDemands = satisfyDemands(g, clients, content, activeDemands, representations)


    print "Current Values for ", len(activeDemands), " demands:"

    for d in activeDemands:
        print d['from'] + ',' + d['to'] + ',' + str(d['value'])

    plot(g, "actual_demands_" + str(cnt) + ".pdf", layout = layout, bbox = (1400, 750), margin=50,
         vertex_label = [shortLabel(nodeName) for nodeName in g.vs["label"]],
         edge_color = [edgeColor(a, b)  for a,b in zip (g.es["originalCapacity"], g.es["capacity"]) ] ,
         edge_width = [edgeWidth(a, b)  for a,b in zip (g.es["originalCapacity"], g.es["capacity"]) ] )
    cnt += 1


