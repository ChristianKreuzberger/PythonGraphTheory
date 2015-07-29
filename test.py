__author__ = 'ckreuz'
from igraph import *
import random
import RandomNetwork
from ZipfDistribution import sample_requests_from_zipf_distribution


# make sure to isntall at least version 0.7 of igraph
# sudo pip install python-igraph

import igraph

print igraph.__version__

def edgeColor(flowValue):
    if flowValue == 0:
        return "black"
    else:
        return "red"

def remainingCapacity(capacity):
    if capacity == 0:
        return "red"
    elif capacity < 1500:
        return "orange"
    else:
        return "black"

def edgeLabel(flowValue):
    if flowValue == 0:
        return ""
    else:
        return str(flowValue)

def edgeWeight(originalCapacity, currentCapacity, currentDemand):
    if currentCapacity <= 0.01: # literally no capacity remaining
        print "link dead, remaining capacity = ", currentCapacity, " demand=", currentDemand
        return 10000 # very high weight, do not use this link
    if currentDemand > currentCapacity:
        print "not enough cap"
        return 100000 # cant satisfy this either

    usedCapacity = originalCapacity - currentCapacity
    if usedCapacity > originalCapacity * 0.8: # if more than 80 % o the capacity is used, we must avoid this link
        return 4

    if usedCapacity > originalCapacity * 0.6: # if more than 60 % of the capacity is used, we should try to avoid this link
        return 2 # try to avoid this link

    return 1



def exportNetwork(network):
    graph = network["graph"]
    routers = network["routers"]
    clients = network["clients"]
    servers = network["servers"]

    print "data; "
    print ""
    print "set Peer := " + ",".join(clients) + ";"
    print "set Server := " + ",".join(servers) + ";"
    print "set Router := " + ",".join(routers) + ";"
    print ""
    print "param: E: EdgeCapacity := "

    for edge in graph.es:
        print graph.vs[edge.source]["name"], "\t", graph.vs[edge.target]["name"], "\t", edge["capacity"]

    print ";"
    print "end;"


def exportDemands(demands):
    print "data;"
    print ""
    print "param: Demands: MaxDemand := "
    for demand in demands:
        print demand["from"], "\t", demand["to"], "\t10000"
    print ";"
    print "end;"

random.seed(0)

numRouters = 100
numClients = 1000
numServers  = 10

serverCapacity = 100000.0 # 100 mbit
clientCapacity = 10000.0 # 10 mbit
routerCapacity = 100000.0 # 100 Mbit
#print "Generating random network..."

network = RandomNetwork.read_from_file('network.dat')

# network = RandomNetwork.generate_random_network(numRouters, numClients, numServers, routerCapacity, clientCapacity, serverCapacity)

#exportNetwork(network)

g = network["graph"]
layout = g.layout("kk")
plot(g, layout = layout)

print network["clients"]
clients = network["clients"]
servers = network["servers"]
content = network["content"]

#print "random network generated. Testing max flow..."

#flow = g.maxflow(source=servers["Server0"], target=clients["Client2"], capacity=g.es["capacity"])
#flow.graph["color"] = "red"


#g.es["color"] = [edgeColor(flowValue) for flowValue in flow.flow]
#g.es["label"] = [edgeLabel(flowValue) for flowValue in flow.flow]


#plot(g, layout=layout)




def findPathForDemand(graph, client, server, demandValue, edgeWeightFunction=edgeWeight):
    # find edges with no demand remaining and delete them
    for edge in graph.es:
        if edge["capacity"] < demandValue:
            #print "capacity too low,removing...", edge.index, edge
            graph.delete_edges(edge.index)

    weights = [edgeWeightFunction(a,b, demandValue) for a,b in zip (graph.es["originalCapacity"], graph.es["capacity"])]


    # todo: output="epath"
    shortestPaths = []
    try:
        shortestPaths = graph.get_shortest_paths(v=server, to=client, weights=weights)
    except:
        return False

    if len(shortestPaths[0]) == 0:
        return False


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


zipfAlpha = 1.0

# demands = sample_requests_from_zipf_distribution(zipfAlpha, clients, servers)
demands = network['demand']

for demand in demands:
    demand["value"] = 0.0
    demand["enabled"] = True

#print demands
#exportDemands(demands)

#oldGraph = g.copy()

#representations = [ 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 7000, 8500, 10000 ]
representations = content['Content0']['representations']


for rep in representations:
    for demand in demands:
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

#print g.es["capacity"]
g.es["color"] = [remainingCapacity(capacity) for capacity in g.es["capacity"]]

sumBitrates = 0.0

max = 0
min = 1000000

for demand in demands:
    #print demand
    sumBitrates += demand["value"]
    if demand["value"] < min:
        min = demand["value"]
    if demand["value"] > max:
        max = demand["value"]

print "Sum of all flows =", sumBitrates
print "Avg(Flow)=", (sumBitrates)/len(demands)
print "Max(Flow)=", max
print "Min(Flow)=", min

#plot(g, layout=layout)

print "done"

