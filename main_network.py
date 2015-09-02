__author__ = 'ckreuz'
from network import *
from math_network import *
from nonlinopt_network import *
import scipy
import scipy.optimize
import time


import numpy as np
from numpy.linalg import inv

n = Network()

n.import_network("network.dat")

cols = []

capacity = {}

for e in n.graph.es:
    capacity[e.index] = e['capacity']


num_edges = n.graph.ecount()
num_demands = len(n.demands)

M = n.export_demand_paths_matrix()
C = n.export_edge_capacity()

flows1 = n.calculate_fixed_single_path_blocking_min_flows()
flows2 = n.calculate_fixed_single_path_blocking_maxmin_flows()


print(c_log_sum_flows(flows1))
print(c_sum_flows(flows1))

res = M.dot(flows1) - C
print("Residuals=", max(res))

print(c_log_sum_flows(flows2))
print(c_sum_flows(flows2))


res = M.dot(flows2) - C
print("Residuals=", max(res))


start = time.time()
# gradient_based_line_search linear_decreasing_line_search
flows3 = nlp_optimize_network(M,C,flows1,c_neg_log_sum_flows,c_grad_neg_log_sum_flows,5000,line_search=smart_linear_decreasing_line_search)
stop = time.time()

print "Took ", stop - start, " seconds!"
exit()


res = M.dot(flows3) - C
print("Residuals=", max(res))

exit()

print M.dot(inv(MT.dot(M))).dot(MT)

# print as matlab stuff
rowlist = []
for row in M:
    a = " ".join(['{:1.0f}'.format(i) for i in row])
    rowlist.append(a)
# end for
print "A=[" + ";\n".join(rowlist) + "];"



# print C
capacities = " ".join(['{:1.0f}'.format(i) for i in capacity.values()])

capacities = "[" + capacities + "]"

print "C=" + capacities + ";"



exit(0)



# print as mathematica stuff
matrix = "{\n"

rowlist = []

for row in cols:
    a = ",".join(['{:1.0f}'.format(i) for i in row])
    rowlist.append( "{" + a + "}")

matrix = ",\n".join(rowlist)
matrix = "{\n" + matrix + "}"

print "A=Transpose[" + matrix + "];"

capacities = ",".join(['{:1.0f}'.format(i) for i in capacity.values()])

capacities = "{" + capacities + "}"

print "CC=" + capacities + ";"



print "x={"

print ",".join( ["x" + str(i) for i in range(1,len(n.demands)+1)  ] )


print "};"