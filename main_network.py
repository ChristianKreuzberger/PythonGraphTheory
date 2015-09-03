__author__ = 'ckreuz'
from network import *
from math_network import *
from nonlinopt_network import *
import time


import numpy as np
from numpy.linalg import inv

n = Network()

n.import_network("network.dat")
#n.create_random_network(800, 1000, 50)

num_edges = n.graph.ecount()
num_demands = len(n.demands)


print"Determining M matrix"

M = n.export_demand_paths_matrix()
C = n.export_edge_capacity()

print np_array_to_mathematica(M)
print np_array_to_mathematica(C)




#print np_array_to_matlab(M)
#print np_array_to_matlab(C)


#exit()

flows = {}

print "Determining some example flows..."

start = time.time()

flows[1] = n.calculate_fixed_single_path_blocking_min_flows()

stop = time.time()

time1 = stop - start

start = time.time()

#print"Determining another set of example flows..."
flows[2] = n.calculate_fixed_single_path_blocking_maxmin_flows()

stop = time.time()

time2 = stop - start





#res = M.dot(flows2) - C
#print("Residuals=", max(res))


start = time.time()
# gradient_based_line_search linear_decreasing_line_search
flows[3] = nlp_optimize_network(M,C,flows[1],c_neg_log_sum_flows,c_grad_neg_log_sum_flows,5000,line_search=gradient_based_line_search)
stop = time.time()

time3 = stop - start

print "!!!!!!!! gradient_based_line_search Took ", stop - start, " seconds!"


start = time.time()
# gradient_based_line_search linear_decreasing_line_search
flows[4] = nlp_optimize_network(M,C,flows[1],c_neg_log_sum_flows,c_grad_neg_log_sum_flows,5000,line_search=smart_linear_decreasing_line_search)
stop = time.time()

time4 = stop - start

print "!!!!!!!! smart_linear_decreasing_line_search Took ", stop - start, " seconds!"



start = time.time()
# gradient_based_line_search linear_decreasing_line_search
flows[5] = nlp_optimize_network(M,C,flows[2],c_neg_log_sum_flows,c_grad_neg_log_sum_flows,5000,line_search=gradient_based_line_search)
stop = time.time()

time5 = stop - start

print "!!!!!!!! gradient_based_line_search Took ", stop - start, " seconds!"


start = time.time()
# gradient_based_line_search linear_decreasing_line_search
flows[6] = nlp_optimize_network(M,C,flows[2],c_neg_log_sum_flows,c_grad_neg_log_sum_flows,5000,line_search=smart_linear_decreasing_line_search)
stop = time.time()

time6 = stop - start

print "!!!!!!!! smart_linear_decreasing_line_search Took ", stop - start, " seconds!"


res = {}


for i in range(1,7):
    res[i] = M.dot(flows[i]) - C

print ("Log(flows)","Sum(flows)","Time used","Total time used","Residuals")

print(c_log_sum_flows(flows[1]), c_sum_flows(flows[1]), time1, time1, max(res[1]))

print(c_log_sum_flows(flows[2]), c_sum_flows(flows[2]), time2, time2, max(res[2]))

print(c_log_sum_flows(flows[3]), c_sum_flows(flows[3]), time3, time3 + time1, max(res[3]))

print(c_log_sum_flows(flows[4]), c_sum_flows(flows[4]), time4, time4 + time1, max(res[4]))

print(c_log_sum_flows(flows[5]), c_sum_flows(flows[5]), time5, time5 + time2, max(res[5]))

print(c_log_sum_flows(flows[6]), c_sum_flows(flows[6]), time6, time6 + time2, max(res[6]))


