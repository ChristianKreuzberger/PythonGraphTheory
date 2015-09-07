__author__ = 'ckreuz'
from network import *
from math_network import *
from nonlinopt_network import *
import time
import random

if len(sys.argv) > 1:
    minval = int(sys.argv[1])
    maxval = int(sys.argv[2])
else:
    minval = 1
    maxval = 2

for rk in range(minval, maxval):
    rk = 203
    random.seed(rk)
    print "random seed = ", rk

    try:
        n = Network()

        #n.import_network("network.dat")
        n.create_random_network(200, 100, 10)

        num_edges = n.graph.ecount()
        num_demands = len(n.demands)


        print"Determining M matrix"

        M = n.export_demand_paths_matrix()
        C = n.export_edge_capacity()

        flows = {}

        print "Determining some example flows..."

        start = time.time()

        flows[1] = {'xk': n.calculate_fixed_single_path_blocking_min_flows(), 'k': 0 }

        stop = time.time()

        time1 = stop - start

        start = time.time()

        flows[2] = {'xk': n.calculate_fixed_single_path_blocking_maxmin_flows(), 'k': 0}

        stop = time.time()

        time2 = stop - start



        start = time.time()
        # gradient_based_line_search linear_decreasing_line_search
        flows[3] = nlp_optimize_network(M,C,flows[1]['xk'],c_neg_log_sum_flows,c_grad_neg_log_sum_flows,5000,line_search=gradient_based_line_search,fgradmu=gradmu)
        stop = time.time()

        time3 = stop - start

        print "!!!!!!!! gradient_based_line_search Took ", stop - start, " seconds!"


        start = time.time()
        # gradient_based_line_search linear_decreasing_line_search
        flows[4] = nlp_optimize_network(M,C,flows[1]['xk'],c_neg_log_sum_flows,c_grad_neg_log_sum_flows,5000,line_search=smart_linear_decreasing_line_search)
        stop = time.time()

        time4 = stop - start

        print "!!!!!!!! smart_linear_decreasing_line_search Took ", stop - start, " seconds!"



        start = time.time()
        # gradient_based_line_search linear_decreasing_line_search
        flows[5] = nlp_optimize_network(M,C,flows[2]['xk'],c_neg_log_sum_flows,c_grad_neg_log_sum_flows,5000,line_search=gradient_based_line_search,fgradmu=gradmu)
        stop = time.time()

        time5 = stop - start

        print "!!!!!!!! gradient_based_line_search Took ", stop - start, " seconds!"


        start = time.time()
        # gradient_based_line_search linear_decreasing_line_search
        flows[6] = nlp_optimize_network(M,C,flows[2]['xk'],c_neg_log_sum_flows,c_grad_neg_log_sum_flows,5000,line_search=smart_linear_decreasing_line_search)
        stop = time.time()

        time6 = stop - start

        print "!!!!!!!! smart_linear_decreasing_line_search Took ", stop - start, " seconds!"


        start = time.time()
        flows[7] = nlp_optimize_network(M,C,flows[1]['xk'],c_neg_log_sum_flows,c_grad_neg_log_sum_flows,5000,line_search=binary_line_search_gradient)
        stop = time.time()

        time7 = stop - start

        print "!!!!!!!! binary_line_search Took ", stop - start, " seconds!"

        start = time.time()
        flows[8] = nlp_optimize_network(M,C,flows[2]['xk'],c_neg_log_sum_flows,c_grad_neg_log_sum_flows,5000,line_search=binary_line_search_gradient)
        stop = time.time()

        time8 = stop - start

        print "!!!!!!!! binary_line_search Took ", stop - start, " seconds!"

        res = {}


        for i in range(1,9):
            res[i] = M.dot(flows[i]['xk']) - C

        print ("Log(flows)","Sum(flows)","Time used","Total time used","Residuals","NumIterations")

        print(c_log_sum_flows(flows[1]['xk']), c_sum_flows(flows[1]['xk']), time1, time1, max(res[1]), flows[1]['k'])

        print(c_log_sum_flows(flows[2]['xk']), c_sum_flows(flows[2]['xk']), time2, time2, max(res[2]), flows[2]['k'])

        print(c_log_sum_flows(flows[3]['xk']), c_sum_flows(flows[3]['xk']), time3, time3 + time1, max(res[3]), flows[3]['k'])

        print(c_log_sum_flows(flows[4]['xk']), c_sum_flows(flows[4]['xk']), time4, time4 + time1, max(res[4]), flows[4]['k'])

        print(c_log_sum_flows(flows[5]['xk']), c_sum_flows(flows[5]['xk']), time5, time5 + time2, max(res[5]), flows[5]['k'])

        print(c_log_sum_flows(flows[6]['xk']), c_sum_flows(flows[6]['xk']), time6, time6 + time2, max(res[6]), flows[6]['k'])

        print(c_log_sum_flows(flows[7]['xk']), c_sum_flows(flows[7]['xk']), time7, time7 + time1, max(res[7]), flows[7]['k'])

        print(c_log_sum_flows(flows[8]['xk']), c_sum_flows(flows[8]['xk']), time8, time8 + time2, max(res[8]), flows[8]['k'])
    except:
        print "Error in iteration for random seed =", rk
        raise

# end for
