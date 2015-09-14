__author__ = 'ckreuz'
from network import *
from math_network import *
from nonlinopt_network import *
import time
import random
import logging

if len(sys.argv) > 1:
    minval = int(sys.argv[1])
    maxval = int(sys.argv[2])
else:
    minval = 16
    maxval = 500

for rk in range(minval, maxval):
    random.seed(rk)

    try:
        n = Network()

        #n.import_network("network.dat")
        n.create_random_network(500, 1000, 50)

        num_edges = n.graph.ecount()
        num_demands = len(n.demands)


        print"Determining M matrix"

        M = n.export_demand_paths_matrix()
        C = n.export_edge_capacity()

        # remove unnecessary inequalities
        [Mnew,Cnew] = n.remove_unnecessary_edges(M,C)

        print "Reduced M from ", M.shape, " to ", Mnew.shape

        M = Mnew
        C = Cnew

        flows = {}
        runtimes = {}

        print "Determining intial set of flows..."

        start = time.time()

        flows[1] = {'xk': n.calculate_fixed_single_path_blocking_min_flows(), 'k': 0 }

        stop = time.time()

        runtimes[1] = stop - start

        print "Done after ", runtimes[1]

        print "Determining second initial set of flows (should be better)..."

        start = time.time()

        flows[2] = {'xk': n.calculate_fixed_single_path_blocking_maxmin_flows(), 'k': 0}

        stop = time.time()

        runtimes[2] = stop - start

        print "Done after ", runtimes[2]


        start = time.time()
        # gradient_based_line_search linear_decreasing_line_search
        flows[3] = nlp_optimize_network(M,C,flows[1]['xk'],c_neg_log_sum_flows,c_grad_neg_log_sum_flows,5000,line_search=gradient_based_line_search,fgradmu=gradmu)
        stop = time.time()

        runtimes[3] = stop - start

        print "!!!!!!!! gradient_based_line_search Took ", stop - start, " seconds!"



        start = time.time()
        # gradient_based_line_search linear_decreasing_line_search
        flows[4] = nlp_optimize_network(M,C,flows[1]['xk'],c_neg_log_sum_flows,c_grad_neg_log_sum_flows,5000,line_search=smart_linear_decreasing_line_search)
        stop = time.time()

        runtimes[4] = stop - start

        print "!!!!!!!! smart_linear_decreasing_line_search Took ", stop - start, " seconds!"



        start = time.time()
        # gradient_based_line_search linear_decreasing_line_search
        flows[5] = nlp_optimize_network(M,C,flows[2]['xk'],c_neg_log_sum_flows,c_grad_neg_log_sum_flows,5000,line_search=gradient_based_line_search,fgradmu=gradmu)
        stop = time.time()

        runtimes[5] = stop - start

        print "!!!!!!!! gradient_based_line_search Took ", stop - start, " seconds!"


        start = time.time()
        # gradient_based_line_search linear_decreasing_line_search
        flows[6] = nlp_optimize_network(M,C,flows[2]['xk'],c_neg_log_sum_flows,c_grad_neg_log_sum_flows,5000,line_search=smart_linear_decreasing_line_search)
        stop = time.time()

        runtimes[6] = stop - start

        print "!!!!!!!! smart_linear_decreasing_line_search Took ", stop - start, " seconds!"


        res = {}

        print "RESULTS FOR RANDOM SEED rk=" + str(rk)
        print ("Log(flows)","Diff(Log(bestflow))", "Sum(flows)","Diff(Sum(bestflow))", "Time used","Total time used","Residuals","NumIterations")

        max_sum_value = 0.0
        max_sum_i = 0
        max_log_sum_value = 0.0
        max_log_sum_i = 0

        for i in range(1,7):
            res[i] = M.dot(flows[i]['xk']) - C
            flows[i]['logsum'] = c_log_sum_flows(flows[i]['xk'])
            flows[i]['sum'] = c_sum_flows(flows[i]['xk'])

            if max_sum_value < flows[i]['sum']:
                max_sum_value = flows[i]['sum']
                max_sum_i = i
            if max_log_sum_value < flows[i]['logsum']:
                max_log_sum_value = flows[i]['logsum']
                max_log_sum_i = i

            flows[i]['res'] = res[i]

        for i in range(1,7):
            print(flows[i]['logsum'], max_log_sum_value-flows[i]['logsum'], flows[i]['sum'], max_sum_value - flows[i]['sum'], runtimes[i], runtimes[i], max(flows[i]['res']), flows[i]['k'])

        for i in range(3,7):
            if max_log_sum_value-flows[i]['logsum'] > 5:
                print "CHECKPOINT REACHED - result for algo", i, " is not optimal!"
                exit()
        exit()
    except:
        print "Error in iteration for random seed =", rk
        raise

# end for
