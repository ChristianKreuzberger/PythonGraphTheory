__author__ = 'ckreuz'
from network import *
from math_network import *
from nonlinopt_network import *
import time
import random
import logging
#from cvxopt import matrix, solvers


if len(sys.argv) > 1:
    minval = int(sys.argv[1])
    maxval = int(sys.argv[2])
else:
    minval = 1
    maxval = 100

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



        flowcnt = 0

        # print "Running linear program"
        # # run a linear program from cvxopt
        # coeff = np.ones(num_demands) * (-1)
        #
        # coeff = np.asarray(coeff)
        #
        # Acvx = matrix(M)
        # coeffcvx = matrix(coeff)
        # b = []
        # for i in range(0,len(C)):
        #     b.append(float(C[i]))
        #
        #
        # # need to add x_i >= 0 at the end of Acvx
        #
        # for i in range(0,num_demands):
        #     constraint = np.zeros(num_demands)
        #     constraint[i] = -1.0 # -1 x_i <= 0
        #     b.append(0.0)
        #     Acvx = matrix([Acvx, matrix(constraint).trans()])
        #
        # b = matrix(b)
        # start = time.time()
        # res = solvers.lp(c=coeffcvx, G=Acvx, h=b)
        #
        # stop = time.time()
        #
        # runtimes[flowcnt] = stop - start
        #
        # print "Done after ", runtimes[flowcnt]
        #
        #
        # xk = np.squeeze(np.array(res['x']).transpose())
        #
        # xk -= 1
        #
        # flows[flowcnt] = {'xk': xk, 'k': res['iterations'], 'name': 'linear_program' }
        #
        # lp_flow_idx = flowcnt
        # flowcnt += 1


        print "Determining intial set of flows..."

        start = time.time()

        flows[flowcnt] = {'xk': n.calculate_fixed_single_path_one_flow(M,C), 'k': 0 , 'name': 'blocking one flow'}

        stop = time.time()

        runtimes[flowcnt] = stop - start
        spof_flow_idx = flowcnt

        print "Done after ", runtimes[flowcnt]



        flowcnt += 1

        print "Determining second initial set of flows (should be a little better)..."

        start = time.time()

        flows[flowcnt] = {'xk': n.calculate_fixed_single_path_blocking_min_flows_new(M,C), 'k': 0, 'name': 'blocking min' }

        stop = time.time()

        runtimes[flowcnt] = stop - start
        spbm_flow_idx = flowcnt

        print "Done after ", runtimes[flowcnt]



        flowcnt += 1

        print "Determining third initial set of flows (should be better)..."

        start = time.time()

        flows[flowcnt] = {'xk': n.calculate_fixed_single_path_blocking_maxmin_flows(M,C), 'k': 0, 'name': 'blocking maxmin'}

        stop = time.time()

        runtimes[flowcnt] = stop - start
        mmfl_flow_idx = flowcnt

        print "Done after ", runtimes[flowcnt]


        flowcnt += 1
        # we are done calculating "initial" values, now start with the "real fun"


        # lp_flow_idx
        initial_flows = [
            # lp_flow_idx,
            spof_flow_idx,
            spbm_flow_idx,
            mmfl_flow_idx
        ]

        max_iterations = 5000

        for idx in initial_flows:
            start = time.time()
            flows[flowcnt] = nlp_optimize_network(M,C,flows[idx]['xk'],c_neg_log_sum_flows,c_grad_neg_log_sum_flows,max_iterations,line_search=gradient_based_line_search,fgradmu=gradmu)
            stop = time.time()

            runtimes[flowcnt] = stop - start
            flows[flowcnt]['initial_alg'] = flows[idx]['name']
            flows[flowcnt]['initial_alg_idx'] = idx
            flows[flowcnt]['name'] = 'gradient'

            print "!!!!!!!! gradient_based_line_search with ", flows[idx]['name'], " took ", stop - start, " seconds!"


            flowcnt += 1

            start = time.time()
            flows[flowcnt] = nlp_optimize_network(M,C,flows[idx]['xk'],c_neg_log_sum_flows,c_grad_neg_log_sum_flows,max_iterations,line_search=smart_linear_decreasing_line_search)
            stop = time.time()

            runtimes[flowcnt] = stop - start
            flows[flowcnt]['initial_alg'] = flows[idx]['name']
            flows[flowcnt]['initial_alg_idx'] = idx
            flows[flowcnt]['name'] = 'smart_linear'

            print "!!!!!!!! smart_linear_decreasing_line_search with ", flows[idx]['name'], " took ", stop - start, " seconds!"

            flowcnt += 1


        res = {}

        print "RESULTS FOR RANDOM SEED rk=" + str(rk)
        print ",".join(["Name", "Number", "Log(flows)","Diff(Log(bestflow))", "Sum(flows)","Diff(Sum(bestflow))", "Time used","Total time used","Residuals","NumIterations", "StopCondition"])

        max_sum_value = 0.0
        max_sum_i = 0
        max_log_sum_value = 0.0
        max_log_sum_i = 0

        for i in range(0, flowcnt):
            res[i] = M.dot(flows[i]['xk']) - C
            try:
                flows[i]['logsum'] = c_log_sum_flows(flows[i]['xk'])
            except:
                flows[i]['logsum'] = -float("inf")
            flows[i]['sum'] = c_sum_flows(flows[i]['xk'])

            if max_sum_value < flows[i]['sum']:
                max_sum_value = flows[i]['sum']
                max_sum_i = i
            if max_log_sum_value < flows[i]['logsum']:
                max_log_sum_value = flows[i]['logsum']
                max_log_sum_i = i

            flows[i]['res'] = res[i]


        for i in range(0, flowcnt):
            if i in initial_flows or 'initial_alg' not in flows[i]:
                print ",".join(map(str,[flows[i]['name'], i, flows[i]['logsum'], max_log_sum_value-flows[i]['logsum'],\
                    flows[i]['sum'], max_sum_value - flows[i]['sum'], runtimes[i], runtimes[i], max(flows[i]['res']), flows[i]['k'], -1]))
            else:
                print ",".join(map(str,[flows[i]['name'] + " using " + flows[i]['initial_alg'], i, flows[i]['logsum'], max_log_sum_value-flows[i]['logsum'],\
                    flows[i]['sum'], max_sum_value - flows[i]['sum'], runtimes[i], \
                    runtimes[i]+runtimes[flows[i]['initial_alg_idx']], max(flows[i]['res']), flows[i]['k'], flows[i]['stop']]))


        for i in range(0, flowcnt):
            if i not in initial_flows:
                if max_log_sum_value-flows[i]['logsum'] > 10:
                    print "CHECKPOINT REACHED - result for algo", i, " is not optimal!"
                    exit()
                if max(flows[i]['res']) > 0.01:
                    print "CHECKPOINT REACHED - result for algo", i, " has large residual!"
                    exit()
    except:
        print "Error in iteration for random seed =", rk
        raise

# end for
