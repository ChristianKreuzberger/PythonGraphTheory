#!/usr/bin/env python
"""Some utils for performing non-linear optimization for the network
"""
__author__ = 'Christian Kreuzberger'
__email__ = 'christian.kreuzberger@itec.aau.at'
__license__ = 'Not decided yet'
__maintainer__ = 'Christian Kreuzberger'
__status__= 'Development'

import numpy as np
import numpy.linalg as lin
import math
import logging

import matplotlib.pyplot as plt




def is_feasible(A,C,x,eps):
    """ checks whether A * x <= C
    :param A:
    :param c:
    :param x:
    :return: True if A * x <= C
    """
    return max(A.dot(x) - C) <= eps



def get_max_feasible_mu(A,C,x,r, residuals, eps):
    mu_max = -max([ (x[i]/r[i] if r[i] < 0 else -float("inf") ) for i in range(0,len(x))])

    impact = A.dot(r)

    for j in range(0,len(C)):
        if abs(impact[j]) > eps: # yes this inequality has an impact on the result here
            # check if inequality is not (yet) active
            if residuals[j] < 0: # it was not active, means we can improve it
                mu = (-residuals[j])/impact[j]
                if mu > eps/100 and mu < mu_max: # we only want positive mu
                    mu_max = mu

    mu_max = mu_max * (1.0 - eps) # just to be sure

    return min(mu_max,float("10e10"))



def gradmu(xk,r,mu):
    return -sum([ r[i] / (xk[i] + mu * r[i]) for i in range(0,len(xk)) ])

def numgradmu(xk,r,mu,eps=float("10e-13")):
    print "mu=", mu
    lower = -sum(([ math.log(xk[i] + (mu-eps) * r[i]) for i in range(0,len(xk)) ]))
    upper = -sum(([ math.log(xk[i] + (mu+eps) * r[i]) for i in range(0,len(xk)) ]))
    return (upper-lower)/(2.0 * eps)


def secondorder(xk,r,mu):
    upper = gradmu(xk,r,mu)
    lower = -sum([ math.pow(r[i],2) / math.pow(xk[i] + mu * r[i],2) for i in range(0,len(xk)) ])
    return upper/lower



def gradient_based_line_search(xk,r,f,gradf,A,C,mu_min, mu_max,eps, fgradmu=gradmu):
    # check gradmu(0) * gradmu(mu_max)


    try:
        grad_mumin = fgradmu(xk,r,mu_min)
        grad_mumax = fgradmu(xk,r,mu_max)

        value_mumax = f(xk+mu_max*r)
        value_mumin = f(xk + mu_min * r)
    except:
        print "mu_max = ", mu_max
        print "xk + r * mu_max = ", (xk + r * mu_max)
        print "r = ", r
        raise


    if abs(grad_mumax) < eps and value_mumax < f(xk):
        logging.info("gradient_based_line_search: Optimum at mu_max (condition 1)")
        return mu_max

    if grad_mumin * grad_mumax < 0:
        logging.info("gradient_based_line_search: There exists a global minimum somewhere here...")
    else:
        if value_mumax > value_mumin:
            logging.info("gradient_based_line_search: Optimum at mu_min (condition 2)")
            return mu_min
        else:
            logging.info("gradient_based_line_search: Optimum at mu_max (condition 3)")
            return mu_max

    #plotmu(xk,r,f,gradf,A,C,mu_min, mu_max,eps)

    # perform newton search to find f'(mu) = 0
    newmu = mu = 0.0
    logging.info("gradient_based_line_search: approaching optimum with newton, starting at mu=", mu)
    # this is easy now, just do a newton search, starting at mu=mu_min

    lastgradmu = 1000
    curgradmu = 1000

    while True:
        newmu = mu + secondorder(xk,r,mu)

        if newmu < 0:
            newmu = eps

        while newmu > mu_max:
            logging.info("gradient_based_line_search: Repairing newton solution...")
            diff = newmu - mu
            newmu -= diff/2


        newgradmu = fgradmu(xk,r,newmu)

        logging.info("gradient_based_line_search: newmu=", newmu, "grad(newmu)=", newgradmu)


        mu = newmu

        if abs(curgradmu - newgradmu) < eps/100:
            break
        curgradmu = newgradmu

    # end while
    return newmu


def plotmu(xk, r, f, gradf, A, C, mu_min, mu_max,eps):
    mu = mu_min

    diff = float(mu_max - mu_min)

    low_val = float("inf")
    low_mu = -1
    values = []
    mus = []
    gradvals = []
    numgradvals = []
    last_val = -1

    numgradvals.append(0)

    while mu < mu_max:
        if min(xk + mu *r) > eps: # xk + mu * r > 0 must hold for logarithm
            cur_val = f(xk + mu * r)

            values.append(cur_val)
            mus.append(mu)

            # determine derive of f(mu)
            gradvalue = -sum([ r[i] / (xk[i] + mu * r[i]) for i in range(0,len(xk)) ])

            gradvals.append(gradvalue)
            if cur_val < low_val:
                low_val = cur_val
                low_mu = mu
            # end if

            if last_val != -1:
                numgradvalue = (cur_val - last_val)/(diff/1000.0)
                numgradvals.append(numgradvalue)
            # end if
            last_val = cur_val
        else:
            print "Infeasible for mu=",mu,"(min(xk + mu * r)= ",min(xk + mu * r) , ",mu_max=",mu_max,")"

        # end if
        mu = mu + diff/1000.0
    # end while


    print mus
    print values
    print gradvals
    print numgradvals

    # plot values
    plt.figure(1)
    plt.subplot(211)
    plt.plot(mus,values,'r')

    plt.subplot(212)
    plt.plot(mus,gradvals, 'b', mus, numgradvals, 'g')

    plt.show()

    return mu


def binary_line_search_gradient(xk,r,f,gradf,A,C,mu_min, mu_max,eps, fgradmu=gradmu):
    low_val = float("inf")

    left_mu = mu_min
    right_mu = mu_max * (1 - eps)

    left_val = gradmu(xk, r, left_mu)
    right_val = gradmu(xk, r, right_mu)

    if left_val * right_val > eps:
        return mu_max

    if left_val < 0:
        k = 1
        while k < 25:
            #print "k=", k, ", left_mu=", left_mu, ", right_mu=", right_mu
            #left_val = f(xk + left_mu * r)
            #right_val = f(xk + right_mu * r)
            left_val = gradmu(xk, r, left_mu)
            right_val = gradmu(xk, r, right_mu)

            if left_val < 0:
                left_mu += (right_mu - left_mu) / 2.0

            if right_val > 0:
                right_mu -= (right_mu - left_mu) / 2.0

            k+=1

        return (right_mu + left_mu)/2.0
    else:
        k = 1
        while k < 25:
            #print "k=", k, ", left_mu=", left_mu, ", right_mu=", right_mu
            #left_val = f(xk + left_mu * r)
            #right_val = f(xk + right_mu * r)
            left_val = gradmu(xk, r, left_mu)
            right_val = gradmu(xk, r, right_mu)

            if left_val > 0:
                left_mu += (right_mu - left_mu) / 2.0

            if right_val < 0:
                right_mu -= (right_mu - left_mu) / 2.0

            k+=1

        return (right_mu + left_mu)/2.0


def binary_line_search(xk,r,f,gradf,A,C,mu_min, mu_max,eps, fgradmu=gradmu):
    low_val = float("inf")

    left_mu = mu_min
    right_mu = mu_max * (1 - eps)
    k = 1
    while k < 25:
        #print "k=", k, ", left_mu=", left_mu, ", right_mu=", right_mu
        left_val = f(xk + left_mu * r)
        right_val = f(xk + right_mu * r)
        # find out which one is lower
        if left_val < right_val:
            # continue with left_val
            right_mu -= (right_mu - left_mu) / 2.0
        else:
            # continue with right_val
            left_mu += (right_mu - left_mu) / 2.0
        k+=1

    return (right_mu + left_mu)/2.0

def linear_decreasing_line_search(xk, r, f, gradf, A, C, mu_min, mu_max,eps, fgradmu = gradmu):
    mu = mu_max

    diff = float(mu_max - mu_min)

    low_val = float("inf")
    low_mu = -1

    while mu > mu_min:
        if min(xk + mu *r) > eps: # xk + mu * r > 0 must hold for logarithm
            cur_val = f(xk + mu * r)

            if cur_val < low_val:
                low_val = cur_val
                low_mu = mu
            # end if
        else:
            print "Infeasible for mu=",mu,"(mu_max=",mu_max,")"

        # end if
        mu = mu - diff/100.0
    # end while

    mu = low_mu
    while not is_feasible(A,C, xk + mu * r, eps):
        mu = mu / 2.0
        print ("not feasible!")


    return mu


def smart_linear_decreasing_line_search(xk, r, f, gradf, A, C, mu_min, mu_max,eps, fgradmu=gradmu):
    mu = mu_max
    diff = float(mu_max - mu_min)

    low_val = f(xk)
    low_mu = 0
    while mu > mu_min:
        if min(xk + mu *r) > eps: # xk + mu * r > 0 must hold for logarithm
            cur_val = f(xk + mu * r)

            if cur_val < low_val:
                low_val = cur_val
                low_mu = mu
            elif low_mu > 0: # it increased --> stop
                break
            # end if
        else:
            print "Infeasible for mu=",mu,"(mu_max=",mu_max,")"

        # end if
        mu = mu - diff/30.0
    # end while

    new_min_mu = max(0.0,low_mu - diff/15.0)
    new_max_mu = min(low_mu + diff/15.0, mu_max)

    if diff > 5:
        return smart_linear_decreasing_line_search(xk,r,f,gradf,A,C,new_min_mu,new_max_mu,eps,fgradmu)
    else:
        return low_mu



def nlp_optimize_network(A,C,x0,f,gradf,max_iterations=1000,line_search=linear_decreasing_line_search,eps=0.00000001,fgradmu=gradmu,goal_acc=0.000000000001):
    """

    :param A:
    :type A: np.array
    :param C:
    :type C: np.array
    :param x0:
    :param f:
    :param gradf:
    :param max_iterations:
    :param eps:
    :return: dictionary: k = number of iterations, xk = found value for x,
        stop =
        {0 ... max_iterations reached, 1 ... line_search says 0, 2 ... relative change of x < eps/100,
         3 ... projection matrix is 0, 4 ... descent direction 0 and lagrange >= 0}
    """

    A = np.array(A)

    m = A.shape[0]
    n = A.shape[1]
    lastobj = obj = f(x0)


    sumA = np.zeros(m)

    for j in range(0,m):
        sumA[j] = sum(A[j,:])


    history = []
    AT = A.transpose()

    stop_condition = 0


    xk = x0
    P = None # projection matrix
    lastActiveConstraints = []

    print "Iteration 0: Obj=", f(xk) , ", sum(x)=", sum(xk)

    residuals = A.dot(xk) - C

    for k in range(1,max_iterations+1):
        logging.info("Iteration",k)

        # figure out active constraints
        active_constraints = []

        for j in range(0, m):
            if residuals[j] >= -eps:
                active_constraints.append(j)
            # end if
        # end for

        # check if active constraints differ from last iteration
        if active_constraints != lastActiveConstraints and len(active_constraints) > 0:
            logging.info("We have",len(active_constraints), "active constraints")
            M = A[active_constraints, :]
            MT = M
            M = M.transpose()
            try:
                # calculate projection matrix
                # Check: Using pseudo inverse (pinv) instead of normal inverse, to compensate for linear dependent rows
                tmp = M.dot(lin.pinv(MT.dot(M))).dot(MT)
                # the disadvantage of the pseudo inverse is that it might be numerically inaccurate
                # therefore we check all values if they are < eps and set them to 0
                # this should greatly help when doing the projection
                smallvalues=abs(tmp) < eps
                tmp[smallvalues] = 0.0

                P = np.identity(n) - tmp
            except:
                print "Error..."
                print M
                print "Activeconstraints=", active_constraints
                print "M.size=", M.size
                print "Rank(M)=", lin.matrix_rank(M)
                raise

            # check if P has only entries close to 0
            if P.max() > -eps and P.max() < eps: # equals: P.max() == 0
                print "STOP Condition: Projection matrix is (almost) 0. No improvement possible. Exiting..."
                stop_condition = 3
                break

        elif len(active_constraints) == 0:
            # means we do not need to project anything
            P = np.identity(n)
        # end if


        # store currently active constraints for next iteration
        lastActiveConstraints = active_constraints

        gradbefore = gradf(xk)

        # calculate projected descent vector
        r = -P.dot(gradbefore)

        if max(r) > -eps and max(r) < eps and len(active_constraints) > 0:
            print "residuals = ", A.dot(xk) - C
            print active_constraints
            print r
            # descent direction is almost 0, check lagrange multipliers
            u = -lin.pinv(MT.dot(M)).dot(MT).dot(gradbefore)

            if min(u) > -0.001:
                print "STOP Condition: Descent = 0, Lagrange > 0 --> KKT Point found."
                stop_condition = 4
                break
            # else:
            print "descent = ", r

            print "lagrange=", u
            print "min(lagrange)=", min(u)

            raise Exception("Descent direction is almost 0, but negative lagrange multipliers found")
            break

        # get maximal feasible mu (such that xk + mu * r >= 0 and A * (xk + mu * r) <= C )
        mu_max = get_max_feasible_mu(A,C,xk,r,residuals, eps)

        # find argmin_{0 < mu < mu_max): f (xk + mu * r)
        mu = line_search(xk, r, f, gradf, A, C, eps, mu_max, eps, fgradmu)
        logging.info("line search --> mu=", mu)

        if mu == 0.0:
            print "STOP Condition: Line search suggests that there is no improvement possible in this direction (mu=", mu, ", mu_max=", mu_max, ")"
            stop_condition = 1
            break

        oldxk = xk
        # modify xk, and check the new objective
        xk = xk + mu * r


        # check new residuals, and fix the result if needed
        for j in range(0,m):
            # calculate new residual for constraint j
            residual = A[j,:].dot(xk) - C[j]
            if residual > eps/100: # >= 0
                residuals[j] = 0.0
                # need to subtract residual / sum A_j for all xk_i
                for i in range(0,n):
                    if A[j,i] == 1: # only fix if this one is part of it
                        xk[i] -= (residual/sumA[j])
            else:
                residuals[j] = residual
        # xk should be fine now


        obj = f(xk)
        history.append(obj)


        maxres = max(residuals)


        print "Iteration", k, ": Obj=", obj, ", mu_max = ", mu_max, ", mu=", mu, ", sum(x)=", sum(xk), ", maxres=",maxres, ", constraints=", len(active_constraints)

        if abs(lastobj-obj) < goal_acc:
            print "STOP Condition: Relative change < ", goal_acc, ", stopping..."
            stop_condition = 2
            break

        if obj > lastobj:
            print "STOP Condition: Objective is becoming less good.... please check!"
            #plotmu(oldxk, r, f, gradf, A, C, eps, mu_max, eps)
            stop_condition = 5

        # update objective value and store it for next iteration
        lastobj = obj

    print("Done!")

    #plt.plot(history)
    #plt.show()
    optimal = False
    if check_KKT(A,C,xk,gradf,eps):
        print "Result is (close enough to) optimal!"
        optimal = True

    return {'xk': xk, 'k': k, 'stop': stop_condition, 'optimal': optimal, 'history': history}

def check_KKT(A, C, x, gradf, eps):
    m = A.shape[0]
    n = A.shape[1]

    gradvalue = gradf(x)
    residuals = A.dot(x) - C

    # figure out active constraints
    active_constraints = []

    for j in range(0, m):
        if residuals[j] >= -eps:
            active_constraints.append(j)
        # end if
    # end for

    M = A[active_constraints, :]
    MT = M
    M = M.transpose()

    # calculate lagrange multiplicators
    u = -lin.pinv(MT.dot(M)).dot(MT).dot(gradvalue)

    if min(u) < -0.01:
        print "Negative Lagrange multipliers found! Result not optimal!"
        print u
        return False

    # determine the real lagrange multiplicators (0 or u_j)
    real_lagrange = np.zeros(m)
    cnt = 0
    activecnt = 0

    for j in range(0, m):
        if residuals[j] >= -eps:
            # active
            real_lagrange[cnt] = u[activecnt]
            activecnt += 1
        # else: inactive, lagrange must be 0
        cnt += 1

    # sum them up
    sumLagrange = np.zeros(n)
    for j in range(0,m):
        sumLagrange += real_lagrange[j] * A[j,:]

    nabla_lagrange = gradvalue + sumLagrange
    if max(abs(nabla_lagrange)) < 0.001:
        return True
    else:
        print "nabla_lagrange is not 0"
        print nabla_lagrange
        return False