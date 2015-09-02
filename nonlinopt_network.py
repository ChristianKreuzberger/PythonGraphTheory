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


import matplotlib.pyplot as plt




def is_feasible(A,C,x,eps):
    """ checks whether A * x <= C
    :param A:
    :param c:
    :param x:
    :return: True if A * x <= C
    """
    return max(A.dot(x) - C) <= eps



def get_max_feasible_mu(A,C,x,r, eps):
    mu_max = -max([ (x[i]/r[i] if r[i] < 0 else -float("inf") ) for i in range(0,len(x))])
    mu_max = mu_max * (1.0 - eps)

    impact = A.dot(r)

    for j in range(0,len(C)):
        if abs(impact[j]) > eps: # yes this inequality has an impact on the result here
            # check if inequality is not active
            val  = A[j,:].dot(x) - C[j]
            if val < -eps: # it was active, we can improve it
                mu = (-val)/impact[j]
                if mu > eps and mu < mu_max: # we only want positive mu
                    mu_max = mu
    return mu_max



def gradmu(xk,r,mu):
    return -sum([ r[i] / (xk[i] + mu * r[i]) for i in range(0,len(xk)) ])

def secondorder(xk,r,mu):
    upper = gradmu(xk,r,mu)
    lower = -sum([ math.pow(r[i],2) / math.pow(xk[i] + mu * r[i],2) for i in range(0,len(xk)) ])
    return upper/lower

def gradient_based_line_search(xk,r,f,gradf,A,C,mu_min, mu_max,eps):
    # check gradmu(0) * gradmu(mu_max)
    if gradmu(xk,r,mu_min) * gradmu(xk,r,mu_max) < 0:
        print "There exists a global minimum somewhere here..."
    else:
        if f(xk + mu_max * r) > f(xk + mu_min * r):
            print "Optimum at mu_min"
            return mu_min
        else:
            print "Optimum at mu_max"
            return mu_max

    #plotmu(xk,r,f,gradf,A,C,mu_min, mu_max,eps)

    # perform newton search to find f'(mu) = 0
    newmu = mu = mu_min
    print "approaching optimum with newton, starting at mu=", mu
    # this is easy now, just do a newton search, starting at mu=mu_min

    while mu < mu_max:
        newmu = mu + secondorder(xk,r,mu)
        #print "newmu=", newmu
        if abs(newmu - mu) < 0.0001:
            break
        mu = newmu
    # end while

    print"mu = ", newmu
    return newmu


def plotmu(xk, r, f, gradf, A, C, mu_min, mu_max,eps):
    mu = mu_max

    diff = float(mu_max - mu_min)

    low_val = float("inf")
    low_mu = -1
    values = []
    mus = []
    gradvals = []
    while mu > mu_min:
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
        else:
            print "Infeasible for mu=",mu,"(mu_max=",mu_max,")"

        # end if
        mu = mu - diff/1000.0
    # end while

    mu = low_mu
    while not is_feasible(A,C, xk + mu * r, eps):
        mu = mu / 2.0
        print ("not feasible!")

    # plot values
    plt.figure(1)
    plt.subplot(211)
    plt.plot(mus,values,'r')

    plt.subplot(212)
    plt.plot(mus,gradvals, 'b')
    plt.show()

    return mu

def linear_decreasing_line_search(xk, r, f, gradf, A, C, mu_min, mu_max,eps):
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


def smart_linear_decreasing_line_search(xk, r, f, gradf, A, C, mu_min, mu_max,eps):
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
            elif low_mu > 0: # it increased --> stop
                break
            # end if
        else:
            print "Infeasible for mu=",mu,"(mu_max=",mu_max,")"

        # end if
        mu = mu - diff/100.0
    # end while

    mu = low_mu

    return mu



def nlp_optimize_network(A,C,x0,f,gradf,max_iterations=1000,line_search=linear_decreasing_line_search,eps=0.00000001):
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
    :return:
    """

    A = np.array(A)

    m = A.shape[0]
    n = A.shape[1]
    lastobj = obj = f(x0)


    history = []

    xk = x0
    P = None # projection matrix
    lastActiveConstraints = []

    for k in range(1,max_iterations+1):
        print "Iteration",k
        constraints = A.dot(xk) - C

        # figure out active constraints
        activeConstraints = []

        for j in range(0,m):
            if constraints[j] >= -eps:
                activeConstraints.append(j)
            # end if
        # end for

        # check if active constraints differ from last iteration
        if activeConstraints != lastActiveConstraints:
            print "We have",len(activeConstraints), "active constraints"

            M = A[activeConstraints,:]
            MT = M
            M = M.transpose()

            # calculate projection matrix
            P = np.identity(n) - M.dot(lin.inv(MT.dot(M))).dot(MT)
        # end if

        # store currently active constraints for next iteration
        lastActiveConstraints = activeConstraints

        # calculate projected descent vector
        r = -P.dot(gradf(xk))

        # make sure that r is actually 0 when it is needed
        for i in range(0,len(xk)):
            if abs(r[i]) < eps:
                r[i] = 0

        # normalize r
        # r = r/lin.norm(r)

        # TODO: perform line search
        mu = line_search(xk, r, f, gradf, A, C, eps, get_max_feasible_mu(A,C,xk,r,eps), eps)


        #mu1 = gradient_based_line_search(xk, r, f, gradf, A, C, eps, get_max_feasible_mu(A,C,xk,r,eps), eps)
        #mu2 = linear_decreasing_line_search(xk, r, f, gradf, A, C, eps, get_max_feasible_mu(A,C,xk,r,eps), eps)

        #print "gradient mu=", mu1, "; linear mu=", mu2
        # who is right?
        #print "vals=", f(xk + mu1 * r), "; ", f(xk + mu2 * r), "==>", f(xk + mu1 * r) < f(xk + mu2 * r)



        # modify xk, and check the new objective
        xk = xk + mu * r
        obj = f(xk)
        history.append(obj)
        print("Objective=",obj)

        if lin.norm(lastobj-obj) < eps/100:
            print("Relative change < eps, stopping...")
            break

        lastobj = obj

    print("Done!")


    #plt.plot(history)
    #plt.show()

    return xk

