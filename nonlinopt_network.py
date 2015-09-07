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
    mu_max = -max([ (x[i]/r[i] if r[i] < -eps else -float("inf") ) for i in range(0,len(x))])
    mu_max = mu_max * (1.0 - eps) # just to be sure

    impact = A.dot(r)

    for j in range(0,len(C)):
        if abs(impact[j]) > eps: # yes this inequality has an impact on the result here
            # check if inequality is not active
            val  = A[j,:].dot(x) - C[j]
            if val < -eps: # it was active, we can improve it
                mu = (-val)/impact[j]
                if mu > eps and mu < mu_max: # we only want positive mu
                    mu_max = mu
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

    grad_mumin = fgradmu(xk,r,mu_min)
    grad_mumax = fgradmu(xk,r,mu_max)

    print "grad_mumin=", grad_mumin, ", grad_mumax=", grad_mumax

    if abs(grad_mumax) < eps:
        print "Optimum at mu_max (condition 1 met)"
        return mu_max

    if grad_mumin * grad_mumax < 0:
        print "There exists a global minimum somewhere here..."
    else:
        if f(xk + mu_max * r) > f(xk + mu_min * r):
            print "Optimum at mu_min (condition 2 met)"
            return mu_min
        else:
            print "Optimum at mu_max (condition 3 met)"
            return mu_max

    #plotmu(xk,r,f,gradf,A,C,mu_min, mu_max,eps)

    # perform newton search to find f'(mu) = 0
    newmu = mu = mu_max / 2.0
    print "approaching optimum with newton, starting at mu=", mu
    # this is easy now, just do a newton search, starting at mu=mu_min

    lastgradmu = 1000
    curgradmu = 1000

    while True:
        newmu = mu + secondorder(xk,r,mu)

        if newmu < 0:
            newmu = eps

        while newmu > mu_max:
            print "Repairing newton solution..."
            diff = newmu - mu
            newmu -= diff/2


        newgradmu = fgradmu(xk,r,newmu)

        print "newmu=", newmu, "newgradmu=", newgradmu


        mu = newmu

        if abs(curgradmu - newgradmu) < eps:
            break
        curgradmu = newgradmu
        #if mu > mu_max:

    # end while

    print"mu = ", newmu
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



def nlp_optimize_network(A,C,x0,f,gradf,max_iterations=1000,line_search=linear_decreasing_line_search,eps=0.00000001,fgradmu=gradmu):
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
    AT = A.transpose()


    xk = x0
    P = None # projection matrix
    lastActiveConstraints = []

    for k in range(1,max_iterations+1):
        print "Iteration",k
        constraints = A.dot(xk) - C

        # figure out active constraints
        active_constraints = []

        for j in range(0, m):
            if constraints[j] >= -eps:
                active_constraints.append(j)
            # end if
        # end for

        # check if active constraints differ from last iteration
        if active_constraints != lastActiveConstraints and len(active_constraints) > 0:
            print "We have",len(active_constraints), "active constraints"
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
        elif len(active_constraints) == 0:
            # means we do not need to project anything
            P = np.identity(n)
        # end if

        # store currently active constraints for next iteration
        lastActiveConstraints = active_constraints

        if P.max() == 0:
            print "STOP Condition: Projection matrix is 0. Exiting..."
            break

        gradbefore = gradf(xk)

        # calculate projected descent vector
        r = -P.dot(gradbefore)

        mu_max = get_max_feasible_mu(A,C,xk,r,eps)

        print "mu_max = ", mu_max, ", norm(r)=", lin.norm(r)
        # TODO: perform line search
        mu = line_search(xk, r, f, gradf, A, C, eps, mu_max, eps, fgradmu)


        #mu1 = gradient_based_line_search(xk, r, f, gradf, A, C, eps, mu_max, eps)
        #mu2 = linear_decreasing_line_search(xk, r, f, gradf, A, C, eps, mu_max, eps)

        #if abs(mu1-mu2) > eps:
        #    print "Gradient based approach Gradmu=", mu1, ", linearmu= ", mu2
        #    print "Who is right?"
        #    plotmu(xk, r, f, gradf, A, C, eps, mu_max, eps)
        #    print "vals=", f(xk + mu1 * r), "; ", f(xk + mu2 * r), "==> f(xk + mu1 * r) < f(xk + mu2 * r) =", f(xk + mu1 * r) < f(xk + mu2 * r)
        # end if
        #print "gradient mu=", mu1, "; linear mu=", mu2
        # who is right?
        #print "vals=", f(xk + mu1 * r), "; ", f(xk + mu2 * r), "==>", f(xk + mu1 * r) < f(xk + mu2 * r)


        oldxk = xk
        # modify xk, and check the new objective
        xk = xk + mu * r

        gradafter = gradf(xk)

        if min(xk) < eps:
            print "Found a bad xk in step ", k, ", this could lead to a crash..."
            print "mu=", mu, ", mu_max=", mu_max
            print xk
            print "norm(r)=", lin.norm(r)
            plotmu(oldxk, r, f, gradf, A, C, eps, mu_max, eps)

        obj = f(xk)
        history.append(obj)
        print("Objective=",obj)

        if lin.norm(lastobj-obj) < eps/100:
            print("STOP Condition: Relative change < eps/100, stopping...")
            break

        lastobj = obj

    print("Done!")


    #plt.plot(history)
    #plt.show()

    return {'xk': xk, 'k': k}

