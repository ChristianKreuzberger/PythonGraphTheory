#!/usr/bin/env python
"""Some mathematic utils for networks
"""
__author__ = 'Christian Kreuzberger'
__email__ = 'christian.kreuzberger@itec.aau.at'
__license__ = 'Not decided yet'
__maintainer__ = 'Christian Kreuzberger'
__status__= 'Development'


import math
import numpy as np


def c_sum_flows(flows):
    return sum(flows)


def c_neg_sum_flows(flows):
    return -sum(flows)


def c_log_sum_flows(flows):
    return sum([(math.log(x) if x > 0 else -1000000) for x in flows])

def c_neg_log_sum_flows(flows):
    return -c_log_sum_flows(flows)

def c_grad_neg_log_sum_flows(flows):
    return [-1/xk for xk in flows]


def np_array_to_mathematica(arr,nfor='{:1.0f}'):
    """
    :type arr: np.array
    :param arr: the array that should be converted to a mathematica string
    :return: the mathematica string
    """
    arr = np.array(arr)

    if arr.ndim == 2:
        rowlist = []

        for row in arr:
            a = ",".join([nfor.format(i) for i in row])
            rowlist.append( "{" + a + "}")

        matrix = ",\n".join(rowlist)

        return "{" + matrix + "}"
    else: # arr.ndim == 1
        a = ",".join([nfor.format(i) for i in arr])
        return "{" + a + "}"


def np_array_to_matlab(arr,nfor='{:1.0f}'):
    """
    :type arr: np.array
    :param arr: the array that should be converted to a matlab string
    :return: the matlab string
    """
    arr = np.array(arr)

    if arr.ndim == 2:
        rowlist = []

        for row in arr:
            a = " ".join([nfor.format(i) for i in row])
            rowlist.append( " " + a + " ")

        matrix = ";\n".join(rowlist)

        return "[" + matrix + "]"
    else: # arr.ndim == 1
        a = " ".join([nfor.format(i) for i in arr])
        return "[" + a + "]"
