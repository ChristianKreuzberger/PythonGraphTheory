#!/usr/bin/env python
"""Some mathematic utils for networks
"""
__author__ = 'Christian Kreuzberger'
__email__ = 'christian.kreuzberger@itec.aau.at'
__license__ = 'Not decided yet'
__maintainer__ = 'Christian Kreuzberger'
__status__= 'Development'


import math


def c_sum_flows(flows):
    return sum(flows)

def c_log_sum_flows(flows):
    return sum([math.log(x) for x in flows])

def c_neg_log_sum_flows(flows):
    return -c_log_sum_flows(flows)

