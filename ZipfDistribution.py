__author__ = 'ckreuz'

import random
import sys

def calculate_zipf_density_table(numberOfContents, alpha):
    # calculate sum
    zipf_sum = 0.0
    for i in range(1,numberOfContents+1):
        zipf_sum += (1.0/(i**alpha))
    # calculate table
    zipf_table = []
    for i in range(1,numberOfContents+1):
        zipf_table.append((1.0/(i**alpha)) / zipf_sum)

    return zipf_table


def calculate_zipf_density_table_cumulative(zipf_table):
    local_sum = 0.0
    # calculate sums and assign
    for i in range(0,len(zipf_table)):
        local_sum += zipf_table[i]
        # assign new value (= sum)
        zipf_table[i] = local_sum
    return zipf_table

def sample_requests_from_zipf_distribution(zipfAlpha, clients, servers):
    requests = []

    origZipfTable = calculate_zipf_density_table(len(servers), zipfAlpha)
    zipfTable = calculate_zipf_density_table_cumulative(origZipfTable[:])

    for client in clients.keys():
        randomNumber = random.uniform(0.0, 1.0)
        # find randomNumber in zipfTable
        serverNumber = 0
        while randomNumber > zipfTable[serverNumber]:
            serverNumber += 1
        # request from servers[serverNumber]
        request = { "from": client, "to": servers.keys()[serverNumber] }
        requests.append(request)
    return requests