import os
import numpy as np
import pandas as pd
import sys

def analyze_solutions(res_solutions, mode, n_p_l0, n_p_l1, n_np_l0, n_np_l1):
    best_soln = None
    if mode == 'under':
        best_value = -sys.maxsize
    if mode == 'over':
        best_value = sys.maxsize

    for soln in res_solutions:
        if mode == 'under':
            print(soln)
            print(soln[0] <= n_p_l0, soln[1] <= n_p_l1, soln[2]<=n_np_l0, soln[3]<= n_np_l1)

            if( (soln[0] <= n_p_l0) and (soln[1] <= n_p_l1) and (soln[2] <= n_np_l0) and (soln[3] <= n_np_l1)):
                value = (n_p_l0 - soln[0]) + (n_p_l1 - soln[1]) + (n_np_l0 - soln[2]) + (n_np_l1 - soln[3])

                print(soln, value)
                if value > best_value:
                    best_value = value
                    best_soln = soln
        else:
            print(soln)
            print(soln[0] >= n_p_l0, soln[1] >= n_p_l1, soln[2]>=n_np_l0, soln[3] >= n_np_l1)
            if((soln[0] >= n_p_l0) and (soln[1] >= n_p_l1) and (soln[2] >= n_np_l0) and (soln[3] >= n_np_l1)):
                value = (soln[0] - n_p_l0) + (soln[1] - n_p_l1) + (soln[2] - n_np_l0) + (soln[3] - n_np_l1)
                print(value)
                if value < best_value:
                    best_value = value
                    best_soln = soln
    
    return best_soln[0], best_soln[1], best_soln[2], best_soln[3]

# VERSION 2A 
def sample_orig_50_50_undersample(orig_fraction, n_p_l0, n_p_l1, n_np_l0, n_np_l1):
    res_solutions = []
    '''
    Fix P_L0
    '''
    orig_fraction = float(n_p_l0 + n_p_l1)/(n_np_l0 + n_np_l1)
    print("orig_fraction="+str(orig_fraction))
    
    req_p_l0 = int(n_p_l0)
    req_p_l1 = int(n_p_l0)
    
    req_p = req_p_l0 + req_p_l1
    req_np = int((1.0/orig_fraction) * req_p)
    
    req_np_l0 = int(req_np/2.0)
    req_np_l1 = int(req_np/2.0)
    res_solutions.append([req_p_l0, req_p_l1, req_np_l0, req_np_l1])
    
    '''
    FIX P_L1
    '''
    req_p_l1 = int(n_p_l1)
    req_p_l0 = int(n_p_l1)

    req_p = req_p_l0 + req_p_l1
    req_np = int((1.0/orig_fraction) * req_p)
    
    req_np_l0 = int(req_np/2.0)
    req_np_l1 = int(req_np/2.0)
    res_solutions.append([req_p_l0, req_p_l1, req_np_l0, req_np_l1])

    '''
    FIX NP_L0
    '''
    req_np_l0 = int(n_np_l0)
    req_np_l1 = int(n_np_l0)
    
    req_np = req_np_l0 + req_np_l1
    req_p = int(orig_fraction * req_np)
    
    req_p_l0 = int(req_p/2.0)
    req_p_l1 = int(req_p/2.0)
    
    res_solutions.append([req_p_l0, req_p_l1, req_np_l0, req_np_l1])

    '''
    FIX NP_L1
    '''
    req_np_l1 = int(n_np_l1)
    req_np_l0 = int(n_np_l1)
    
    req_np = req_np_l0 + req_np_l1
    req_p = int(orig_fraction * req_np)
    
    req_p_l0 = int(req_p/2.0)
    req_p_l1 = int(req_p/2.0)
    
    res_solutions.append([req_p_l0, req_p_l1, req_np_l0, req_np_l1])

    best_solution = analyze_solutions(res_solutions, 'under', n_p_l0, n_p_l1, n_np_l0, n_np_l1)

    return best_solution

def sample_orig_50_50_oversample(orig_fraction, n_p_l0, n_p_l1, n_np_l0, n_np_l1):
    orig_fraction = float(n_p_l0 + n_p_l1)/(n_np_l0 + n_np_l1)

    res_solutions = []
    '''
    Fix P_L0
    '''
    req_p_l0 = int(n_p_l0)
    req_p_l1 = int(n_p_l0)
    
    req_p = req_p_l0 + req_p_l1
    req_np = int((1.0/orig_fraction) * req_p)

    req_np_l0 = int(req_np/2.0)
    req_np_l1 = int(req_np/2.0)

    res_solutions.append([req_p_l0, req_p_l1, req_np_l0, req_np_l1])
    
    '''
    FIX P_L1
    '''
    req_p_l1 = int(n_p_l1)
    req_p_l0 = int(n_p_l1)

    req_p = req_p_l0 + req_p_l1
    req_np = int((1.0/orig_fraction) * req_p)
    
    req_np_l0 = int(req_np/2.0)
    req_np_l1 = int(req_np/2.0)

    res_solutions.append([req_p_l0, req_p_l1, req_np_l0, req_np_l1])

    '''
    FIX NP_L0
    '''
    req_np_l0 = int(n_np_l0)
    req_np_l1 = int(n_np_l0)

    req_np = req_np_l0 + req_np_l1
    req_p = int(orig_fraction * req_np)

    req_p_l0 = int(req_p/2.0)
    req_p_l1 = int(req_p/2.0)

    res_solutions.append([req_p_l0, req_p_l1, req_np_l0, req_np_l1])

    '''
    FIX NP_L1
    '''
    req_np_l1 = int(n_np_l1)
    req_np_l0 = int(n_np_l1)

    req_np = req_np_l0 + req_np_l1
    req_p = int(orig_fraction * req_np)
    
    req_p_l0 = int(req_p/2.0)
    req_p_l1 = int(req_p/2.0)
    
    res_solutions.append([req_p_l0, req_p_l1, req_np_l0, req_np_l1])

    best_solution = analyze_solutions(res_solutions, 'over', n_p_l0, n_p_l1, n_np_l0, n_np_l1)

    return best_solution

# VERSION 2B
def sample_orig_50_orig_undersample(orig_fraction, n_p_l0, n_p_l1, n_np_l0, n_np_l1):
    res_solutions = []
    orig_fraction = (n_p_l0 + n_p_l1)/(n_np_l0 + n_np_l1)

    '''
    Fix P_L0
    '''
    req_p_l0 = int(n_p_l0)
    req_p_l1 = int(n_p_l0)
    
    np_ldist = n_np_l0/(n_np_l0 + n_np_l1)
    
    req_np = int((1.0/orig_fraction) * (req_p_l0+req_p_l1))
    req_np_l0 = int(np_ldist * req_np)
    req_np_l1 = int(req_np - req_np_l0)
    res_solutions.append([req_p_l0, req_p_l1, req_np_l0, req_np_l1])
    
    '''
    FIX P_L1
    '''
    req_p_l1 = int(n_p_l1)
    req_p_l0 = int(n_p_l1)
    np_ldist = n_np_l0/(n_np_l0 + n_np_l1)
    
    req_np = int((1.0/orig_fraction) * (req_p_l0+req_p_l1))
    req_np_l0 = int(np_ldist * req_np)
    req_np_l1 = int(req_np - req_np_l0)
    res_solutions.append([req_p_l0, req_p_l1, req_np_l0, req_np_l1])

    best_solution = analyze_solutions(res_solutions, 'under', n_p_l0, n_p_l1, n_np_l0, n_np_l1)

    return best_solution

def sample_orig_50_orig_oversample(orig_fraction, n_p_l0, n_p_l1, n_np_l0, n_np_l1):
    res_solutions = []
    orig_fraction = (n_p_l0 + n_p_l1)/(n_np_l0 + n_np_l1)

    '''
    Fix P_L0
    '''
    req_p_l0 = int(n_p_l0)
    req_p_l1 = int(n_p_l0)
    
    np_ldist = n_np_l0/(n_np_l0 + n_np_l1)
    
    req_np = int((1.0/orig_fraction) * (req_p_l0+req_p_l1))
    req_np_l0 = int(np_ldist * req_np)
    req_np_l1 = int(req_np - req_np_l0)
    res_solutions.append([req_p_l0, req_p_l1, req_np_l0, req_np_l1])
    
    '''
    FIX P_L1
    '''
    req_p_l1 = int(n_p_l1)
    req_p_l0 = int(n_p_l1)
    np_ldist = n_np_l0/(n_np_l0 + n_np_l1)
    
    req_np = int((1.0/orig_fraction) * (req_p_l0+req_p_l1))
    req_np_l0 = int(np_ldist * req_np)
    req_np_l1 = int(req_np - req_np_l0)
    res_solutions.append([req_p_l0, req_p_l1, req_np_l0, req_np_l1])

    best_solution = analyze_solutions(res_solutions, 'over', n_p_l0, n_p_l1, n_np_l0, n_np_l1)
    
    return best_solution

# VERSION 2C
def sample_orig_snop_orig_undersample(orig_fraction, n_p_l0, n_p_l1, n_np_l0, n_np_l1):
    res_solutions = []
    orig_fraction = (n_p_l0 + n_p_l1)/(n_np_l0 + n_np_l1)
    ldist_np = n_np_l0/n_np_l1
    
    '''
    FIX P_0
    '''
    req_p_l0 = int(n_p_l0)
    req_p_l1 = int((1.0/ldist_np) * req_p_l0)
    
    req_p = req_p_l0 + req_p_l1
    req_np = (1.0/orig_fraction) * req_p
    
    req_np_l0 = int((n_np_l0/(n_np_l0+n_np_l1)) * req_np)
    req_np_l1 = int(req_np - req_np_l0)

    res_solutions.append([req_p_l0, req_p_l1, req_np_l0, req_np_l1])

    '''
    FIX P_1
    '''
    req_p_l1 = int(n_p_l1)
    req_p_l0 = int((ldist_np) * req_p_l1)

    req_p = req_p_l0 + req_p_l1
    req_np = orig_fraction * req_p

    req_np_l0 = int((n_np_l0/(n_np_l0 + n_np_l1)) * req_np)
    req_np_l1 = int(req_np - req_np_l0)

    res_solutions.append([req_p_l0, req_p_l1, req_np_l0, req_np_l1])

    '''
    FIX NP_0 & NP_1
    '''
    req_np_0 = int(n_np_l0)
    req_np_1 = int(n_np_l1)

    req_np = req_np_0 + req_np_1
    req_p = (1.0/orig_fraction) * req_np

    req_p_l0 = int((n_np_l0/(n_np_l0 + n_np_l1)) * req_p)
    req_p_l1 = int(req_p - req_p_l0)

    res_solutions.append([req_p_l0, req_p_l1, req_np_l0, req_np_l1])

    best_solution = analyze_solutions(res_solutions, 'under', n_p_l0, n_p_l1, n_np_l0, n_np_l1)
    
    return best_solution

def sample_orig_snop_orig_oversample(orig_fraction, n_p_l0, n_p_l1, n_np_l0, n_np_l1):
    res_solutions = []
    orig_fraction = (n_p_l0 + n_p_l1)/(n_np_l0 + n_np_l1)
    ldist_np = n_np_l0/n_np_l1
    
    '''
    FIX P_0
    '''
    req_p_l0 = int(n_p_l0)
    req_p_l1 = int((1.0/ldist_np) * req_p_l0)
    
    req_p = req_p_l0 + req_p_l1
    req_np = (1.0/orig_fraction) * req_p
    
    req_np_l0 = int((n_np_l0/(n_np_l0+n_np_l1)) * req_np)
    req_np_l1 = int(req_np - req_np_l0)

    res_solutions.append([req_p_l0, req_p_l1, req_np_l0, req_np_l1])

    '''
    FIX P_1
    '''
    req_p_l1 = int(n_p_l1)
    req_p_l0 = int((ldist_np) * req_p_l1)

    req_p = req_p_l0 + req_p_l1
    req_np = (1.0/orig_fraction) * req_p

    req_np_l0 = int((n_np_l0/(n_np_l0 + n_np_l1)) * req_np)
    req_np_l1 = int(req_np - req_np_l0)

    res_solutions.append([req_p_l0, req_p_l1, req_np_l0, req_np_l1])

    '''
    FIX NP_0 & NP_1
    '''
    req_np_0 = int(n_np_l0)
    req_np_1 = int(n_np_l1)

    req_np = req_np_0 + req_np_1
    req_p = (1.0/orig_fraction) * req_np

    req_p_l0 = int((n_np_l0/(n_np_l0 + n_np_l1)) * req_p)
    req_p_l1 = int(req_p - req_p_l0)

    res_solutions.append([req_p_l0, req_p_l1, req_np_l0, req_np_l1])

    best_solution = analyze_solutions(res_solutions, 'over', n_p_l0, n_p_l1, n_np_l0, n_np_l1)
    
    return best_solution


# VERSION 3A
def sample_1_50_50_undersample(orig_fraction, n_p_l0, n_p_l1, n_np_l0, n_np_l1):
    min_x = int(min(n_p_l0, n_p_l1, n_np_l0, n_np_l1))

    req_p_l0 = min_x
    req_p_l1 = min_x
    req_np_l0 = min_x
    req_np_l1 = min_x

    res_solution = [[req_p_l0, req_p_l1, req_np_l0, req_np_l1]]

    best_solution = analyze_solutions(res_solution, 'under', n_p_l0, n_p_l1, n_np_l0, n_np_l1)

    return best_solution

def sample_1_50_50_oversample(orig_fraction, n_p_l0, n_p_l1, n_np_l0, n_np_l1):
    max_x = int(max(n_p_l0, n_p_l1, n_np_l0, n_np_l1))

    req_p_l0 = max_x
    req_p_l1 = max_x
    req_np_l0 = max_x
    req_np_l1 = max_x

    res_solution = [[req_p_l0, req_p_l1, req_np_l0, req_np_l1]]

    best_solution = analyze_solutions(res_solution, 'over', n_p_l0, n_p_l1, n_np_l0, n_np_l1)

    return best_solution

# VERSION 3B
def sample_1_snop_orig_undersample(orig_fraction, n_p_l0, n_p_l1, n_np_l0, n_np_l1):
    res_solutions = []

    np_ldist = float((n_np_l0)/(n_np_l1))

    '''
    FIX P_L0
    '''
    req_p_l0 = int(n_p_l0)
    req_p_l1 = int((1.0/np_ldist) * req_p_l0)

    req_p = req_p_l0 + req_p_l1
    req_np = req_p

    frac = float(n_np_l0)/float(n_np_l0+n_np_l1)

    req_np_l0 = int(frac * req_np)
    req_np_l1 = int(req_np - req_np_l0)

    res_solutions.append([req_p_l0, req_p_l1, req_np_l0, req_np_l1])

    '''
    FIX P_L1
    '''
    req_p_l1 = int(n_p_l1)
    req_p_l0 = int(np_ldist * req_p_l1)

    req_p = req_p_l0 + req_p_l1
    req_np = req_p

    frac = float(n_np_l0)/float(n_np_l0 + n_np_l1)

    req_np_l0 = int(frac * req_np)
    req_np_l1 = int(req_np - req_np_l0)

    res_solutions.append([req_p_l0, req_p_l1, req_np_l0, req_np_l1])


    '''
    FIX NP_L0 and hence also NP_L1 -- But that will also mean ends up balancing out P_L0 and also P_L1
    '''
    req_np_l0 = int(n_np_l0)
    req_np_l1 = int(n_np_l1)

    req_p = req_np

    frac = float(n_np_l0)/float(n_np_l0 + n_np_l1)

    req_p_l0 = int(frac * req_p)
    req_p_l1 = int(req_p - req_p_l0)
    
    res_solutions.append([req_p_l0, req_p_l1, req_np_l0, req_np_l1])

    best_solution = analyze_solutions(res_solutions, 'under', n_p_l0, n_p_l1, n_np_l0, n_np_l1)

    return best_solution

def sample_1_snop_orig_oversample(orig_fraction, n_p_l0, n_p_l1, n_np_l0, n_np_l1):
    res_solutions = []

    np_ldist = float((n_np_l0)/(n_np_l1))

    '''
    FIX P_L0
    '''
    req_p_l0 = int(n_p_l0)
    req_p_l1 = int((1.0/np_ldist) * req_p_l0)
    req_p = req_p_l0 + req_p_l1
    
    req_np = req_p

    frac = float(n_np_l0)/float(n_np_l0 + n_np_l1)

    req_np_l0 = int(frac * req_np)
    req_np_l1 = int(req_np - req_np_l0)

    print("Fixing PL0")
    print(req_p_l0, req_p_l1, req_np_l0, req_np_l1)
    res_solutions.append([req_p_l0, req_p_l1, req_np_l0, req_np_l1])

    '''
    FIX P_L1
    '''
    req_p_l1 = int(n_p_l1)
    req_p_l0 = int(np_ldist * req_p_l1)

    req_p = req_p_l0 + req_p_l1
    req_np = req_p

    frac = float(n_np_l0)/float(n_np_l0 + n_np_l1)

    req_np_l0 = int(frac * req_np)
    req_np_l1 = int(req_np - req_np_l0)
    
    print("Fixing PL1")
    print(req_p_l0, req_p_l1, req_np_l0, req_np_l1)

    res_solutions.append([req_p_l0, req_p_l1, req_np_l0, req_np_l1])

    '''
    FIX NP_L0 and hence also NP_L1 -- But that will also mean ends up balancing out P_L0 and also P_L1
    '''
    req_np_l0 = int(n_np_l0)
    req_np_l1 = int(n_np_l1)

    req_p = req_np_l0 + req_np_l1
    frac = float(req_np_l0)/float(req_np)
    
    print("req p = "+str(req_p))
    req_p_l0 = int(frac * req_p)
    req_p_l1 = int(req_p - req_p_l0)

    req_p_l0 = int((float(n_np_l0)/(n_np_l0 + n_np_l1)) * req_p)
    req_p_l1 = int(req_p - req_p_l0)

    res_solutions.append([req_p_l0, req_p_l1, req_np_l0, req_np_l1])

    print("FiXING nPl0, npl1")
    print(req_p_l0, req_p_l1, req_np_l0, req_np_l1)
    
    best_solution = analyze_solutions(res_solutions, 'over', n_p_l0, n_p_l1, n_np_l0, n_np_l1)

    return best_solution

if __name__ == '__main__':
    print("ORIG-50-50-UNDER")
    best_solution = sample_orig_50_50_undersample(0.7065217391, 27968, 11227, 42360, 13116)
    print(best_solution)
    
    print("="*30)
    
    print("ORIG-50-50-OVER")
    best_solution = sample_orig_50_50_oversample(0.7065217391, 27968, 11227, 42360, 13116)
    print(best_solution)

    print("="*30)

    print("ORIG-50-ORIG-UNDER")
    best_solution = sample_orig_50_orig_undersample(0.7065217391, 27968, 11227, 42360, 13116)
    print(best_solution)

    print("="*30)

    print("ORIG-50-ORIG-OVER")
    best_solution = sample_orig_50_orig_oversample(0.7065217391, 27968, 11227, 42360, 13116)
    print(best_solution)

    print("="*30)

    print("ORIG-SNOP-ORIG-UNDER")
    best_solution = sample_orig_snop_orig_undersample(0.7065217391, 27968, 11227, 42360, 13116)
    print(best_solution)

    print("="*30)

    print("ORIG-SNOP-ORIG-OVER")
    best_solution = sample_orig_snop_orig_oversample(0.7065217391, 27968, 11227, 42360, 13116)
    print(best_solution)

    print("="*30)

    print("ORIG-1-50-50-UNDER")
    best_solution = sample_1_50_50_undersample(0.7065217391, 27968, 11227, 42360, 13116)
    print(best_solution)

    print("="*30)

    print("ORIG-1-50-50-OVER")
    best_solution = sample_1_50_50_oversample(0.7065217391, 27968, 11227, 42360, 13116)
    print(best_solution)

    print("="*30)

    print("ORIG-1-SNOP-ORIG-UNDER")
    best_solution = sample_1_snop_orig_undersample(0.7065217391, 27968, 11227, 42360, 13116)
    print(best_solution)

    print("="*30)

    print("ORIG-1-SNOP-ORIG-OVER")
    best_solution = sample_1_snop_orig_oversample(0.7065217391, 27968, 11227, 42360, 13116)
    print(best_solution)