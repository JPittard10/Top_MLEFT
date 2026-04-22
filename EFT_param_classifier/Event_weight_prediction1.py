import numpy as np
from itertools import combinations

def generate_weight_configurations(num_WCs):
    """
    Generates the Wilson coefficient configurations for the 153 weights.
    """
    weight_configs = []
    weight_configs.append([0] * num_WCs)  # First row of all zeros
    
    for i in range(num_WCs):
        config = [0] * num_WCs
        config[i] = 1
        weight_configs.append(config)
    
    for i in range(num_WCs):
        config = [0] * num_WCs
        config[i] = 2
        weight_configs.append(config)
        
        for j in range(i + 1, num_WCs):
            config = [0] * num_WCs
            config[i] = 1
            config[j] = 1
            weight_configs.append(config)
    
    return weight_configs

def associate_weights_to_configs(weight_configs, mg_weights):
    """
    Associates weight values from `mg_weights` to each Wilson coefficient configuration.
    """
    weights_correspon = [mg_weights[:, 200 + i] for i in range(len(weight_configs))]
    return weights_correspon

def obtain_structure_constant(num_WCs, mg_weights):
    """
    Computes structure constants using least squares fitting.
    """
    
    weight_configs = generate_weight_configurations(num_WCs)
    weight_all = associate_weights_to_configs(weight_configs, mg_weights)
    
    num_events = len(weight_all[0])
    structures = []
    weight_matrix = np.stack(weight_all, axis=1)
    
    for i in range(num_events):
        A = []
        for wc_values in weight_configs:
            row = [1] + wc_values + [wc_values[k]**2 for k in range(num_WCs)]
            row += [wc_values[k1] * wc_values[k2] for k1, k2 in combinations(range(num_WCs), 2)]
            A.append(row)
        
        A = np.array(A)
        w = weight_matrix[i]
        s, _, _, _ = np.linalg.lstsq(A, w, rcond=None)
        structures.append(s)
        
    return structures
    

#Compute total,  linear and qudratic term in arbitary WC value
def event_weights_lin_quad(structure_constants, wc_values):
    """
    Computes the total event weight along with linear and quadratic contributions.

    Parameters:
    structure_constants (list): Structure constants [s_0, s_1, ..., s_15, s_11, s_22, ..., s_ij] for a single event.
    wc_values (list): Arbitrary Wilson coefficient values [c1, c2, ..., cN].

    Returns:
    tuple: (total_weight, linear_weight, quadratic_weight)
    """
    num_WCs = len(wc_values)  # Number of Wilson coefficients

    # Construct linear terms (without the constant term)
    linear_terms = wc_values  # (c1, c2, ..., cN)
    s_linear = structure_constants[:, 1:len(linear_terms) + 1]   # Skip s0 and match the rest to the linear terms
    w_linear = np.dot(s_linear, linear_terms)

    # Construct quadratic terms (without the constant term)
    quadratic_terms = [wc_values[k]**2 for k in range(num_WCs)]  # c1^2, c2^2, ..., cN^2
    quadratic_terms.extend([wc_values[k1] * wc_values[k2] for k1, k2 in combinations(range(num_WCs), 2)])  # c1*c2, etc.

    s_quad = structure_constants[:,len(linear_terms) + 1 :]  # Skip s0 and linear terms, match to the quadratic terms
    w_quad = np.dot(s_quad, quadratic_terms)
    sm = structure_constants[:,0]
    # Add the structure constant for the constant term separately
    w_linear_with_sm= sm + w_linear  # Adding s0 to the linear contribution
    w_quad_with_sm= sm + w_quad  # Adding s0 to the quadratic contribution

    # Compute total weight
    total_weight = w_linear_with_sm + w_quad_with_sm - sm #  when this was sm + w_linear + w_quad it gave the wrong values dispite mathematically being the same 

    return sm, w_linear_with_sm, w_quad_with_sm, total_weight
    



