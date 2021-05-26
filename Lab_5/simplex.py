import numpy as np
import re
import pandas as pd

def _convert_string_coefs_into_array(func:str):
    _coefs_func = re.findall(r"[+-]*\d+\*", func)
    coefs_func = []
    for _coef_func in _coefs_func:
        _coef_func = _coef_func.replace("*", '')
        coefs_func.append(int(_coef_func))
    return coefs_func

def _get_indicies_for_constraints(string:str):
    return [int(elem.replace("_", "")) for elem in re.findall(r"\_\d+", string)]

def _get_vec_for_constraints(string:str):
    return int(re.findall("\=\s*\-*\d+", string)[0].replace("=", ""))

def convert_string_problem_into_arrays(func:str, constr_eq:list = None, 
                                constr_less:list = None, constr_gr:list = None,
                                free_vars:str = None):
    '''
    Converts strings, readeble for robots with skin and meat, into np.arrays and indicators
    Params:
        func: function to maximize (written in form c1*x_1+c2*x_2+...+cn*x_n)
        constr_eq: list of equal constraints
        constr_less: list of less constraints
        constr_gr: list of greater constraints
    Returns:
        dictionary with full information about the problem
    '''
    
    # Working with the function
    coefs_func = _convert_string_coefs_into_array(func)
    #Working with equalities
    coefs_eq = []
    for eq in constr_eq:
        coefs_eq.append({"coefs":_convert_string_coefs_into_array(eq), 
                        "ind":_get_indicies_for_constraints(eq), 
                        "val":_get_vec_for_constraints(eq)})
    
    #Working with less equalities
    coefs_l_eq = []
    for eq in constr_less:
        coefs_l_eq.append({"coefs":_convert_string_coefs_into_array(eq), 
                            "ind":_get_indicies_for_constraints(eq), 
                            "val":_get_vec_for_constraints(eq)})

    #Working with greater equalities
    coefs_g_eq = []
    for eq in constr_gr:
        coefs_g_eq.append({"coefs":_convert_string_coefs_into_array(eq), 
                            "ind":_get_indicies_for_constraints(eq), 
                            "val":_get_vec_for_constraints(eq)})

    return {"func": coefs_func,
            "equal": coefs_eq,
            "less": coefs_l_eq, 
            "greater": coefs_g_eq, 
            "free_vars": _get_indicies_for_constraints(free_vars) if free_vars is not None else []}



def canonic_form_of_problem(initial_pr:dict):
    '''
    Matches the initial problem into the canonic form written in the form:
    <c, x> -> min
    Ax = b; x >= 0; b>= 0;
    Params:
        initial_pr: dict got from convert_string_problem_into_arrays
    Returns:
        A: np.array
        b: np.array
        c: np.array
        add: int - number of added vars
    '''
    func, equal, less, greater = initial_pr["func"], initial_pr["equal"], initial_pr["less"], initial_pr["greater"]
    free_vars = initial_pr["free_vars"]
    #detecting x_k <= 0
    leq_vars = []
    for leq in less:
        if not leq["val"] and len(leq["coefs"]) == 1:
            leq_vars.append(leq["ind"][0])
            less.remove(less[less.index(leq)-1])
    n = len(func)
    vars = [i+1 for i in range(len(func))]
    for var in free_vars:
        vars.insert(var, var)
        func.insert(vars.index(var)+1, -func[vars.index(var)])
    
    #function
    for var in leq_vars:
        func[vars.index(var)]*=-1
    #vars = [i+1 for i in range(len(func))]
    var_start_index = [vars.index(i) for i in range(1, n+1)]
    leq_vars = [vars.index(i) for i in leq_vars]
    #equalities
    initial_pr["func"] = np.array(func)
    initial_pr["func"]=-1*initial_pr["func"]

    for eq in equal:
        list_eq = eq["ind"].copy()
        eq["ind"] = [var_start_index[i-1] for i in list_eq]
        list_index = [var_start_index[i-1] for i in list_eq]
        for var in free_vars:
            if var in list_eq:
                eq["ind"].insert(list_index[var-1]+1, list_index[var-1]+1)
                eq["coefs"].insert(list_index[var-1]+1, -eq["coefs"][list_index[var-1]])
        eq["coefs"] = np.array(eq["coefs"])
        if eq["val"] < 0:
            eq["coefs"]*=-1
            eq["val"]*=-1
        for var in leq_vars:
            if var in eq["ind"]:
                eq["coefs"][eq["ind"].index(var)]*=-1
            
    add = len(vars)
    
    #less equalities
    for eq in less:
        if eq["val"] or len(eq["ind"]) > 1:
            list_eq = eq["ind"].copy()
            eq["ind"] = [var_start_index[i-1] for i in list_eq]
            list_index = [var_start_index[i-1] for i in list_eq]
            for var in free_vars:
                if var in list_eq:
                    eq["ind"].insert(list_index[var-1]+1, list_index[var-1]+1)
                    eq["coefs"].insert(list_index[var-1]+1, -eq["coefs"][list_index[var-1]])
            if eq["val"] < 0:
                eq["coefs"]*=-1
                eq["val"]*=-1
            for var in leq_vars:
                if var in eq["ind"]:
                    eq["coefs"][eq["ind"].index(var)]*=-1
            eq["coefs"] = np.append(eq["coefs"], 1)
            eq["ind"].append(add)
            add+=1

    #greater equaltities
    for eq in greater:
        if eq["val"] or len(eq["ind"]) > 1:
            list_eq = eq["ind"].copy()
            eq["ind"] = [var_start_index[i-1] for i in list_eq]
            list_index = [var_start_index[i-1] for i in list_eq]
            for var in free_vars:
                if var in list_eq:
                    eq["ind"].insert(list_index[var-1]+1, list_index[var-1]+1)
                    eq["coefs"].insert(list_index[var-1]+1, -eq["coefs"][list_index[var-1]])
            eq["coefs"] = np.array(eq["coefs"])
            if eq["val"] < 0:
                eq["coefs"]*=-1
                eq["val"]*=-1
            for var in leq_vars:
                if var in eq["ind"]:
                    eq["coefs"][eq["ind"].index(var)]*=-1
            eq["coefs"] = np.append(eq["coefs"], -1)
            eq["ind"].append(add)
            add+=1   
    #return to initial coordinates

    #setting the canonic matrix
    A = np.zeros((0, add))
    b = np.array([])
    for eq in equal:
        temp_eq = np.zeros(add)
        temp_eq[eq["ind"]] = eq["coefs"]
        A = np.append(A, np.array([temp_eq]), 0)
        b = np.append(b, eq["val"])
    for eq in less:
        temp_eq = np.zeros(add)
        temp_eq[eq["ind"]] = eq["coefs"]
        A = np.append(A, np.array([temp_eq]), 0)
        b = np.append(b, eq["val"])
    for eq in greater:
        temp_eq = np.zeros(add)
        temp_eq[eq["ind"]] = eq["coefs"]
        A = np.append(A, np.array([temp_eq]), 0)
        b = np.append(b, eq["val"])
    
    c = np.zeros(add)
    c[range(len(vars))] = initial_pr["func"]
    return A, b, c, add - len(func)


def simplex_init(A:np.array, b:np.array, c:np.array, add:int):
    '''
    Initiates simplex method written in the form:
    <c, x> -> min
    Ax = b; x >= 0; b>= 0;
    Returns:
        first_table: np.array for the initial table
        bases: names for the bases
    '''
    first_table = A.copy()
    #first_table = np.append(first_table, np.flip(np.eye(1, A.shape[0]), 1), 1)
    first_table = np.insert(first_table, 0, np.zeros(A.shape[0]), 1)

    first_row = c.copy()
    first_row = np.insert(first_row, 0, 1, 0)

    first_table = np.insert(first_table, 0, first_row, 0)

    last_column = b.copy()
    last_column = np.insert(last_column, 0, 0, 0)

    first_table = np.insert(first_table, first_table.shape[1], last_column, 1)
    bases = ["z"]
    bases.extend([f"x_{i+1}" for i in range(A.shape[1] - add)])
    bases.extend([f"s_{i+1}" for i in range(add)])
    bases.append("sol")

    #determing the first base
    first_base = []
    index_base = []
    for i in range(A.shape[1]):
        if sum(A[:,i])==1 and set(A[:, i]) == set([1, 0]):
            first_base.append(i)
            index_base.append(np.where(A[:,i] == 1)[0])
            if (len(first_base) > A.shape[0]):
                break
    
    if len(first_base) < A.shape[0]:
        raise ArithmeticError(f"You have only {len(first_base)} bases only! This problem should be transformed first!")
    base_elements = c[first_base]
    print(index_base)
    for i, base in enumerate(base_elements):
        first_table[0, ] = first_table[0, ] - base*first_table[index_base[i]+1, ]

    return first_table, np.array(bases)

def simplex_step(table:np.array):
    '''
    Makes a step for the simplex method
    Params:
        table: the table for simplex method described in Takha
    Returns:
        table_new: modified table for the simplex method
    '''
    
    min_bases = np.argmin(table[0, :-1])
    pivot_column = table[:, min_bases]
    last_column = table[:, -1]

    column_check = np.divide(last_column[1:], pivot_column[1:])
    column_check[column_check <= 0] = np.inf
    pivot_row = np.argmin(column_check)+1
    pivot_elem = table[pivot_row, min_bases]
    table_new = table.copy()
    table_new[pivot_row, :]/=pivot_elem

    for i in range(table.shape[0]):
        if i == pivot_row:
            continue
        table_new[i, :] = table_new[i, :] - table_new[pivot_row, :]*table_new[i, min_bases]

    return table_new

def simplex_method(table:np.array, bases):

    '''
    Solves the problem using simplex method
    Params:
        table: the first table for simplex method described in Takha
    Returns:
        table_new: the final table for the simplex method
        res: solution for the maximization problem
    '''
    table_new = simplex_step(table)
    k = 1
    print("Iteration: 1")
    print(pd.DataFrame(table_new, columns = bases))
    while(sum(sum(np.where(table_new[0, :-1]<0))) and k<=table_new.shape[0]):
        table_new = simplex_step(table_new)
        print(f"Iteration: {k+1}")
        print(pd.DataFrame(table_new, columns = bases))
        k+=1
    else:
        if k>table_new.shape[0]:
            raise ArithmeticError("The problem has no optimal solution!")

    res = {"z": table_new[0, -1]}
    ind = [base[0]=="x" for base in bases]
    x = bases[ind]
    for i in range(len(x)):
        if sum(table_new[:,i+1])==1 and set(table_new[:, i+1]) == set([1, 0]):
            res[x[i]] = np.dot(table_new[:, i+1], table_new[:, -1] )
        else:
            res[x[i]] = 0
    return table_new, res
