from simplex import *

func = "-1*x_1+1*x_2"
constr_1 = "-2*x_1+1*x_2 <= 1"
constr_2 = "1*x_1-2*x_2 >= 4"
problem_str = convert_string_problem_into_arrays(func, [], [constr_1], [constr_2], "")
A, b, c, add = canonic_form_of_problem(problem_str)
print(A, c)
first_table, bases = simplex_init(A, b, c, add)
print("First table")
print(pd.DataFrame(first_table, columns = bases))
table_final, res = simplex_method(first_table, bases)
print("Results")
print(pd.DataFrame(table_final, columns = bases))
print(res)