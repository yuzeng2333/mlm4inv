import random
import sympy
import os
import argparse
import json
from sympy.parsing.sympy_parser import parse_expr
from collections import Counter

# this file is used to generate a dataset for invariance training

RUN_REAL = True 
PRINT_SEPARATELY = True
EXPERIMENT_TO_RUN = 1

MAX_DEGREE = 2
MAX_VAR_NUM = 10
MAX_TERM_NUM = 5
MAX_EXPR_NUM = 2
if RUN_REAL:
    SOL_NUM = 16*1024*4
else:
    SOL_NUM = 4
CONST_MAX = 512
X_MAX = 32 
if RUN_REAL:
    MIN_SOL_NUM = SOL_NUM / 4
else:
    MIN_SOL_NUM = 3
ENABLE_INEQ = False
PRINT_SOL = False
WRITE_LABEL_FOR_POLY = False # if false, write labels for dependent variables

assert SOL_NUM >= MIN_SOL_NUM

def create_variable_dict(N):
    """
    Creates a dictionary where keys are integers from 0 to N-1,
    and values are strings 'x' concatenated with the key.
    
    Parameters:
    N (int): The number of key-value pairs in the dictionary.
    
    Returns:
    dict: A dictionary with keys from 0 to N-1 and values as 'x' followed by the key.
    """
    # Use dictionary comprehension to create the dictionary
    return {i: f'x{i}' for i in range(N)}

class SingleItem:
    # degress list have a fixed length of var_num
    def __init__(self, coeff, degree_list) -> None:
        self.coeff = coeff
        self.degree_list = degree_list
    

class SingleExpr:
    # is_eq is a boolean value
    # const is constant number on the right hand side
    def __init__(self, is_eq, item_list, const) -> None:
        self.is_eq = is_eq
        self.item_list = item_list
        self.const = const
        self.str = ""
        self.used_x_indices = set()

    def gen_str(self, var_dict):
        if self.str != "":
            return
        for item in self.item_list:
            self.str += str(item.coeff) + "*"
            idx = 0
            # check if degree_list has non-zero value
            all_zero = True
            for degree in item.degree_list:
                if degree != 0:
                    all_zero = False
                    break
            if all_zero == True:
                self.str += "1"
            else:
                for degree in item.degree_list:
                    # print the expression with variable name
                    if degree != 0:
                        var = var_dict[idx]
                        self.used_x_indices.add(idx)
                        self.str += "(" + var
                        if degree != 1:
                            self.str += "**"
                            self.str += str(degree)
                        self.str += ")*"  
                    idx += 1
            # remove the last "*" if it is "*"
            if self.str[-1] == "*":
                self.str = self.str[:-1]
            self.str += " + "
        # remove the last " + "
        if self.str[-3:] == " + ":
            self.str = self.str[:-3]

    def print_expr(self):
        if self.is_eq == 1:
            print(self.str + " - " + str(self.const) + " = w?")
        else:
            print(self.str + " < " + str(self.const))

    def get_max_degrees(self):
        max_degrees = [0 for _ in range(MAX_VAR_NUM)]
        for item in self.item_list:
            if len(item.degree_list) != len(max_degrees):
                print("Error: degree_list length not equal to max_degrees length")
                assert len(item.degree_list) == len(max_degrees)
            max_degrees = [max(x, y) for x, y in zip(max_degrees, item.degree_list)]
        return max_degrees


def check_imaginary(sol_list):
    for sol in sol_list:
        for key in sol.keys():
            if sol[key].is_real == False:
                return True
    return False


def get_expr_list():
    expr_list = []
    var_dict = create_variable_dict(MAX_VAR_NUM)
    # var_set is a set of indices of variables
    var_set = set()
    expr_num = random.randint(1, MAX_EXPR_NUM)
    for i in range(expr_num):
        # generate a single expression
        if ENABLE_INEQ == True:
            is_eq = random.randint(0, 1)
        else:
            is_eq = 1
        term_num = random.randint(1, MAX_TERM_NUM)
        item_list = []
        for j in range(term_num):
            # generate a single item
            # determine the degree of the item
            degree = random.randint(1, MAX_DEGREE)
            coeff = random.randint(1, 10)
            degree_list = []
            # determine the variables in this item and their degrees
            for k in range(MAX_VAR_NUM):
                var_not_included = random.randint(0, 1)
                if var_not_included == 1:
                    degree_list.append(0)
                else:
                    single_degree = random.randint(0, degree)
                    degree_list.append(single_degree)
                    degree = degree - single_degree
            # if the degree_list are all 0, set a random variable to 1
            if sum(degree_list) == 0:
                idx = random.randint(0, MAX_VAR_NUM - 1)
                degree_list[idx] = 1
            # add indices whose degree is non-zero to the var_set
            for idx, degree in enumerate(degree_list):
                if degree != 0:
                    var_set.add(idx)
            item_list.append(SingleItem(coeff, degree_list))
        # with 50% chance, the const is 0
        const = 0
        if random.randint(0, 1) == 1:
            const = random.randint(0, CONST_MAX)
        expr_list.append(SingleExpr(is_eq, item_list, const))

    # print the expression
    for expr in expr_list:
        expr.gen_str(var_dict)
        expr.print_expr()
    # get the used_var set of each expression
    all_used_x_indices = []
    for expr in expr_list:
        all_used_x_indices = all_used_x_indices + [expr.used_x_indices]
    return expr_list, var_set, all_used_x_indices, var_dict


# return the poly label for the degree
def get_poly_label(degree):
    if degree == 0:
        return "1"
    elif degree == 1:
        return "x"
    elif degree == 2:
        return "x2"
    else:
        # raise an exception
        raise Exception("degree is not supported")


def save_poly_labels(label_dir_name, file_name, expr_list, MAX_VAR_NUM, get_poly_label):
    """
    Saves polynomial labels and other related information to a JSON file.

    Parameters:
    label_dir_name (str): The directory path where the file will be saved.
    file_name (str): The name of the file to save.
    expr_list (list): A list of expressions to process and save.
    MAX_VAR_NUM (int): The maximum number of variables to consider for max degrees.
    get_poly_label (function): A function that returns polynomial labels based on degree sum.
    """
    with open(label_dir_name + file_name + ".json", "w") as f:
        num_expr = len(expr_list)
        eq_str = ", ".join(["\"eq\""] * num_expr)
        f.write("{\n")
        f.write("  \"eq\": [" + eq_str + "],\n")
        f.write("  \"op\": [\n")
        for expr_idx, expr in enumerate(expr_list):
            poly = {"x"}  # Initialize set with 'x'
            for item in expr.item_list:
                degree_sum = sum(item.degree_list)
                poly_label = get_poly_label(degree_sum)
                poly.add(poly_label)
            poly_list = list(poly)
            poly_str = ", ".join(f"\"{label}\"" for label in poly_list)
            f.write(f"    [{poly_str}]")
            if expr_idx < len(expr_list) - 1:
                f.write(",\n")
            else:
                f.write("\n")
        f.write("  ],\n")
        max_degrees = [0 for _ in range(MAX_VAR_NUM)]
        for expr in expr_list:
            one_max_degrees = expr.get_max_degrees()
            for idx, degree in enumerate(one_max_degrees):
                max_degrees[idx] = max(max_degrees[idx], degree)
        max_degrees_str = ", ".join(str(degree) for degree in max_degrees + [1, 1])
        f.write(f"  \"max_degree\": [{max_degrees_str}]\n")
        f.write("}\n")
    f.close()


def save_used_var_set(label_dir_name, file_name, all_used_x_indices):
    with open(label_dir_name + file_name + ".json", "w") as f:
        f.write("{\n")
        f.write("  \"used_var\": [\n")
        w_idx = 0
        for idx, used_x_indices in enumerate(all_used_x_indices):
            used_x_indices_list = list(used_x_indices)
            used_var_str = ", ".join('x'+str(var) for var in used_x_indices_list)
            used_var_str += ", w" + str(w_idx)
            f.write(f"    [{used_var_str}]")
            if idx < len(all_used_x_indices) - 1:
                f.write(",\n")
            else:
                f.write("\n")
            w_idx += 1
        f.write("  ]\n")
        f.write("}\n")
    f.close()


def print_result_to_separate_file(expr_list, sol_list, data_point_idx, MAX_DIGIT_WIDTH, all_used_x_indices, is_val):
    equation_dir_name = "./val_equations/" if is_val else "./equations/"
    data_dir_name = "./val_data/" if is_val else "./data/"
    label_dir_name = "./val_label/" if is_val else "./label/"
    # check if the three directories exist
    if not os.path.exists(equation_dir_name):
        os.makedirs(equation_dir_name)
    if not os.path.exists(data_dir_name):
        os.makedirs(data_dir_name)
    if not os.path.exists(label_dir_name):
        os.makedirs(label_dir_name)
    # count how many files there are in the directory of equations
    path = "./val_data/" if is_val else "./data/"
    num_files = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
    if num_files >= data_point_idx:
        data_point_idx = num_files

    file_name = str(data_point_idx)
    if sol_list.__len__() >= MIN_SOL_NUM:
        # store the equations
        with open(equation_dir_name+file_name+".txt", "w") as f:
            for expr in expr_list:
                f.write(expr.str + "\n")
        f.close()
        # store the solutions
        with open(data_dir_name+file_name+".csv", "w") as f:
            # if the file is empty, write the variables from the sol in the first line
            for idx, key in enumerate(sol_list[0][0].keys()):
                num_spaces = MAX_DIGIT_WIDTH - len(str(key))
                line = " " * num_spaces + str(key)
                f.write(line)
                if idx != sol_list[0][0].keys().__len__() - 1:
                    f.write(",")
            f.write("\n")
            for sol in sol_list:
                for idx, key in enumerate(sol[0].keys()):
                    num_spaces = MAX_DIGIT_WIDTH - len(str(sol[0][key]))
                    line = " " * num_spaces + str(sol[0][key])
                    f.write(line)
                    if idx != sol[0].keys().__len__() - 1:
                        f.write(",")
                f.write("\n")
        f.close()        
        # store the poly lables to the file
        if WRITE_LABEL_FOR_POLY == True:
            save_poly_labels(label_dir_name, file_name, expr_list, MAX_VAR_NUM, get_poly_label)
        else:
            save_used_var_set(label_dir_name, file_name, all_used_x_indices)
        return data_point_idx + 1
    else:
        return -1


class Stats:
    def __init__(self):        
        self.stats_expr_num = []
        self.stats_item_num = []
        self.stats_var_num = []
        self.stats_max_degree = []
        self.stats_sol_num = []

    def analyze_expr(self, expr_list):
        self.stats_expr_num.append(expr_list.__len__())
        for expr in expr_list:
            self.stats_item_num.append(expr.item_list.__len__())
            max_degree = 0
            for item in expr.item_list:
                for degree in item.degree_list:
                    if degree > max_degree:
                        max_degree = degree
            self.stats_max_degree.append(max_degree)

    def analyze_var(self, var_set):
        self.stats_var_num.append(var_set.__len__())

    def analyze_sol(self, sol_list):
        self.stats_sol_num.append(sol_list.__len__())
    
    def print_expr_num_dist(self):
        count = Counter(self.stats_expr_num)
        for number, frequency in sorted(count.items()):
            print(f"{number}: {'*' * frequency}")

    def print_item_num_dist(self):
        count = Counter(self.stats_item_num)
        for number, frequency in sorted(count.items()):
            print(f"{number}: {'*' * frequency}")

    def print_var_num_dist(self):
        count = Counter(self.stats_var_num)
        for number, frequency in sorted(count.items()):
            print(f"{number}: {'*' * frequency}")

    def print_max_degree_dist(self):
        count = Counter(self.stats_max_degree)
        for number, frequency in sorted(count.items()):
            print(f"{number}: {'*' * frequency}")

    def print_sol_num_dist(self):
        count = Counter(self.stats_sol_num)
        for number, frequency in sorted(count.items()):
            print(f"{number}: {'*' * frequency}")

    def print_stats(self):
        print("expr num: ")
        self.print_expr_num_dist()
        print("item num: ")
        self.print_item_num_dist()
        print("var num: ")
        self.print_var_num_dist()
        print("max degree: ")
        self.print_max_degree_dist()
        print("sol num: ")
        self.print_sol_num_dist()


def main(args):
  # the program begins here
  # declare the variables
  x, y, z = sympy.symbols("x y z")
  
  # before run, check if the file exists: equations.txt solutions.txt poly_labels.txt
  # if only some of them exist, delete them and regenerate them
  all_exist = True
  if os.path.exists("poly_labels.txt") == False:
      all_exist = False
  if os.path.exists("solutions.txt") == True and all_exist == False:
      os.remove("solutions.txt")
      all_exist = False
  if os.path.exists("equations.txt") == True and all_exist == False:
      os.remove("equations.txt")
      all_exist = False
  
   # instead of solve the equation for solutions,
   # we do in this way:
      # 1. w is always on the rhs of the equaltion
      # 2. we assign random numbers to x, y, z
  
  data_point_num = 0
  data_point_idx = 10
  MAX_DIGIT_WIDTH = 8
  stats = Stats()
  # 16 is the number of data points (a set of equations and inequalities)
  #  we want to generate
  while data_point_num < EXPERIMENT_TO_RUN:
      print("data point number: " + str(data_point_num))
      data_point_num += 1
      expr_list, var_set, all_used_x_indices, var_dict = get_expr_list()
      stats.analyze_expr(expr_list)
      stats.analyze_var(var_set)
      expr_num = expr_list.__len__()
      # w_list stores the variables on the RHS of the equations
      w_list = []
      for i in range(expr_num):
          w_list.append(sympy.symbols('w{}'.format(i)))
      # find up to SOL_NUM solutions to the equations
      sol_list = []
      # solve the equations with sympy
      # try to get SOL_NUM solutions
      run_num = 0
      while sol_list.__len__() < SOL_NUM:
          run_num += 1
          #print("run number: " + str(run_num))
          if run_num > 2 * SOL_NUM:
              break
          equations = []
          max_xyz = 0
          # assign a random number to x is 0 is in var_set

          for i in var_dict:
            if i in var_set:
                # Assign a random number to the variable
                var_val = int(random.gauss(-1 * X_MAX / 3, X_MAX / 3))
                max_xyz = max(max_xyz, var_val)
                var = var_dict[i]
                # Construct the equation
                var_eq = f"{var} + {var_val}"
                var_eq_expr = parse_expr(var_eq)
                equations.append(sympy.Eq(var_eq_expr, 0))
          

          # add all the equations to the list
          #assert expr_list.__len__() == expr_num
          for idx, expr in enumerate(expr_list):
              expr_str = expr.str
              if expr.is_eq == 1:
                  expr_str += " - " + str(expr.const)
                  expr_str_expr = parse_expr(expr_str)
                  equations.append(sympy.Eq(expr_str_expr, w_list[idx]))
          # print all the equations
          #for eq in equations:
          #    print(eq)
          # solve the equations
          sol = sympy.solve(equations, dict=True)
          # skip the sol if it is empty
          if sol.__len__() == 0:
              print("empty solution")
              break
          # if the solution has imaginary number, skip it
          if check_imaginary(sol) == True:
              print ("imaginary number")
              continue
          # if the value of any w in w_list is one order of magnitude larger than x, y, z,
          # skip the solution
          #all_w = True
          #for key in sol[0].keys():
          #    if key[0] == "w":
          #        if sol[0][key] > max_xyz*100:
          #            all_w = False
          #            break
          #if all_w == False:
          #    print("w is too large")
          #    continue
          # check if the solutions satisfy all the inequalities
          all_pass = True
          for expr in expr_list:
              if expr.is_eq == 0:
                  expr_str = expr.str
                  expr_str += " < " + str(expr.const)
                  # evaluate the expression
                  expr_val = eval(expr_str)
                  if expr_val == False:
                      all_pass = False
                      break
          # print the sol
          if all_pass == True:
              # append the solution to the list
              sol_num = sol_list.__len__()
              if PRINT_SOL == True:
                  print(sol)
              sol_list.append(sol)
              print("sol number: " + str(sol_num))
  
      stats.analyze_sol(sol_list)
      data_point_idx = print_result_to_separate_file(expr_list, sol_list, data_point_idx, MAX_DIGIT_WIDTH, all_used_x_indices, is_val=args.val)
  
  stats.print_stats()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Process some arguments.')
  parser.add_argument('--val', action='store_true', help='Generate validation data')

  args = parser.parse_args()
  main(args)
