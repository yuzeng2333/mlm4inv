import sympy as sp
import random
import csv

# Function to parse expressions and find variables
def parse_expression(expr):
    parsed_expr = sp.sympify(expr)
    variables = sorted(parsed_expr.free_symbols, key=lambda symbol: symbol.name)
    return parsed_expr, variables

# Function to sample inputs and evaluate functions
def sample_and_evaluate(expressions, num_samples):
    parsed_expressions = [parse_expression(expr) for expr in expressions]
    variables = sorted({var for expr, vars in parsed_expressions for var in vars}, key=lambda symbol: symbol.name)
    
    data = []
    for _ in range(num_samples):
        # Sample values for each variable
        variable_values = {var: random.uniform(-10, 10) for var in variables}
         
        # Evaluate the functions
        results = [expr.evalf(subs=variable_values) for expr, vars in parsed_expressions]
        
        # Append the results to the data list
        data.append(list(variable_values.values()) + results)
    return variables, data

# Input from the user
#f_expression = input("Enter the expression for f: ")
#g_expression = input("Enter the expression for g: ")
f_expression = "x * y - x - y"
g_expression = "x + 2*z"

# Number of samples
#num_samples = int(input("Enter the number of samples: "))
num_samples = 16*1024*4

# Get the data
variables, data = sample_and_evaluate([f_expression, g_expression], num_samples)

# Write the data to a CSV file
with open('function_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([str(var) for var in variables] + [f'f({",".join(str(var) for var in variables)})', f'g({",".join(str(var) for var in variables)})'])  # write the header
    writer.writerows(data)

print("Data has been written to function_data.csv")

