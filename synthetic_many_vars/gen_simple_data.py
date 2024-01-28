import csv
import random

# Number of data rows to generate
num_rows = 1024 * 16 * 16

# File name
filename = 'simple.csv'

# Header for CSV file
header = ['a', 'b', 'w0', 'x', 'y', 'w1']

# Function to calculate w0 and w1 based on provided a, b, x, and y
def calculate_w_values(a, b, x, y):
    w0 = a + b
    w1 = x - y
    return w0, w1

# Generating and writing data to the CSV file
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    # Writing the header
    writer.writerow(header)
    
    # Generating and writing data rows
    for _ in range(num_rows):
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        x = random.randint(1, 100)
        y = random.randint(1, 100)
        w0, w1 = calculate_w_values(a, b, x, y)
        writer.writerow([a, b, w0, x, y, w1])

print(f"CSV file '{filename}' with {num_rows} rows has been created.")
