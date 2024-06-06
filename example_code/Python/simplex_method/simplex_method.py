import heapq
# Calculate the dot product of two vectors
def dot(a,b):
   return sum(x*y for x,y in zip(a,b))

# Get a specific column from a 2D list (matrix)
def column(A, j):
   return [row[j] for row in A]

# Transpose a 2D list (matrix)
def transpose(A):
   return [column(A, j) for j in range(len(A[0]))]

# Check if a column is a pivot column
def isPivotCol(col):
   return (len([c for c in col if c == 0]) == len(col) - 1) and sum(col) == 1

# Find the value of a variable for a pivot column
def variableValueForPivotColumn(tableau, column):
   pivotRow = [i for (i, x) in enumerate(column) if x == 1][0]
   return tableau[pivotRow][-1]

# Check if we can improve the current solution
def canImprove(tableau):
   lastRow = tableau[-1]
   return any(x > 0 for x in lastRow[:-1])

# Check if there's more than one minimum in a list
def moreThanOneMin(L):
   if len(L) <= 1:
      return False
   x,y = heapq.nsmallest(2, L, key=lambda x: x[1])
   return x == y

# Create an identity matrix with certain specifications
def identity(numRows, numCols, val=1, rowStart=0):
   return [[(val if i == j else 0) for j in range(numCols)] for i in range(rowStart, numRows)] 

def standardForm(cost, greaterThans=[], gtThreshold=[], lessThans=[], ltThreshold=[], equalities=[], eqThreshold=[], maximization=True):
   # Initialize variables for the count of new variables and the number of rows in the tableau
   newVars = 0
   numRows = 0

   # Adjust the number of variables and rows based on the constraints
   # Add slack variables for 'greater than' inequalities
   if gtThreshold != []:
      newVars += len(gtThreshold)
      numRows += len(gtThreshold)

   if ltThreshold != []:
      newVars += len(ltThreshold)
      numRows += len(ltThreshold)
   
   # Equalities don't need slack variables but add to the number of rows
   if eqThreshold != []:
      numRows += len(eqThreshold)

   # If the problem is a minimization, convert it to a maximization by negating the cost vector   
   if not maximization:
      cost = [-x for x in cost]

   # If no new variables are needed, the problem is already in standard form
   if newVars == 0:
      return cost, equalities, eqThreshold

   # Extend the cost function with zeros for the new slack variables
   newCost = list(cost) + [0] * newVars
   constraints = []
   threshold = []

   # Prepare the constraints for each condition ('greater than', 'less than', 'equal to')
   oldConstraints = [(greaterThans, gtThreshold, -1), (lessThans, ltThreshold, 1), (equalities, eqThreshold, 0)]
   offset = 0

   # Process each set of constraints
   for constraintList, oldThreshold, coefficient in oldConstraints:
      # Append the identity matrix multiplied by the coefficient for slack variables
      constraints += [c + r for c, r in zip(constraintList, identity(numRows, newVars, coefficient, offset))]
      # Append the thresholds for each constraint
      threshold += oldThreshold
      # Increase the offset for the identity matrix used for the next set of constraints
      offset += len(oldThreshold)

   return newCost, constraints, threshold

'''
   simplex: [float], [[float]], [float] -> [float], float
   Solve the given standard-form linear program:
      max <c,x>
      s.t. Ax = b
           x >= 0
   providing the optimal solution x* and the value of the objective function
'''
def simplex(c, A, b):
   # assume the last m columns of A are the slack variables; the initial basis is the set of slack variables
   tableau = [row[:] + [x] for row, x in zip(A, b)]
   tableau.append([ci for ci in c] + [0])
   print("Initial tableau:")
   for row in tableau:
      print(row)
   print()

   # Iterate until no improvements can be made
   while canImprove(tableau):
      # Choose entering variable (minimum positive index of the last row)
      column_choices = [(i,x) for (i,x) in enumerate(tableau[-1][:-1]) if x > 0]
      column = min(column_choices, key=lambda a: a[1])[0]

      # Check for unboundedness
      if all(row[column] <= 0 for row in tableau):
         raise Exception('Linear program is unbounded.')

      # Check for degeneracy: more than one minimizer of the quotient
      quotients = [(i, r[-1] / r[column]) for i,r in enumerate(tableau[:-1]) if r[column] > 0]

      if moreThanOneMin(quotients):
         raise Exception('Linear program is degenerate.')

      # Chosing leaving variable (row index minimizing the quotient)
      row = min(quotients, key=lambda x: x[1])[0]

      #pivots on the chosen row and column
      pivot = row, column

      print("Next pivot index is=%d,%d \n" % pivot)
      i,j = pivot
      pivotDenom = tableau[i][j]

      # Normalize the pivot row
      tableau[i] = [x / pivotDenom for x in tableau[i]]

      # Zero out the other entries in the pivot column
      for k,row in enumerate(tableau):
         if k != i:
            pivotRowMultiple = [y * tableau[k][j] for y in tableau[i]]
            tableau[k] = [x - y for x,y in zip(tableau[k], pivotRowMultiple)]
      print("Tableau after pivot:")
      for row in tableau:
         print(row)
      print()
   
   # Transpose the tableau to make it easier to work with columns
   columns = transpose(tableau)

   # Identify pivot columns in the tableau. A column is a pivot column if it has a single 1 and the rest of its entries are 0.
   indices = [j for j, col in enumerate(columns[:-1]) if isPivotCol(col)]

   # Looking at the rightmost entry (the value part of the tableau row) of the row where the 1 in the pivot column is located.
   primal_solution = [(colIndex, variableValueForPivotColumn(tableau, columns[colIndex])) for colIndex in indices]
   
   # The last entry of the last row of the tableau gives us the negation of the objective function value.
   objective_value = -(tableau[-1][-1])

   return tableau, primal_solution, objective_value