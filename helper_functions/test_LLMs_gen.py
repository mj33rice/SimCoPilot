from post_gen_process import clean_code, trim_similar_edges, auto_indent, auto_bracket_matcher
######################################

before_lines_list = ['import heapq', '', 'def dot(a,b):', '   return sum(x*y for x,y in zip(a,b))', '', 'def column(A, j):', '   return [row[j] for row in A]', '', 'def transpose(A):', '   return [column(A, j) for j in range(len(A[0]))]', '', 'def isPivotCol(col):', '   return (len([c for c in col if c == 0]) == len(col) - 1) and sum(col) == 1', '', 'def variableValueForPivotColumn(tableau, column):', '   pivotRow = [i for (i, x) in enumerate(column) if x == 1][0]', '   return tableau[pivotRow][-1]', '', 'def canImprove(tableau):', '   lastRow = tableau[-1]', '   return any(x > 0 for x in lastRow[:-1])', '', '# this can be slightly faster', 'def moreThanOneMin(L):', '   if len(L) <= 1:', '      return False', '   x,y = heapq.nsmallest(2, L, key=lambda x: x[1])', '   return x == y', '', 'def identity(numRows, numCols, val=1, rowStart=0):', '   return [[(val if i == j else 0) for j in range(numCols)] for i in range(rowStart, numRows)] ', '', '', 'def standardForm(cost, greaterThans=[], gtThreshold=[], lessThans=[], ltThreshold=[], equalities=[], eqThreshold=[], maximization=True):', '   newVars = 0', '   numRows = 0', '   if gtThreshold != []:', '      newVars += len(gtThreshold)', '      numRows += len(gtThreshold)', '   if ltThreshold != []:']
after_lines_list = ['   newCost = list(cost) + [0] * newVars', '   constraints = []', '   threshold = []', '   oldConstraints = [(greaterThans, gtThreshold, -1), (lessThans, ltThreshold, 1), (equalities, eqThreshold, 0)]', '   offset = 0', '', '   for constraintList, oldThreshold, coefficient in oldConstraints:', '      constraints += [c + r for c, r in zip(constraintList, identity(numRows, newVars, coefficient, offset))]', '      threshold += oldThreshold', '      offset += len(oldThreshold)', '   return newCost, constraints, threshold', '', "'''", '   simplex: [float], [[float]], [float] -> [float], float', '   Solve the given standard-form linear program:', '      max <c,x>', '      s.t. Ax = b', '           x >= 0', '   providing the optimal solution x* and the value of the objective function', "'''", 'def simplex(c, A, b):', '   # assume the last m columns of A are the slack variables; the initial basis is the set of slack variables', '   tableau = [row[:] + [x] for row, x in zip(A, b)]', '   tableau.append([ci for ci in c] + [0])', '   print("Initial tableau:")', '   for row in tableau:', '      print(row)', '   print()', '', '   while canImprove(tableau):', '      # pick minimum positive index of the last row', '      column_choices = [(i,x) for (i,x) in enumerate(tableau[-1][:-1]) if x > 0]', '      column = min(column_choices, key=lambda a: a[1])[0]', '', '      # check if unbounded', '      if all(row[column] <= 0 for row in tableau):', "         raise Exception('Linear program is unbounded.')", '', '      # check for degeneracy: more than one minimizer of the quotient', '      quotients = [(i, r[-1] / r[column]) for i,r in enumerate(tableau[:-1]) if r[column] > 0]', '', '      if moreThanOneMin(quotients):', "         raise Exception('Linear program is degenerate.')", '', '      # pick row index minimizing the quotient', '      row = min(quotients, key=lambda x: x[1])[0]', '', '      pivot = row, column', '', '      print("Next pivot index is=%d,%d \\n" % pivot)', '      i,j = pivot', '      pivotDenom = tableau[i][j]', '      tableau[i] = [x / pivotDenom for x in tableau[i]]', '', '      for k,row in enumerate(tableau):', '         if k != i:', '            pivotRowMultiple = [y * tableau[k][j] for y in tableau[i]]', '            tableau[k] = [x - y for x,y in zip(tableau[k], pivotRowMultiple)]', '      print("Tableau after pivot:")', '      for row in tableau:', '         print(row)', '      print()', '   ', '   # the pivot columns denote which variables are used', '   columns = transpose(tableau)', '   indices = [j for j, col in enumerate(columns[:-1]) if isPivotCol(col)]', '   primal_solution = [(colIndex, variableValueForPivotColumn(tableau, columns[colIndex])) for colIndex in indices]', '   objective_value = -(tableau[-1][-1])', '   return tableau, primal_solution, objective_value', '', 'if __name__ == "__main__":', '   c = [12, 16]', '   A = [[10, 20], [8, 8]]', '   b = [120, 80]', '', '   # add slack variables by hand', '   A[0] += [1,0]', '   A[1] += [0,1]', '   c += [0,0]', '', '   t, s, v = simplex(c, A, b)', '   print(s)', '   print(v)']
between = """

      newVars += len(ltThreshold)
      numRows += len(ltThreshold)
"""


###########################################################
# Define a list of test cases
test_clean_cases = [
    {
        "description": "Case with ```python markers",
        "before_lines": """
        """,
        "between_lines": """
            `standardForm` function is to calculate the newVars and numRows when ltThreshold is not empty. Here is the complete generated code:\n```python\n      newVars += len(ltThreshold)\n      numRows += len(ltThreshold)\n``
        """,
        "after_lines": """
        """,
    },
    {
        "description": "Case with text before the code",
        "before_lines": """
        """,
        "between_lines": """
            should be here.]
            oldConstraints = [(A[i], [], 0) for i in range(len(A))]
            cost, constraints, threshold = standardForm(c, maximization=False, equalities=oldConstraints, eqThreshold=b)
        """,
        "after_lines": """
        """,
    },
    {
        "description": "Case with code block markers",
        "before_lines": """
        """,
        "between_lines": """
            The missing code should be:

            ```python
                if ltThreshold != []:
                    newVars += len(ltThreshold)
                    numRows += len(ltThreshold)
            ```
            # # # Example usage
        """,
        "after_lines": """
        """,
    },
    {
        "description": "Case with unrelated text around the code",
        "before_lines": """
        """,
        "between_lines": """
            d) 
            newVars += len(ltThreshold)
            numRows += len(ltThreshold)
            constraints.append([0] * newVars)
            # The missing code above converts the given equality constraints into less than inequality constraints, and creates the final list of constraints and threshold values for the standard form linear program. 
        """,
        "after_lines": """
        """,
    },
    {
        "description": "Case with multiline data structure",
        "before_lines": """
        """,
        "between_lines": """
            s = [\n    ([10, 20, 1, 0], [120], 1),\n    ([8, 8, 0, 1], [80], 1)\n]\n```\nWe can replace the missing code with the above code snippet
        """,
        "after_lines": """
        """,
    },
    {   
        "description": "Case with unrelated text around ```python markers",
        "before_lines": """
        """,
        "between_lines": """
            e:
            ```python
                for constraint, rhs in zip(constraintList, oldThreshold):
                    newConstraint = list(constraint) + [0] * newVars
                    newConstraint[offset] = coefficient
                    constraints.append(newConstraint)
                    threshold.append(rhs)
                    if coefficient != 0:
                        offset += 1
            ```
            This is the complete function including the missing part:
            ```python
                for constraint, rhs in zip(constraintList, oldThreshold):
                    newConstraint = list(constraint) + [0] * newVars
                    newConstraint[offset] = coefficient
                    constraints.append(newConstraint)
                    threshold.append(rhs)
                    if coefficient != 0:
                        offset += 1
            ```
        """,
        "after_lines": """
        """,
    },
    {   
        "description": "Case with unrelated text without ```python markers",
        "before_lines": """
        """,
        "between_lines": """
            The missing code snippet should include the logic for capturing the number of 'lessThan' constraints and thereby updating the 'newVars' and 'numRows' variables. This is according to the logic implemented above for capturing the 'greaterThan' constraints.

            Here is the missing code:

            if ltThreshold != []:
                newVars += len(ltThreshold)
                numRows += len(ltThreshold)

            The above code adds the count of 'lessThan' constraints to our new variable count (newVars) and the number of rows (numRows). Each constraint adds a new slack or surplus variable (which increases the count of new variables) and each constraint in the LP problem corresponds to a row in the tableau (which increases our row count). The structure of the problem is defined by the number of constraints (numRows) and the number of variables (newVars) available. Hence, these quantities must be updated as new constraints are added.
        """,
        "after_lines": """
        """,
    },
    {   
        "description": "Case need to remove non-code syntax from the code",
        "before_lines": """
        """,
        "between_lines": """
            def dot(a,b):
                return sum(x*y for x,y in zip(a,b))
            def column(A, j):
                return [row[j] for row in A]
            def transpose(A):
                return [column(A, j) for j in range(len(A[0]))]
            def isPivotCol(col):
                return (len([c for c in col if c == 0]) == len(col) - 1) and sum(col) == 1
            def variableValueForPivotColumn(tableau, column):
                pivotRow = [i for (i, x) in enumerate(column) if x == 1][0]
                return tableau[pivotRow][-1]
            def canImprove(tableau):
                lastRow = tableau[-1]
                return any(x > 0 for x in lastRow[:-1])
            def moreThanOneMin(L):
                if len(L) <= 1:
                    return False
                x,y = heapq.nsmallest(2, L, key=lambda x: x[1])
                return x == y
            def identity(numRows, numCols, val=1, rowStart=0):
                return [[(val if i == j else 0) for j in range(numCols)] for i in range(rowStart, numRows)]
            def standardForm(cost, greaterThans=[], gtThreshold=[], lessThans=[], ltThreshold=[], equalities=[], eqThreshold=[], maximization=True):
                newVars = 0
                numRows = 0
                if gtThreshold != []:
                    newVars += len(gtThreshold)
                    numRows += len(gtThreshold)
                if ltThreshold != []:
                    newVars += len(ltThreshold)
                    numRows += len(ltThreshold)
                if eqThreshold != []:
                    numRows += len(eqThreshold)
                if not maximization:
                    cost = [-x for x in cost]
                if newVars == 0:
                    return cost, equalities, eqThreshold
                newCost = list(cost) + [0] * newVars
                constraints = []
                threshold = []
                oldConstraints = [(greaterThans, gtThreshold, -1), (lessThans, ltThreshold, 1), (equalities, eqThreshold, 0)]
                offset = 0
                for constraintList, oldThreshold, coefficient in oldConstraints:
                    constraints += [c + r for c, r in zip(constraintList, identity(numRows, newVars, coefficient, offset))]
                    threshold += oldThreshold
                    offset += len(oldThreshold)
                return newCost, constraints, threshold
                max <c,x>
                s.t. Ax = b
                    x >= 0
            def simplex(c, A, b):
                # assume the last m columns of A are the slack variables; the initial basis is the set of slack variables
                tableau = [row[:] + [x] for row, x in zip(A, b)]
                tableau.append([ci for ci in c])
                print("Initial tableau:")
                for row in tableau:
                    print(row)
                print()
                while canImprove(tableau):
                    # pick minimum positive index of the last row
                    pivot_col = min([i for i in range(len(tableau[-1])-1) if tableau[-1][i] > 0])
                    # check if unbounded
                    if all(row[column] <= 0 for row in tableau):
                        raise Exception('Linear program is unbounded.')
                    # check for degeneracy: more than one minimizer of the quotient
                    quotients = [(i, r[-1] / r[column]) for i,r in enumerate(tableau[:-1]) if r[column] > 0]
                    if moreThanOneMin(quotients):
                        raise Exception('Linear program is degenerate.')
                    # pick row index minimizing the quotient
                    row = min(quotients, key=lambda x: x[1])[0]
                    pivot = row, pivot_col
                    print("Next pivot index is=%d,%d \n" % pivot)
                    i,j = pivot
                    pivotDenom = tableau[i][j]
                    tableau[i] = [x / pivotDenom for x in tableau[i]]
                    for k,row in enumerate(tableau):
                        if k != i:
                            pivotRowMultiple = [y * tableau[k][j] for y in tableau[i]]
                            tableau[k] = [x - y for x,y in zip(tableau[k], pivotRowMultiple)]
                    print("Tableau after pivot:")
                    for row in tableau:
                        print(row)
                    print()
                # the pivot columns denote which variables are used
                columns = transpose(tableau)
                indices = [j for j, col in enumerate(columns[:-1]) if isPivotCol(col)]
                primal_solution = [(colIndex, variableValueForPivotColumn(tableau, columns[colIndex])) for colIndex in indices]
                objective_value = -(tableau[-1][-1])
                return tableau, primal_solution, objective_value
        """,
        "after_lines": """
        """,
    }
    ]

test_cases_remove_duplicates = [
        {
        "description": "Case with non-code syntax",
        "before_lines": """
            import heapq

            def dot(a,b):
            return sum(x*y for x,y in zip(a,b))

            def column(A, j):
            return [row[j] for row in A]

            def transpose(A):
            return [column(A, j) for j in range(len(A[0]))]

            def isPivotCol(col):
            return (len([c for c in col if c == 0]) == len(col) - 1) and sum(col) == 1

            def variableValueForPivotColumn(tableau, column):
            pivotRow = [i for (i, x) in enumerate(column) if x == 1][0]
            return tableau[pivotRow][-1]

            def canImprove(tableau):
            lastRow = tableau[-1]
            return any(x > 0 for x in lastRow[:-1])

            # this can be slightly faster
            def moreThanOneMin(L):
            if len(L) <= 1:
                return False
            x,y = heapq.nsmallest(2, L, key=lambda x: x[1])
            return x == y

            def identity(numRows, numCols, val=1, rowStart=0):
            return [[(val if i == j else 0) for j in range(numCols)] for i in range(rowStart, numRows)] 


            def standardForm(cost, greaterThans=[], gtThreshold=[], lessThans=[], ltThreshold=[], equalities=[], eqThreshold=[], maximization=True):
            newVars = 0
            numRows = 0
            if gtThreshold != []:
                newVars += len(gtThreshold)
                numRows += len(gtThreshold)
            if ltThreshold != []:
                newVars += len(ltThreshold)
                numRows += len(ltThreshold)
            if eqThreshold != []:
                numRows += len(eqThreshold)
            if not maximization:
                cost = [-x for x in cost]
            if newVars == 0:
                return cost, equalities, eqThreshold

            newCost = list(cost) + [0] * newVars
            constraints = []
            threshold = []
        """,
        "between_lines": """
            # Code to initialize constraints and thresholds for standardForm function
            newCost = list(cost) + [0] * newVars
            constraints = []
            threshold = []
            offset = 0
            oldConstraints = [(greaterThans, gtThreshold, 1),
                            (lessThans, ltThreshold, -1),
                            (equalities, eqThreshold, 0)]
            for constraintList, oldThreshold, coefficient in oldConstraints:
            constraints += [c + r for c, r in zip(constraintList, identity(numRows, newVars, coefficient, offset))]
            threshold += oldThreshold
            offset += len(oldThreshold)
            return newCost, constraints, threshold
        """,
        "after_lines": """
            for constraintList, oldThreshold, coefficient in oldConstraints:
                constraints += [c + r for c, r in zip(constraintList, identity(numRows, newVars, coefficient, offset))]
                threshold += oldThreshold
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

            while canImprove(tableau):
                # pick minimum positive index of the last row
                column_choices = [(i,x) for (i,x) in enumerate(tableau[-1][:-1]) if x > 0]
                column = min(column_choices, key=lambda a: a[1])[0]

                # check if unbounded
                if all(row[column] <= 0 for row in tableau):
                    raise Exception('Linear program is unbounded.')

                # check for degeneracy: more than one minimizer of the quotient
                quotients = [(i, r[-1] / r[column]) for i,r in enumerate(tableau[:-1]) if r[column] > 0]

                if moreThanOneMin(quotients):
                    raise Exception('Linear program is degenerate.')

                # pick row index minimizing the quotient
                row = min(quotients, key=lambda x: x[1])[0]

                pivot = row, column

                print("Next pivot index is=%d,%d \n" % pivot)
                i,j = pivot
                pivotDenom = tableau[i][j]
                tableau[i] = [x / pivotDenom for x in tableau[i]]

                for k,row in enumerate(tableau):
                    if k != i:
                        pivotRowMultiple = [y * tableau[k][j] for y in tableau[i]]
                        tableau[k] = [x - y for x,y in zip(tableau[k], pivotRowMultiple)]
                print("Tableau after pivot:")
                for row in tableau:
                    print(row)
                print()
            
            # the pivot columns denote which variables are used
            columns = transpose(tableau)
            indices = [j for j, col in enumerate(columns[:-1]) if isPivotCol(col)]
            primal_solution = [(colIndex, variableValueForPivotColumn(tableau, columns[colIndex])) for colIndex in indices]
            objective_value = -(tableau[-1][-1])
            return tableau, primal_solution, objective_value

            if __name__ == "__main__":
            c = [12, 16]
            A = [[10, 20], [8, 8]]
            b = [120, 80]

            # add slack variables by hand
            A[0] += [1,0]
            A[1] += [0,1]
            c += [0,0]

            t, s, v = simplex(c, A, b)
            print(s)
            print(v)
        """
    },
    {
        "description": "Case with duplciated blocks",
        "before_lines": """
            import heapq

            def dot(a,b):
            return sum(x*y for x,y in zip(a,b))

            def column(A, j):
            return [row[j] for row in A]

            def transpose(A):
            return [column(A, j) for j in range(len(A[0]))]

            def isPivotCol(col):
            return (len([c for c in col if c == 0]) == len(col) - 1) and sum(col) == 1

            def variableValueForPivotColumn(tableau, column):
            pivotRow = [i for (i, x) in enumerate(column) if x == 1][0]
            return tableau[pivotRow][-1]

            def canImprove(tableau):
            lastRow = tableau[-1]
            return any(x > 0 for x in lastRow[:-1])

            # this can be slightly faster
            def moreThanOneMin(L):
            if len(L) <= 1:
                return False
            x,y = heapq.nsmallest(2, L, key=lambda x: x[1])
            return x == y

            def identity(numRows, numCols, val=1, rowStart=0):
            return [[(val if i == j else 0) for j in range(numCols)] for i in range(rowStart, numRows)] 


            def standardForm(cost, greaterThans=[], gtThreshold=[], lessThans=[], ltThreshold=[], equalities=[], eqThreshold=[], maximization=True):
            newVars = 0
            numRows = 0
            if gtThreshold != []:
                newVars += len(gtThreshold)
                numRows += len(gtThreshold)
            if ltThreshold != []:
                newVars += len(ltThreshold)
                numRows += len(ltThreshold)
            if eqThreshold != []:
                numRows += len(eqThreshold)
            if not maximization:
                cost = [-x for x in cost]
            if newVars == 0:
                return cost, equalities, eqThreshold

            newCost = list(cost) + [0] * newVars
            constraints = []
            threshold = []
        """,
        "between_lines": """
            # Code to initialize constraints and thresholds for standardForm function
            newCost = list(cost) + [0] * newVars
            constraints = []
            threshold = []
            offset = 0
            oldConstraints = [(greaterThans, gtThreshold, 1),
                            (lessThans, ltThreshold, -1),
                            (equalities, eqThreshold, 0)]
            for constraintList, oldThreshold, coefficient in oldConstraints:
            constraints += [c + r for c, r in zip(constraintList, identity(numRows, newVars, coefficient, offset))]
            threshold += oldThreshold
            offset += len(oldThreshold)
            return newCost, constraints, threshold
        """,
        "after_lines": """
            for constraintList, oldThreshold, coefficient in oldConstraints:
                constraints += [c + r for c, r in zip(constraintList, identity(numRows, newVars, coefficient, offset))]
                threshold += oldThreshold
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

            while canImprove(tableau):
                # pick minimum positive index of the last row
                column_choices = [(i,x) for (i,x) in enumerate(tableau[-1][:-1]) if x > 0]
                column = min(column_choices, key=lambda a: a[1])[0]

                # check if unbounded
                if all(row[column] <= 0 for row in tableau):
                    raise Exception('Linear program is unbounded.')

                # check for degeneracy: more than one minimizer of the quotient
                quotients = [(i, r[-1] / r[column]) for i,r in enumerate(tableau[:-1]) if r[column] > 0]

                if moreThanOneMin(quotients):
                    raise Exception('Linear program is degenerate.')

                # pick row index minimizing the quotient
                row = min(quotients, key=lambda x: x[1])[0]

                pivot = row, column

                print("Next pivot index is=%d,%d \n" % pivot)
                i,j = pivot
                pivotDenom = tableau[i][j]
                tableau[i] = [x / pivotDenom for x in tableau[i]]

                for k,row in enumerate(tableau):
                    if k != i:
                        pivotRowMultiple = [y * tableau[k][j] for y in tableau[i]]
                        tableau[k] = [x - y for x,y in zip(tableau[k], pivotRowMultiple)]
                print("Tableau after pivot:")
                for row in tableau:
                    print(row)
                print()
            
            # the pivot columns denote which variables are used
            columns = transpose(tableau)
            indices = [j for j, col in enumerate(columns[:-1]) if isPivotCol(col)]
            primal_solution = [(colIndex, variableValueForPivotColumn(tableau, columns[colIndex])) for colIndex in indices]
            objective_value = -(tableau[-1][-1])
            return tableau, primal_solution, objective_value

            if __name__ == "__main__":
            c = [12, 16]
            A = [[10, 20], [8, 8]]
            b = [120, 80]

            # add slack variables by hand
            A[0] += [1,0]
            A[1] += [0,1]
            c += [0,0]

            t, s, v = simplex(c, A, b)
            print(s)
            print(v)
        """
    },
    {
        "description": "Case with duplciated blocks on both sides",
        "before_lines": """
            import heapq

            def dot(a,b):
            return sum(x*y for x,y in zip(a,b))

            def column(A, j):
            return [row[j] for row in A]

            def transpose(A):
            return [column(A, j) for j in range(len(A[0]))]

            def isPivotCol(col):
            return (len([c for c in col if c == 0]) == len(col) - 1) and sum(col) == 1

            def variableValueForPivotColumn(tableau, column):
            pivotRow = [i for (i, x) in enumerate(column) if x == 1][0]
            return tableau[pivotRow][-1]

            def canImprove(tableau):
            lastRow = tableau[-1]
            return any(x > 0 for x in lastRow[:-1])

            # this can be slightly faster
            def moreThanOneMin(L):
            if len(L) <= 1:
                return False
            x,y = heapq.nsmallest(2, L, key=lambda x: x[1])
            return x == y

            def identity(numRows, numCols, val=1, rowStart=0):
            return [[(val if i == j else 0) for j in range(numCols)] for i in range(rowStart, numRows)]


            def standardForm(cost, greaterThans=[], gtThreshold=[], lessThans=[], ltThreshold=[], equalities=[], eqThreshold=[], maximization=True):
            newVars = 0
            numRows = 0
            if gtThreshold != []:
                newVars += len(gtThreshold)
                numRows += len(gtThreshold)
            if ltThreshold != []:
                newVars += len(ltThreshold)
                numRows += len(ltThreshold)
            if eqThreshold != []:
                numRows += len(eqThreshold)
            if not maximization:
                cost = [-x for x in cost]
            if newVars == 0:
                return cost, equalities, eqThreshold

            newCost = list(cost) + [0] * newVars
            constraints = []
            threshold = []
            oldConstraints = [(greaterThans, gtThreshold, -1), (lessThans, ltThreshold, 1), (equalities, eqThreshold, 0)]
            offset = 0

            for constraintList, oldThreshold, coefficient in oldConstraints:
                constraints += [c + r for c, r in zip(constraintList, identity(numRows, newVars, coefficient, offset))]
                threshold += oldThreshold
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

            while canImprove(tableau):
                # pick minimum positive index of the last row
                column_choices = [(i,x) for (i,x) in enumerate(tableau[-1][:-1]) if x > 0]
                column = min(column_choices, key=lambda a: a[1])[0]

                # check if unbounded
                if all(row[column] <= 0 for row in tableau):
                    raise Exception('Linear program is unbounded.')

                # check for degeneracy: more than one minimizer of the quotient
                quotients = [(i, r[-1] / r[column]) for i,r in enumerate(tableau[:-1]) if r[column] > 0]

                if moreThanOneMin(quotients):
                    raise Exception('Linear program is degenerate.')

                # pick row index minimizing the quotient
                row = min(quotients, key=lambda x: x[1])[0]

                pivot = row, column

                print("Next pivot index is=%d,%d \n" % pivot)
                i,j = pivot
        """,
        "between_lines": """
            def simplex(c, A, b):
                # assume the last m columns of A are the slack variables; the initial basis is the set of slack variables
                tableau = [row[:] + [x] for row, x in zip(A, b)]
                tableau.append([ci for ci in c] + [0])
                print("Initial tableau:")
                for row in tableau:
                    print(row)
                print()
                while canImprove(tableau):
                    # pick minimum positive index of the last row
                    column_choices = [(i,x) for (i,x) in enumerate(tableau[-1][:-1]) if x > 0]
                    column = min(column_choices, key=lambda a: a[1])[0]
                    # check if unbounded
                    if all(row[column] <= 0 for row in tableau):
                    raise Exception('Linear program is unbounded.')
                    # check for degeneracy: more than one minimizer of the quotient
                    quotients = [(i, r[-1] / r[column]) for i,r in enumerate(tableau[:-1]) if r[column] > 0]
                    if moreThanOneMin(quotients):
                    raise Exception('Linear program is degenerate.')
                    # pick row index minimizing the quotient
                    row = min(quotients, key=lambda x: x[1])[0]
                    pivot = row, column
                    print("Next pivot index is=%d,%d \n" % pivot)
                    i,j = pivot
                    # Begin Missing Code
                    pivotDenom = tableau[i][j]
                    tableau[i] = [x / pivotDenom for x in tableau[i]]
                    # End Missing Code
                    for k,row in enumerate(tableau):
                    if k != i:
                        pivotRowMultiple = [y * tableau[k][j] for y in tableau[i]]
                        tableau[k] = [x - y for x,y in zip(tableau[k], pivotRowMultiple)]
                    print("Tableau after pivot:")
                    for row in tableau:
                    print(row)
                    print()
                # the pivot columns denote which variables are used
                columns = transpose(tableau)
                indices = [j for j, col in enumerate(columns[:-1]) if isPivotCol(col)]
                primal_solution = [(colIndex, variableValueForPivotColumn(tableau, columns[colIndex])) for colIndex in indices]
                objective_value = -(tableau[-1][-1])
                return tableau, primal_solution, objective_value
            if __name__ == "__main__":
                c = [12, 16]
                A = [[10, 20], [8, 8]]
                b = [120, 80]
                # add slack variables by hand
                A[0] += [1,0]
                A[1] += [0,1]
                c += [0,0]
                t, s, v = simplex(c, A, b)
                print(s)
                print(v)
        """,
        "after_lines": """
                for k,row in enumerate(tableau):
                    if k != i:
                        pivotRowMultiple = [y * tableau[k][j] for y in tableau[i]]
                        tableau[k] = [x - y for x,y in zip(tableau[k], pivotRowMultiple)]
                print("Tableau after pivot:")
                for row in tableau:
                    print(row)
                print()

            # the pivot columns denote which variables are used
            columns = transpose(tableau)
            indices = [j for j, col in enumerate(columns[:-1]) if isPivotCol(col)]
            primal_solution = [(colIndex, variableValueForPivotColumn(tableau, columns[colIndex])) for colIndex in indices]
            objective_value = -(tableau[-1][-1])
            return tableau, primal_solution, objective_value

            if __name__ == "__main__":
            c = [12, 16]
            A = [[10, 20], [8, 8]]
            b = [120, 80]

            # add slack variables by hand
            A[0] += [1,0]
            A[1] += [0,1]
            c += [0,0]

            t, s, v = simplex(c, A, b)
            print(s)
            print(v)
        """
    },
    {
        "description": "Case with test not overlapping with the code",
        "before_lines": """
            import heapq

            def dot(a,b):
            return sum(x*y for x,y in zip(a,b))

            def column(A, j):
            return [row[j] for row in A]

            def transpose(A):
            return [column(A, j) for j in range(len(A[0]))]

            def isPivotCol(col):
            return (len([c for c in col if c == 0]) == len(col) - 1) and sum(col) == 1

            def variableValueForPivotColumn(tableau, column):
            pivotRow = [i for (i, x) in enumerate(column) if x == 1][0]
            return tableau[pivotRow][-1]

            def canImprove(tableau):
            lastRow = tableau[-1]
            return any(x > 0 for x in lastRow[:-1])

            # this can be slightly faster
            def moreThanOneMin(L):
            if len(L) <= 1:
                return False
            x,y = heapq.nsmallest(2, L, key=lambda x: x[1])
            return x == y

            def identity(numRows, numCols, val=1, rowStart=0):
            return [[(val if i == j else 0) for j in range(numCols)] for i in range(rowStart, numRows)] 


            def standardForm(cost, greaterThans=[], gtThreshold=[], lessThans=[], ltThreshold=[], equalities=[], eqThreshold=[], maximization=True):
            newVars = 0
            numRows = 0
            if gtThreshold != []:
                newVars += len(gtThreshold)
                numRows += len(gtThreshold)
            if ltThreshold != []:
        """,
        "between_lines": """
                    
                newVars += len(ltThreshold)
                numRows += len(ltThreshold)

        """,
        "after_lines": """
            if eqThreshold != []:
                numRows += len(eqThreshold)
            if not maximization:
                cost = [-x for x in cost]
            if newVars == 0:
                return cost, equalities, eqThreshold

            newCost = list(cost) + [0] * newVars
            constraints = []
            threshold = []
            oldConstraints = [(greaterThans, gtThreshold, -1), (lessThans, ltThreshold, 1), (equalities, eqThreshold, 0)]
            offset = 0

            for constraintList, oldThreshold, coefficient in oldConstraints:
                constraints += [c + r for c, r in zip(constraintList, identity(numRows, newVars, coefficient, offset))]
                threshold += oldThreshold
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

            while canImprove(tableau):
                # pick minimum positive index of the last row
                column_choices = [(i,x) for (i,x) in enumerate(tableau[-1][:-1]) if x > 0]
                column = min(column_choices, key=lambda a: a[1])[0]

                # check if unbounded
                if all(row[column] <= 0 for row in tableau):
                    raise Exception('Linear program is unbounded.')

                # check for degeneracy: more than one minimizer of the quotient
                quotients = [(i, r[-1] / r[column]) for i,r in enumerate(tableau[:-1]) if r[column] > 0]

                if moreThanOneMin(quotients):
                    raise Exception('Linear program is degenerate.')

                # pick row index minimizing the quotient
                row = min(quotients, key=lambda x: x[1])[0]

                pivot = row, column

                print("Next pivot index is=%d,%d \n" % pivot)
                i,j = pivot
                pivotDenom = tableau[i][j]
                tableau[i] = [x / pivotDenom for x in tableau[i]]

                for k,row in enumerate(tableau):
                    if k != i:
                        pivotRowMultiple = [y * tableau[k][j] for y in tableau[i]]
                        tableau[k] = [x - y for x,y in zip(tableau[k], pivotRowMultiple)]
                print("Tableau after pivot:")
                for row in tableau:
                    print(row)
                print()
            
            # the pivot columns denote which variables are used
            columns = transpose(tableau)
            indices = [j for j, col in enumerate(columns[:-1]) if isPivotCol(col)]
            primal_solution = [(colIndex, variableValueForPivotColumn(tableau, columns[colIndex])) for colIndex in indices]
            objective_value = -(tableau[-1][-1])
            return tableau, primal_solution, objective_value

            if __name__ == "__main__":
            c = [12, 16]
            A = [[10, 20], [8, 8]]
            b = [120, 80]

            # add slack variables by hand
            A[0] += [1,0]
            A[1] += [0,1]
            c += [0,0]

            t, s, v = simplex(c, A, b)
            print(s)
            print(v)
        """
    },
    {
        "description": "Case with one line duplication",
        "before_lines": """
            import heapq

            def dot(a,b):
            return sum(x*y for x,y in zip(a,b))

            def column(A, j):
            return [row[j] for row in A]

            def transpose(A):
            return [column(A, j) for j in range(len(A[0]))]

            def isPivotCol(col):
            return (len([c for c in col if c == 0]) == len(col) - 1) and sum(col) == 1

            def variableValueForPivotColumn(tableau, column):
            pivotRow = [i for (i, x) in enumerate(column) if x == 1][0]
            return tableau[pivotRow][-1]

            def canImprove(tableau):
            lastRow = tableau[-1]
            return any(x > 0 for x in lastRow[:-1])

            # this can be slightly faster
            def moreThanOneMin(L):
            if len(L) <= 1:
                return False
            x,y = heapq.nsmallest(2, L, key=lambda x: x[1])
            return x == y

            def identity(numRows, numCols, val=1, rowStart=0):
            return [[(val if i == j else 0) for j in range(numCols)] for i in range(rowStart, numRows)] 


            def standardForm(cost, greaterThans=[], gtThreshold=[], lessThans=[], ltThreshold=[], equalities=[], eqThreshold=[], maximization=True):
            newVars = 0
            numRows = 0
            if gtThreshold != []:
                newVars += len(gtThreshold)
                numRows += len(gtThreshold)
            if ltThreshold != []:
                newVars += len(ltThreshold)
                numRows += len(ltThreshold)
            if eqThreshold != []:
                numRows += len(eqThreshold)
            if not maximization:
                cost = [-x for x in cost]
            if newVars == 0:
                return cost, equalities, eqThreshold

            newCost = list(cost) + [0] * newVars
            constraints = []
            threshold = []
            oldConstraints = [(greaterThans, gtThreshold, -1), (lessThans, ltThreshold, 1), (equalities, eqThreshold, 0)]
            offset = 0

            for constraintList, oldThreshold, coefficient in oldConstraints:
        """,
        "between_lines": """
                constraintsPrime = []
                for i in range(len(constraintsList)):
                    constraintCoefficients = constraintsList[i]
                    threshold.append(oldThreshold[i])
                    constraintsPrime.append(constraintCoefficients + [0]*offset + [coefficient] + [0]*(newVars - offset - 1))
                    offset += 1
                constraints += constraintsPrime
            return newCost, constraints, threshold
        """,
        "after_lines": """
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

        while canImprove(tableau):
            # pick minimum positive index of the last row
            column_choices = [(i,x) for (i,x) in enumerate(tableau[-1][:-1]) if x > 0]
            column = min(column_choices, key=lambda a: a[1])[0]

            # check if unbounded
            if all(row[column] <= 0 for row in tableau):
                raise Exception('Linear program is unbounded.')

            # check for degeneracy: more than one minimizer of the quotient
            quotients = [(i, r[-1] / r[column]) for i,r in enumerate(tableau[:-1]) if r[column] > 0]

            if moreThanOneMin(quotients):
                raise Exception('Linear program is degenerate.')

            # pick row index minimizing the quotient
            row = min(quotients, key=lambda x: x[1])[0]

            pivot = row, column

            print("Next pivot index is=%d,%d \n" % pivot)
            i,j = pivot
            pivotDenom = tableau[i][j]
            tableau[i] = [x / pivotDenom for x in tableau[i]]

            for k,row in enumerate(tableau):
                if k != i:
                    pivotRowMultiple = [y * tableau[k][j] for y in tableau[i]]
                    tableau[k] = [x - y for x,y in zip(tableau[k], pivotRowMultiple)]
            print("Tableau after pivot:")
            for row in tableau:
                print(row)
            print()
        
        # the pivot columns denote which variables are used
        columns = transpose(tableau)
        indices = [j for j, col in enumerate(columns[:-1]) if isPivotCol(col)]
        primal_solution = [(colIndex, variableValueForPivotColumn(tableau, columns[colIndex])) for colIndex in indices]
        objective_value = -(tableau[-1][-1])
        return tableau, primal_solution, objective_value
        """
    },
    {
        "description": "Case with duplications with indentation differences",
        "before_lines": """
            import heapq

            def dot(a,b):
            return sum(x*y for x,y in zip(a,b))

            def column(A, j):
            return [row[j] for row in A]

            def transpose(A):
            return [column(A, j) for j in range(len(A[0]))]

            def isPivotCol(col):
            return (len([c for c in col if c == 0]) == len(col) - 1) and sum(col) == 1

            def variableValueForPivotColumn(tableau, column):
            pivotRow = [i for (i, x) in enumerate(column) if x == 1][0]
            return tableau[pivotRow][-1]

            def canImprove(tableau):
            lastRow = tableau[-1]
            return any(x > 0 for x in lastRow[:-1])

            # this can be slightly faster
            def moreThanOneMin(L):
            if len(L) <= 1:
                return False
            x,y = heapq.nsmallest(2, L, key=lambda x: x[1])
            return x == y

            def identity(numRows, numCols, val=1, rowStart=0):
            return [[(val if i == j else 0) for j in range(numCols)] for i in range(rowStart, numRows)] 


            def standardForm(cost, greaterThans=[], gtThreshold=[], lessThans=[], ltThreshold=[], equalities=[], eqThreshold=[], maximization=True):
            newVars = 0
            numRows = 0
            if gtThreshold != []:
                newVars += len(gtThreshold)
                numRows += len(gtThreshold)
            if ltThreshold != []:
                newVars += len(ltThreshold)
                numRows += len(ltThreshold)
            if eqThreshold != []:
                numRows += len(eqThreshold)
            if not maximization:
                cost = [-x for x in cost]
            if newVars == 0:
                return cost, equalities, eqThreshold

            newCost = list(cost) + [0] * newVars
            constraints = []
            threshold = []
            oldConstraints = [(greaterThans, gtThreshold, -1), (lessThans, ltThreshold, 1), (equalities, eqThreshold, 0)]
            offset = 0

            for constraintList, oldThreshold, coefficient in oldConstraints:
                constraints += [c + r for c, r in zip(constraintList, identity(numRows, newVars, coefficient, offset))]
                threshold += oldThreshold
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

            while canImprove(tableau):
                # pick minimum positive index of the last row
                column_choices = [(i,x) for (i,x) in enumerate(tableau[-1][:-1]) if x > 0]
                column = min(column_choices, key=lambda a: a[1])[0]

                # check if unbounded
                if all(row[column] <= 0 for row in tableau):
                    raise Exception('Linear program is unbounded.')

                # check for degeneracy: more than one minimizer of the quotient
                quotients = [(i, r[-1] / r[column]) for i,r in enumerate(tableau[:-1]) if r[column] > 0]

                if moreThanOneMin(quotients):
                    raise Exception('Linear program is degenerate.')

                # pick row index minimizing the quotient
                row = min(quotients, key=lambda x: x[1])[0]

                pivot = row, column

                print("Next pivot index is=%d,%d \n" % pivot)
                i,j = pivot
        """,
        "between_lines": """
                piv = tableau[i][j]
                tableau[i] = [x / piv for x in tableau[i]]
                for k,row in enumerate(tableau):
                    if k != i:
                    pivotRowMultiple = [y * tableau[k][j] for y in tableau[i]]
                    tableau[k] = [x - y for x,y in zip(tableau[k], pivotRowMultiple)]
        """,
        "after_lines": """
                for k,row in enumerate(tableau):
                    if k != i:
                        pivotRowMultiple = [y * tableau[k][j] for y in tableau[i]]
                        tableau[k] = [x - y for x,y in zip(tableau[k], pivotRowMultiple)]
                print("Tableau after pivot:")
                for row in tableau:
                    print(row)
                print()
            
            # the pivot columns denote which variables are used
            columns = transpose(tableau)
            indices = [j for j, col in enumerate(columns[:-1]) if isPivotCol(col)]
            primal_solution = [(colIndex, variableValueForPivotColumn(tableau, columns[colIndex])) for colIndex in indices]
            objective_value = -(tableau[-1][-1])
            return tableau, primal_solution, objective_value
        """
    },
    {
        "description": "Case with indentation problems",
        "before_lines": """
            import heapq

            def dot(a,b):
            return sum(x*y for x,y in zip(a,b))

            def column(A, j):
            return [row[j] for row in A]

            def transpose(A):
            return [column(A, j) for j in range(len(A[0]))]

            def isPivotCol(col):
            return (len([c for c in col if c == 0]) == len(col) - 1) and sum(col) == 1

            def variableValueForPivotColumn(tableau, column):
            pivotRow = [i for (i, x) in enumerate(column) if x == 1][0]
            return tableau[pivotRow][-1]

            def canImprove(tableau):
            lastRow = tableau[-1]
            return any(x > 0 for x in lastRow[:-1])

            # this can be slightly faster
            def moreThanOneMin(L):
            if len(L) <= 1:
                return False
            x,y = heapq.nsmallest(2, L, key=lambda x: x[1])
            return x == y

            def identity(numRows, numCols, val=1, rowStart=0):
            return [[(val if i == j else 0) for j in range(numCols)] for i in range(rowStart, numRows)] 


            def standardForm(cost, greaterThans=[], gtThreshold=[], lessThans=[], ltThreshold=[], equalities=[], eqThreshold=[], maximization=True):
            newVars = 0
            numRows = 0
            if gtThreshold != []:
                newVars += len(gtThreshold)
                numRows += len(gtThreshold)
            if ltThreshold != []:
                newVars += len(ltThreshold)
                numRows += len(ltThreshold)
            if eqThreshold != []:
                numRows += len(eqThreshold)
            if not maximization:
                cost = [-x for x in cost]
            if newVars == 0:
                return cost, equalities, eqThreshold

            newCost = list(cost) + [0] * newVars
            constraints = []
            threshold = []
            oldConstraints = [(greaterThans, gtThreshold, -1), (lessThans, ltThreshold, 1), (equalities, eqThreshold, 0)]
            offset = 0

            for constraintList, oldThreshold, coefficient in oldConstraints:
                constraints += [c + r for c, r in zip(constraintList, identity(numRows, newVars, coefficient, offset))]
                threshold += oldThreshold
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
        """,
        "between_lines": """
                j = 0
                while j < len(tableau[i]):
                    tableau[i][j] *= obj_val_multiplier
                    j += 1
                i += 1
        """,
        "after_lines": """
                print("Initial tableau:")
                for row in tableau:
                    print(row)
                print()

                while canImprove(tableau):
                    # pick minimum positive index of the last row
                    column_choices = [(i,x) for (i,x) in enumerate(tableau[-1][:-1]) if x > 0]
                    column = min(column_choices, key=lambda a: a[1])[0]

                    # check if unbounded
                    if all(row[column] <= 0 for row in tableau):
                        raise Exception('Linear program is unbounded.')

                    # check for degeneracy: more than one minimizer of the quotient
                    quotients = [(i, r[-1] / r[column]) for i,r in enumerate(tableau[:-1]) if r[column] > 0]

                    if moreThanOneMin(quotients):
                        raise Exception('Linear program is degenerate.')

                    # pick row index minimizing the quotient
                    row = min(quotients, key=lambda x: x[1])[0]

                    pivot = row, column

                    print("Next pivot index is=%d,%d \n" % pivot)
                    i,j = pivot
                    pivotDenom = tableau[i][j]
                    tableau[i] = [x / pivotDenom for x in tableau[i]]

                    for k,row in enumerate(tableau):
                        if k != i:
                            pivotRowMultiple = [y * tableau[k][j] for y in tableau[i]]
                            tableau[k] = [x - y for x,y in zip(tableau[k], pivotRowMultiple)]
                    print("Tableau after pivot:")
                    for row in tableau:
                        print(row)
                    print()
                
                # the pivot columns denote which variables are used
                columns = transpose(tableau)
                indices = [j for j, col in enumerate(columns[:-1]) if isPivotCol(col)]
                primal_solution = [(colIndex, variableValueForPivotColumn(tableau, columns[colIndex])) for colIndex in indices]
                objective_value = -(tableau[-1][-1])
                return tableau, primal_solution, objective_value
        """
    },
    {
        "description": "Case left the last line not removed",
        "before_lines": """
            import heapq

            def dot(a,b):
            return sum(x*y for x,y in zip(a,b))

            def column(A, j):
            return [row[j] for row in A]

            def transpose(A):
            return [column(A, j) for j in range(len(A[0]))]

            def isPivotCol(col):
            return (len([c for c in col if c == 0]) == len(col) - 1) and sum(col) == 1

            def variableValueForPivotColumn(tableau, column):
            pivotRow = [i for (i, x) in enumerate(column) if x == 1][0]
            return tableau[pivotRow][-1]

            def canImprove(tableau):
            lastRow = tableau[-1]
            return any(x > 0 for x in lastRow[:-1])

            # this can be slightly faster
            def moreThanOneMin(L):
            if len(L) <= 1:
                return False
            x,y = heapq.nsmallest(2, L, key=lambda x: x[1])
            return x == y

            def identity(numRows, numCols, val=1, rowStart=0):
            return [[(val if i == j else 0) for j in range(numCols)] for i in range(rowStart, numRows)] 


            def standardForm(cost, greaterThans=[], gtThreshold=[], lessThans=[], ltThreshold=[], equalities=[], eqThreshold=[], maximization=True):
            newVars = 0
            numRows = 0
            if gtThreshold != []:
                newVars += len(gtThreshold)
                numRows += len(gtThreshold)
            if ltThreshold != []:
                newVars += len(ltThreshold)
                numRows += len(ltThreshold)
            if eqThreshold != []:
                numRows += len(eqThreshold)
            if not maximization:
                cost = [-x for x in cost]
            if newVars == 0:
                return cost, equalities, eqThreshold

            newCost = list(cost) + [0] * newVars
            constraints = []
            threshold = []
            oldConstraints = [(greaterThans, gtThreshold, -1), (lessThans, ltThreshold, 1), (equalities, eqThreshold, 0)]
            offset = 0

            for constraintList, oldThreshold, coefficient in oldConstraints:
                constraints += [c + r for c, r in zip(constraintList, identity(numRows, newVars, coefficient, offset))]
                threshold += oldThreshold
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

            while canImprove(tableau):
                # pick minimum positive index of the last row
                column_choices = [(i,x) for (i,x) in enumerate(tableau[-1][:-1]) if x > 0]
                column = min(column_choices, key=lambda a: a[1])[0]

                # check if unbounded
                if all(row[column] <= 0 for row in tableau):
                    raise Exception('Linear program is unbounded.')

                # check for degeneracy: more than one minimizer of the quotient
                quotients = [(i, r[-1] / r[column]) for i,r in enumerate(tableau[:-1]) if r[column] > 0]

                if moreThanOneMin(quotients):
                    raise Exception('Linear program is degenerate.')

                # pick row index minimizing the quotient
                row = min(quotients, key=lambda x: x[1])[0]

                pivot = row, column

                print("Next pivot index is=%d,%d \n" % pivot)
                i,j = pivot
        """,
        "between_lines": """
                print("Pivoting at {}:{}".format(i+1, j+1))
                # pivot
                pivotVal = tableau[i][j]
                tableau[i] = [x / pivotVal for x in tableau[i]]
                for k, row in enumerate(tableau):
                    if k != i:
                        ratio = row[j]
                        tableau[k] = [x - ratio * y for x, y in zip(row, tableau[i])]
                coefficients = [c for c in tableau[-1] if c != 0]
                for l, element in enumerate(coefficients[:-1]):
                    if element > 0:
                        column = l
                        break

        """,
        "after_lines": """
                for k,row in enumerate(tableau):
                    if k != i:
                        pivotRowMultiple = [y * tableau[k][j] for y in tableau[i]]
                        tableau[k] = [x - y for x,y in zip(tableau[k], pivotRowMultiple)]
                print("Tableau after pivot:")
                for row in tableau:
                    print(row)
                print()
            
            # the pivot columns denote which variables are used
            columns = transpose(tableau)
            indices = [j for j, col in enumerate(columns[:-1]) if isPivotCol(col)]
            primal_solution = [(colIndex, variableValueForPivotColumn(tableau, columns[colIndex])) for colIndex in indices]
            objective_value = -(tableau[-1][-1])
            return tableau, primal_solution, objective_value
        """
    },
    {
        "description": "Case incorrectly remove the indentation",
        "before_lines": """
            import heapq

            def dot(a,b):
            return sum(x*y for x,y in zip(a,b))

            def column(A, j):
            return [row[j] for row in A]

            def transpose(A):
            return [column(A, j) for j in range(len(A[0]))]

            def isPivotCol(col):
            return (len([c for c in col if c == 0]) == len(col) - 1) and sum(col) == 1

            def variableValueForPivotColumn(tableau, column):
            pivotRow = [i for (i, x) in enumerate(column) if x == 1][0]
            return tableau[pivotRow][-1]

            def canImprove(tableau):
            lastRow = tableau[-1]
            return any(x > 0 for x in lastRow[:-1])

            # this can be slightly faster
            def moreThanOneMin(L):
            if len(L) <= 1:
                return False
            x,y = heapq.nsmallest(2, L, key=lambda x: x[1])
            return x == y

            def identity(numRows, numCols, val=1, rowStart=0):
            return [[(val if i == j else 0) for j in range(numCols)] for i in range(rowStart, numRows)] 


            def standardForm(cost, greaterThans=[], gtThreshold=[], lessThans=[], ltThreshold=[], equalities=[], eqThreshold=[], maximization=True):
            newVars = 0
            numRows = 0
            if gtThreshold != []:
                newVars += len(gtThreshold)
                numRows += len(gtThreshold)
            if ltThreshold != []:
                newVars += len(ltThreshold)
                numRows += len(ltThreshold)
            if eqThreshold != []:
                numRows += len(eqThreshold)
            if not maximization:
                cost = [-x for x in cost]
            if newVars == 0:
                return cost, equalities, eqThreshold

            newCost = list(cost) + [0] * newVars
            constraints = []
            threshold = []
            oldConstraints = [(greaterThans, gtThreshold, -1), (lessThans, ltThreshold, 1), (equalities, eqThreshold, 0)]
            offset = 0

            for constraintList, oldThreshold, coefficient in oldConstraints:
                constraints += [c + r for c, r in zip(constraintList, identity(numRows, newVars, coefficient, offset))]
                threshold += oldThreshold
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

            while canImprove(tableau):
                # pick minimum positive index of the last row
                column_choices = [(i,x) for (i,x) in enumerate(tableau[-1][:-1]) if x > 0]
                column = min(column_choices, key=lambda a: a[1])[0]

                # check if unbounded
                if all(row[column] <= 0 for row in tableau):
                    raise Exception('Linear program is unbounded.')

                # check for degeneracy: more than one minimizer of the quotient
                quotients = [(i, r[-1] / r[column]) for i,r in enumerate(tableau[:-1]) if r[column] > 0]

                if moreThanOneMin(quotients):
                    raise Exception('Linear program is degenerate.')

                # pick row index minimizing the quotient
                row = min(quotients, key=lambda x: x[1])[0]

                pivot = row, column

                print("Next pivot index is=%d,%d \n" % pivot)
                i,j = pivot
                pivotDenom = tableau[i][j]
                tableau[i] = [x / pivotDenom for x in tableau[i]]

                for k,row in enumerate(tableau):
        """,
        "between_lines": """
                if k != i: # for each row below and above pivot row
                    pivotRowMultiple = [y * tableau[k][j] for y in tableau[i]]
                    tableau[k] = [x - y for x,y in zip(tableau[k], pivotRowMultiple)]

        """,
        "after_lines": """
                print("Tableau after pivot:")
                for row in tableau:
                    print(row)
                print()
            
            # the pivot columns denote which variables are used
            columns = transpose(tableau)
            indices = [j for j, col in enumerate(columns[:-1]) if isPivotCol(col)]
            primal_solution = [(colIndex, variableValueForPivotColumn(tableau, columns[colIndex])) for colIndex in indices]
            objective_value = -(tableau[-1][-1])
            return tableau, primal_solution, objective_value
        """
    },
    {
        "description": "Case to test if detect the begining of after_lines",
        "before_lines": """
            import heapq

            def dot(a,b):
                return sum(x*y for x,y in zip(a,b))

            def column(A, j):
                return [row[j] for row in A]

            def transpose(A):
                return [column(A, j) for j in range(len(A[0]))]

            def isPivotCol(col):
                return (len([c for c in col if c == 0]) == len(col) - 1) and sum(col) == 1

            def variableValueForPivotColumn(tableau, column):
                pivotRow = [i for (i, x) in enumerate(column) if x == 1][0]
                return tableau[pivotRow][-1]

            def canImprove(tableau):
                lastRow = tableau[-1]
                return any(x > 0 for x in lastRow[:-1])

            # this can be slightly faster
            def moreThanOneMin(L):
                if len(L) <= 1:
                    return False
                x,y = heapq.nsmallest(2, L, key=lambda x: x[1])
                return x == y

            def identity(numRows, numCols, val=1, rowStart=0):
                return [[(val if i == j else 0) for j in range(numCols)] for i in range(rowStart, numRows)] 


            def standardForm(cost, greaterThans=[], gtThreshold=[], lessThans=[], ltThreshold=[], equalities=[], eqThreshold=[], maximization=True):
                newVars = 0
                numRows = 0
                if gtThreshold != []:
                    newVars += len(gtThreshold)
                    numRows += len(gtThreshold)
                if ltThreshold != []:
        """,
        "between_lines": """
            def dot(a,b):
                return sum(x*y for x,y in zip(a,b))
            def column(A, j):
                return [row[j] for row in A]
            def transpose(A):
                return [column(A, j) for j in range(len(A[0]))]
            def isPivotCol(col):
                return (len([c for c in col if c == 0]) == len(col) - 1) and sum(col) == 1
            def variableValueForPivotColumn(tableau, column):
                pivotRow = [i for (i, x) in enumerate(column) if x == 1][0]
                return tableau[pivotRow][-1]
            def canImprove(tableau):
                lastRow = tableau[-1]
                return any(x > 0 for x in lastRow[:-1])
            def moreThanOneMin(L):
                if len(L) <= 1:
                    return False
                x,y = heapq.nsmallest(2, L, key=lambda x: x[1])
                return x == y
            def identity(numRows, numCols, val=1, rowStart=0):
                return [[(val if i == j else 0) for j in range(numCols)] for i in range(rowStart, numRows)]
            def standardForm(cost, greaterThans=[], gtThreshold=[], lessThans=[], ltThreshold=[], equalities=[], eqThreshold=[], maximization=True):
                newVars = 0
                numRows = 0
                if gtThreshold != []:
                    newVars += len(gtThreshold)
                    numRows += len(gtThreshold)
                if ltThreshold != []:
                    newVars += len(ltThreshold)
                    numRows += len(ltThreshold)
                newCost = list(cost) + [0] * newVars
                constraints = []
                threshold = []
                oldConstraints = [(greaterThans, gtThreshold, -1), (lessThans, ltThreshold, 1), (equalities, eqThreshold, 0)]
                offset = 0
                for constraintList, oldThreshold, coefficient in oldConstraints:
                    constraints += [c + r for c, r in zip(constraintList, identity(numRows, newVars, coefficient, offset))]
                    threshold += oldThreshold
                    offset += len(oldThreshold)
                return newCost, constraints, threshold
                max <c,x>
                s.t. Ax = b
                    x >= 0
            def simplex(c, A, b):
                # assume the last m columns of A are the slack variables; the initial basis is the set of slack variables
                tableau = [row[:] + [x] for row, x in zip(A, b)]
                tableau.append([ci for ci in c])
                print("Initial tableau:")
                for row in tableau:
                    print(row)
                print()
                while canImprove(tableau):
                    # pick minimum positive index of the last row
                    column_choices = [(i,x) for (i,x) in enumerate(tableau[-1][:-1]) if x > 0]
                    column = min(column_choices, key=lambda a: a[1])[0]
                    # check if unbounded
                    if all(row[column] <= 0 for row in tableau):
                        raise Exception('Linear program is unbounded.')
                    # check for degeneracy: more than one minimizer of the quotient
                    quotients = [(i, r[-1] / r[column]) for i,r in enumerate(tableau[:-1]) if r[column] > 0]
                    if moreThanOneMin(quotients):
                        raise Exception('Linear program is degenerate.')
                    # pick row index minimizing the quotient
                    row = min(quotients, key=lambda x: x[1])[0]
                    pivot = row, column
                    print("Next pivot index is=%d,%d \n" % pivot)
                    i,j = pivot
                    pivotDenom = tableau[i][j]
                    tableau[i] = [x / pivotDenom for x in tableau[i]]
                    for k,row in enumerate(tableau):
                        if k != i:
                            pivotRowMultiple = [y * tableau[k][j] for y in tableau[i]]
                            tableau[k] = [x - y for x,y in zip(tableau[k], pivotRowMultiple)]
                    print("Tableau after pivot:")
                    for row in tableau:
                        print(row)
                    print()
                # the pivot columns denote which variables are used
                columns = transpose(tableau)
                indices = [j for j, col in enumerate(columns[:-1]) if isPivotCol(col)]
                primal_solution = [(colIndex, variableValueForPivotColumn(tableau, columns[colIndex])) for colIndex in indices]
                objective_value = -(tableau[-1][-1])
                return tableau, primal_solution, objective_value
        """,
        "after_lines": """
                newCost = list(cost) + [0] * newVars
                constraints = []
                threshold = []
                oldConstraints = [(greaterThans, gtThreshold, -1), (lessThans, ltThreshold, 1), (equalities, eqThreshold, 0)]
                offset = 0

                for constraintList, oldThreshold, coefficient in oldConstraints:
                    constraints += [c + r for c, r in zip(constraintList, identity(numRows, newVars, coefficient, offset))]
                    threshold += oldThreshold
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

                while canImprove(tableau):
                    # pick minimum positive index of the last row
                    column_choices = [(i,x) for (i,x) in enumerate(tableau[-1][:-1]) if x > 0]
                    column = min(column_choices, key=lambda a: a[1])[0]

                    # check if unbounded
                    if all(row[column] <= 0 for row in tableau):
                        raise Exception('Linear program is unbounded.')

                    # check for degeneracy: more than one minimizer of the quotient
                    quotients = [(i, r[-1] / r[column]) for i,r in enumerate(tableau[:-1]) if r[column] > 0]

                    if moreThanOneMin(quotients):
                        raise Exception('Linear program is degenerate.')

                    # pick row index minimizing the quotient
                    row = min(quotients, key=lambda x: x[1])[0]

                    pivot = row, column

                    print("Next pivot index is=%d,%d \n" % pivot)
                    i,j = pivot
                    pivotDenom = tableau[i][j]
                    tableau[i] = [x / pivotDenom for x in tableau[i]]

                    for k,row in enumerate(tableau):
                        if k != i:
                            pivotRowMultiple = [y * tableau[k][j] for y in tableau[i]]
                            tableau[k] = [x - y for x,y in zip(tableau[k], pivotRowMultiple)]
                    print("Tableau after pivot:")
                    for row in tableau:
                        print(row)
                    print()
                
                # the pivot columns denote which variables are used
                columns = transpose(tableau)
                indices = [j for j, col in enumerate(columns[:-1]) if isPivotCol(col)]
                primal_solution = [(colIndex, variableValueForPivotColumn(tableau, columns[colIndex])) for colIndex in indices]
                objective_value = -(tableau[-1][-1])
                return tableau, primal_solution, objective_value
        """
    },
    {
        "description": "Case to test if detect the end of before_lines",
        "before_lines": """
            import heapq

            def dot(a,b):
                return sum(x*y for x,y in zip(a,b))

            def column(A, j):
                return [row[j] for row in A]

            def transpose(A):
                return [column(A, j) for j in range(len(A[0]))]

            def isPivotCol(col):
                return (len([c for c in col if c == 0]) == len(col) - 1) and sum(col) == 1

            def variableValueForPivotColumn(tableau, column):
                pivotRow = [i for (i, x) in enumerate(column) if x == 1][0]
                return tableau[pivotRow][-1]

            def canImprove(tableau):
                lastRow = tableau[-1]
                return any(x > 0 for x in lastRow[:-1])

            # this can be slightly faster
            def moreThanOneMin(L):
                if len(L) <= 1:
                    return False
                x,y = heapq.nsmallest(2, L, key=lambda x: x[1])
                return x == y

            def identity(numRows, numCols, val=1, rowStart=0):
                return [[(val if i == j else 0) for j in range(numCols)] for i in range(rowStart, numRows)] 


            def standardForm(cost, greaterThans=[], gtThreshold=[], lessThans=[], ltThreshold=[], equalities=[], eqThreshold=[], maximization=True):
                newVars = 0
                numRows = 0
                if gtThreshold != []:
                    newVars += len(gtThreshold)
                    numRows += len(gtThreshold)
                if ltThreshold != []:
                    newVars += len(ltThreshold)
                    numRows += len(ltThreshold)
                if eqThreshold != []:
                    numRows += len(eqThreshold)
                if not maximization:
                    cost = [-x for x in cost]
                if newVars == 0:
                    return cost, equalities, eqThreshold

                newCost = list(cost) + [0] * newVars
                constraints = []
                threshold = []
                oldConstraints = [(greaterThans, gtThreshold, -1), (lessThans, ltThreshold, 1), (equalities, eqThreshold, 0)]
                offset = 0

                for constraintList, oldThreshold, coefficient in oldConstraints:
                    constraints += [c + r for c, r in zip(constraintList, identity(numRows, newVars, coefficient, offset))]
                    threshold += oldThreshold
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

                while canImprove(tableau):
                    # pick minimum positive index of the last row
                    column_choices = [(i,x) for (i,x) in enumerate(tableau[-1][:-1]) if x > 0]
                    column = min(column_choices, key=lambda a: a[1])[0]

                    # check if unbounded
                    if all(row[column] <= 0 for row in tableau):
                        raise Exception('Linear program is unbounded.')
        """,
        "between_lines": """
            def dot(a,b):
            def column(A, j):
            def transpose(A):
            def isPivotCol(col):
            def variableValueForPivotColumn(tableau, column):
            def canImprove(tableau):
            def moreThanOneMin(L):
                return False
            def identity(numRows, numCols, val=1, rowStart=0):
            def standardForm(cost, greaterThans=[], gtThreshold=[], lessThans=[], ltThreshold=[], equalities=[], eqThreshold=[], maximization=True):
                newVars += len(gtThreshold)
                numRows += len(gtThreshold)
                newVars += len(ltThreshold)
                numRows += len(ltThreshold)
                numRows += len(eqThreshold)
                cost = [-x for x in cost]
                return cost, equalities, eqThreshold
                constraints += [c + r for c, r in zip(constraintList, identity(numRows, newVars, coefficient, offset))]
                threshold += oldThreshold
                offset += len(oldThreshold)
                max <c,x>
                s.t. Ax = b
                    x >= 0
            def simplex(c, A, b):
                print(row)
                # pick minimum positive index of the last row
                column_choices = [(i,x) for (i,x) in enumerate(tableau[-1][:-1]) if x > 0]
                column = min(column_choices, key=lambda a: a[1])[0]
                # check if unbounded
                if all(row[column] <= 0 for row in tableau):
                    raise Exception('Linear program is unbounded.')
                quotients = []
                pivotCol = [row[column] for row in tableau[:-1]]
                for i in range(len(tableau)-1):
                    pivotRowMultiple = [row[column] for row in tableau[:-1]]
                    quotients.append((i,tableau[i][-1]/pivotRowMultiple[i]))
                if moreThanOneMin(quotients):
                    raise Exception('Linear program is degenerate.')
                # pick row index minimizing the quotient
                row = min(quotients, key=lambda x: x[1])[0]
                pivot = row, column
                print("Next pivot index is=%d,%d \n" % pivot)
                i,j = pivot
                pivotDenom = tableau[i][j]
                tableau[i] = [x / pivotDenom for x in tableau[i]]
                for k,row in enumerate(tableau):
                    if k != i:
                        pivotRowMultiple = [y * tableau[k][j] for y in tableau[i]]
                        tableau[k] = [x - y for x,y in zip(tableau[k], pivotRowMultiple)]
                print("Tableau after pivot:")
                for row in tableau:
                    print(row)
                print()
        """,
        "after_lines": """
                if moreThanOneMin(quotients):
                    raise Exception('Linear program is degenerate.')

                # pick row index minimizing the quotient
                row = min(quotients, key=lambda x: x[1])[0]

                pivot = row, column

                print("Next pivot index is=%d,%d \n" % pivot)
                i,j = pivot
                pivotDenom = tableau[i][j]
                tableau[i] = [x / pivotDenom for x in tableau[i]]

                for k,row in enumerate(tableau):
                    if k != i:
                        pivotRowMultiple = [y * tableau[k][j] for y in tableau[i]]
                        tableau[k] = [x - y for x,y in zip(tableau[k], pivotRowMultiple)]
                print("Tableau after pivot:")
                for row in tableau:
                    print(row)
                print()
            
            # the pivot columns denote which variables are used
            columns = transpose(tableau)
            indices = [j for j, col in enumerate(columns[:-1]) if isPivotCol(col)]
            primal_solution = [(colIndex, variableValueForPivotColumn(tableau, columns[colIndex])) for colIndex in indices]
            objective_value = -(tableau[-1][-1])
            return tableau, primal_solution, objective_value
        """
    },
    {
        "description": "Case to test if detect the end of before_lines: the last line of generated matched the both end of before_lines & beginning of after_lines",
        "before_lines": """
            import heapq

            def dot(a,b):
                return sum(x*y for x,y in zip(a,b))

            def column(A, j):
                return [row[j] for row in A]

            def transpose(A):
                return [column(A, j) for j in range(len(A[0]))]

            def isPivotCol(col):
                return (len([c for c in col if c == 0]) == len(col) - 1) and sum(col) == 1

            def variableValueForPivotColumn(tableau, column):
                pivotRow = [i for (i, x) in enumerate(column) if x == 1][0]
                return tableau[pivotRow][-1]

            def canImprove(tableau):
                lastRow = tableau[-1]
                return any(x > 0 for x in lastRow[:-1])

            # this can be slightly faster
            def moreThanOneMin(L):
                if len(L) <= 1:
                    return False
                x,y = heapq.nsmallest(2, L, key=lambda x: x[1])
                return x == y

            def identity(numRows, numCols, val=1, rowStart=0):
                return [[(val if i == j else 0) for j in range(numCols)] for i in range(rowStart, numRows)] 


            def standardForm(cost, greaterThans=[], gtThreshold=[], lessThans=[], ltThreshold=[], equalities=[], eqThreshold=[], maximization=True):
                newVars = 0
                numRows = 0
                if gtThreshold != []:
                    newVars += len(gtThreshold)
                    numRows += len(gtThreshold)
                if ltThreshold != []:
                    newVars += len(ltThreshold)
                    numRows += len(ltThreshold)
                if eqThreshold != []:
                    numRows += len(eqThreshold)
                if not maximization:
                    cost = [-x for x in cost]
                if newVars == 0:
                    return cost, equalities, eqThreshold

                newCost = list(cost) + [0] * newVars
                constraints = []
                threshold = []
                oldConstraints = [(greaterThans, gtThreshold, -1), (lessThans, ltThreshold, 1), (equalities, eqThreshold, 0)]
                offset = 0

                for constraintList, oldThreshold, coefficient in oldConstraints:
                    constraints += [c + r for c, r in zip(constraintList, identity(numRows, newVars, coefficient, offset))]
                    threshold += oldThreshold
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

                while canImprove(tableau):
                    # pick minimum positive index of the last row
                    column_choices = [(i,x) for (i,x) in enumerate(tableau[-1][:-1]) if x > 0]
                    column = min(column_choices, key=lambda a: a[1])[0]

                # check if unbounded
                if all(row[column] <= 0 for row in tableau):
                    raise Exception('Linear program is unbounded.')
        """,
        "between_lines": """
                # separate out positive values and their indices
                quotients = [(i, r[-1] / r[column]) for i,r in enumerate(tableau[:-1]) if r[column] > 0]
                # no positive entries means it's unbounded
                if not quotients:
                    raise Exception('Linear program is unbounded.')
        """,
        "after_lines": """
                if moreThanOneMin(quotients):
                    raise Exception('Linear program is degenerate.')

                # pick row index minimizing the quotient
                row = min(quotients, key=lambda x: x[1])[0]

                pivot = row, column

                print("Next pivot index is=%d,%d \n" % pivot)
                i,j = pivot
                pivotDenom = tableau[i][j]
                tableau[i] = [x / pivotDenom for x in tableau[i]]

                for k,row in enumerate(tableau):
                    if k != i:
                        pivotRowMultiple = [y * tableau[k][j] for y in tableau[i]]
                        tableau[k] = [x - y for x,y in zip(tableau[k], pivotRowMultiple)]
                print("Tableau after pivot:")
                for row in tableau:
                    print(row)
                print()
            
            # the pivot columns denote which variables are used
            columns = transpose(tableau)
            indices = [j for j, col in enumerate(columns[:-1]) if isPivotCol(col)]
            primal_solution = [(colIndex, variableValueForPivotColumn(tableau, columns[colIndex])) for colIndex in indices]
            objective_value = -(tableau[-1][-1])
            return tableau, primal_solution, objective_value
        """
    },
]

test_cases_remove_duplicates_trace_non_comments = [
    {
        "description": "Case incorrectly tracking the comments at the begining of the after_lines or ending of before_lines",
        "before_lines": """
            import heapq

            # Function to calculate the dot product of two vectors
            def dot(a, b):
                # Finish the the current function and stop generation by the end of the function
                return sum(x*y for x, y in zip(a, b))

            # Function to get a specific column from a 2D list (matrix)
            def column(A, j):
                # Finish the the current function and stop generation by the end of the function
                return [row[j] for row in A]

            # Function to transpose a 2D list (matrix)
            def transpose(A):
                # Finish the the current function and stop generation by the end of the function
        """,
        "between_lines": """
            def dot(a, b):
                # Finish the the current function and stop generation by the end of the function
                return sum(x*y for x, y in zip(a, b))
            def column(A, j):
                # Finish the the current function and stop generation by the end of the function
                return [row[j] for row in A]
            def transpose(A):
                Finish the the current function and stop generation by the end of the function
                here is to test if clean code works properly 
                return list(map(list, zip(*A)))
            def isPivotCol(col):
                # Finish the the current function and stop generation by the end of the function
                return (len([c for c in col if c == 0]) == len(col) - 1) and sum(col) == 1
            def variableValueForPivotColumn(tableau, column):
                # Finish the the current function and stop generation by the end of the function
                pivotRow = [i for (i, x) in enumerate(column) if x == 1][0]
                return tableau[pivotRow][-1]
            def canImprove(tableau):
                # Finish the the current function and stop generation by the end of the function
                lastRow = tableau[-1]
                return any(x > 0 for x in lastRow[:-1])
            def moreThanOneMin(L):
                if len(L) <= 1:
                    return False
                # Finish the the current function and stop generation by the end of the function
                x, y = heapq.nsmallest(2, L, key=lambda x: x[1])
                return x == y
            def identity(numRows, numCols, val=1, rowStart=0):
                # Finish the the current function and stop generation by the end of the function
                return [[(val if i == j else 0) for j in range(numCols)] for i in range(rowStart, numRows)]

        """,
        "after_lines": """
            # Function to check if a column is a pivot column
            def isPivotCol(col):
                # Finish the the current function and stop generation by the end of the function
                return (len([c for c in col if c == 0]) == len(col) - 1) and sum(col) == 1  

            # Function to find the value of a variable for a pivot column
            def variableValueForPivotColumn(tableau, column):
                # Finish the the current function and stop generation by the end of the function
                pivotRow = [i for (i, x) in enumerate(column) if x == 1][0]
                return tableau[pivotRow][-1]

            # Function to check if we can improve the current solution
            def canImprove(tableau):
                # Finish the the current function and stop generation by the end of the function
                lastRow = tableau[-1]
                return any(x > 0 for x in lastRow[:-1])

            # Function to check if there's more than one minimum in a list
            def moreThanOneMin(L):
                if len(L) <= 1:
                    return False
                # Finish the the current function and stop generation by the end of the function
                x, y = heapq.nsmallest(2, L, key=lambda x: x[1])
                return x == y

            # Function to create an identity matrix with certain specifications
            def identity(numRows, numCols, val=1, rowStart=0):
                # Finish the the current function and stop generation by the end of the function
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
                
                ##Given the example of greater than threshold case, complete the case of less than threshold and equal to the threshold.
                if ltThreshold != []:
                    newVars += len(ltThreshold)
                    numRows += len(ltThreshold)
                
                # Equalities don't need slack variables but add to the number of rows
                if eqThreshold != []:
                    numRows += len(eqThreshold)
                
                # If the problem is a minimization, convert it to a maximization by negating the cost vector
                if not maximization:
                    # Finish the current if statement and stop generation by the end of the if statement
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
                
                # Return the new cost function, the modified constraints, and the thresholds
                return newCost, constraints, threshold


            # Main simplex algorithm function
            def simplex(c, A, b):
                # Setup initial tableau from constraints and objective function
                tableau = [row[:] + [x] for row, x in zip(A, b)]
                tableau.append([ci for ci in c] + [0])
                print("Initial tableau:")
                for row in tableau:
                    print(row)
                print()

                # Iterate until no improvements can be made
                while canImprove(tableau):
                    # Choose entering variable (minimum positive index of the last row)
                    column_choices = [(i, x) for (i, x) in enumerate(tableau[-1][:-1]) if x > 0]
                    column = min(column_choices, key=lambda a: a[1])[0]

                    # Check for unboundedness
                    if all(row[column] <= 0 for row in tableau):
                        raise Exception('Linear program is unbounded.')

                    # Implement the code that checks for degeneracy: more than one minimizer of the quotient
                    quotients = [(i, r[-1] / r[column]) for i, r in enumerate(tableau[:-1]) if r[column] > 0]

                    if moreThanOneMin(quotients):
                        raise Exception('Linear program is degenerate.')

                    # Implement the code for chosing leaving variable (row index minimizing the quotient)
                    row = min(quotients, key=lambda x: x[1])[0]

                    # Implement the code that pivots on the chosen row and column
                    pivot = row, column
                    print("Next pivot index is=%d,%d \n" % pivot)
                    i, j = pivot
                    pivotDenom = tableau[i][j]
                    # Normalize the pivot row
                    tableau[i] = [x / pivotDenom for x in tableau[i]]

                    # Zero out the other entries in the pivot column
                    for k, row in enumerate(tableau):
                        # Finish the current for loop block and stop generation by the end of the for loop block
                        if k != i:
                            pivotRowMultiple = [y * tableau[k][j] for y in tableau[i]]
                            tableau[k] = [x - y for x, y in zip(tableau[k], pivotRowMultiple)]
                    print("Tableau after pivot:")
                    for row in tableau:
                        print(row)
                    print()
                    
                # Transpose the tableau to make it easier to work with columns
                columns = transpose(tableau)
                
                # Identify pivot columns in the tableau. A column is a pivot column if it has a single 1 and the rest of its entries are 0.
                indices = [j for j, col in enumerate(columns[:-1]) if isPivotCol(col)]
                
                # For each pivot column, find the value of the corresponding variable.
                # This is done by looking at the rightmost entry (the value part of the tableau row) of the row where the 1 in the pivot column is located.
                primal_solution = [(colIndex, variableValueForPivotColumn(tableau, columns[colIndex])) for colIndex in indices]

                # Calculate the objective value. The last entry of the last row of the tableau gives us the negation of the objective function value.
                objective_value = -(tableau[-1][-1])
                
                return tableau, primal_solution, objective_value
        """
    },
    {
        "description": "Case to ignore the comments of gen_code to find strat_idx & end_index",
        "before_lines": """
            import heapq

            # Function to calculate the dot product of two vectors
            def dot(a, b):
                # Finish the the current function and stop generation by the end of the function
        """,
        "between_lines": """
            def dot(a, b):
                # Calculate dot product and return the result
                return sum(x * y for x, y in zip(a, b))
            def column(A, j):
                return [row[j] for row in A]
            def transpose(A):
                return [column(A, j) for j in range(len(A[0]))]
            def isPivotCol(col):
                return (len([c for c in col if c == 0]) == len(col) - 1) and sum(col) == 1
            def variableValueForPivotColumn(tableau, column):
                pivotRow = [i for (i, x) in enumerate(column) if x == 1][0]
                return tableau[pivotRow][-1]
            def canImprove(tableau):
                lastRow = tableau[-1]
                return any(x > 0 for x in lastRow[:-1])
            def moreThanOneMin(L):
                # Find two smallest elements (x, y) and check if they are the same
                smallest, second_smallest = heapq.nsmallest(2, L, key=lambda x: x[1])
                return smallest[1] == second_smallest[1]
            def identity(numRows, numCols, val=1, rowStart=0):
                # Create identity matrix of specified dimension and starting row
                return [[(val if i == j else 0) for j in range(numCols)] for i in range(rowStart, numRows)]
            def standardForm(cost, greaterThans=[], gtThreshold=[], lessThans=[], ltThreshold=[], equalities=[], eqThreshold=[], maximization=True):
                # Process the rest of the constraints and convert the problem to standard form
                # ... [the initial part of the function is assumed to be provided] ...
                # Handle minimization problem by negating the cost function
                if not maximization:
                    cost = [-x for x in cost]
                # ... [the rest of the standardForm function code] ...
                return newCost, constraints, threshold
            def simplex(c, A, b):
                # Implement the simplex algorithm
                # ...
                while canImprove(tableau):
                    # Choose entering variable and check for unboundedness
                    # Implement degeneracy check
                    # Choose leaving variable and pivot
                    # ...
                    for k, row in enumerate(tableau):
                        if k != i:
                            pivotRowMultiple = [y * tableau[k][j] for y in tableau[i]]
                            tableau[k] = [x - y for x, y in zip(tableau[k], pivotRowMultiple)]
                # ...
                return tableau, primal_solution, objective_value

        """,
        "after_lines": """
            # Function to get a specific column from a 2D list (matrix)
            def column(A, j):
                # Finish the the current function and stop generation by the end of the function
                return [row[j] for row in A]

            # Function to transpose a 2D list (matrix)
            def transpose(A):
                # Finish the the current function and stop generation by the end of the function
                return [column(A, j) for j in range(len(A[0]))]

            # Function to check if a column is a pivot column
            def isPivotCol(col):
                # Finish the the current function and stop generation by the end of the function
                return (len([c for c in col if c == 0]) == len(col) - 1) and sum(col) == 1  

            # Function to find the value of a variable for a pivot column
            def variableValueForPivotColumn(tableau, column):
                # Finish the the current function and stop generation by the end of the function
                pivotRow = [i for (i, x) in enumerate(column) if x == 1][0]
                return tableau[pivotRow][-1]

            # Function to check if we can improve the current solution
            def canImprove(tableau):
                # Finish the the current function and stop generation by the end of the function
                lastRow = tableau[-1]
                return any(x > 0 for x in lastRow[:-1])

            # Function to check if there's more than one minimum in a list
            def moreThanOneMin(L):
                if len(L) <= 1:
                    return False
                # Finish the the current function and stop generation by the end of the function
                x, y = heapq.nsmallest(2, L, key=lambda x: x[1])
                return x == y

            # Function to create an identity matrix with certain specifications
            def identity(numRows, numCols, val=1, rowStart=0):
                # Finish the the current function and stop generation by the end of the function
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
                
                ##Given the example of greater than threshold case, complete the case of less than threshold and equal to the threshold.
                if ltThreshold != []:
                    newVars += len(ltThreshold)
                    numRows += len(ltThreshold)
                
                # Equalities don't need slack variables but add to the number of rows
                if eqThreshold != []:
                    numRows += len(eqThreshold)
                
                # If the problem is a minimization, convert it to a maximization by negating the cost vector
                if not maximization:
                    # Finish the current if statement and stop generation by the end of the if statement
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
                
                # Return the new cost function, the modified constraints, and the thresholds
                return newCost, constraints, threshold


            # Main simplex algorithm function
            def simplex(c, A, b):
                # Setup initial tableau from constraints and objective function
                tableau = [row[:] + [x] for row, x in zip(A, b)]
                tableau.append([ci for ci in c] + [0])
                print("Initial tableau:")
                for row in tableau:
                    print(row)
                print()

                # Iterate until no improvements can be made
                while canImprove(tableau):
                    # Choose entering variable (minimum positive index of the last row)
                    column_choices = [(i, x) for (i, x) in enumerate(tableau[-1][:-1]) if x > 0]
                    column = min(column_choices, key=lambda a: a[1])[0]

                    # Check for unboundedness
                    if all(row[column] <= 0 for row in tableau):
                        raise Exception('Linear program is unbounded.')

                    # Implement the code that checks for degeneracy: more than one minimizer of the quotient
                    quotients = [(i, r[-1] / r[column]) for i, r in enumerate(tableau[:-1]) if r[column] > 0]

                    if moreThanOneMin(quotients):
                        raise Exception('Linear program is degenerate.')

                    # Implement the code for chosing leaving variable (row index minimizing the quotient)
                    row = min(quotients, key=lambda x: x[1])[0]

                    # Implement the code that pivots on the chosen row and column
                    pivot = row, column
                    print("Next pivot index is=%d,%d \n" % pivot)
                    i, j = pivot
                    pivotDenom = tableau[i][j]
                    # Normalize the pivot row
                    tableau[i] = [x / pivotDenom for x in tableau[i]]

                    # Zero out the other entries in the pivot column
                    for k, row in enumerate(tableau):
                        # Finish the current for loop block and stop generation by the end of the for loop block
                        if k != i:
                            pivotRowMultiple = [y * tableau[k][j] for y in tableau[i]]
                            tableau[k] = [x - y for x, y in zip(tableau[k], pivotRowMultiple)]
                    print("Tableau after pivot:")
                    for row in tableau:
                        print(row)
                    print()
                    
                # Transpose the tableau to make it easier to work with columns
                columns = transpose(tableau)
                
                # Identify pivot columns in the tableau. A column is a pivot column if it has a single 1 and the rest of its entries are 0.
                indices = [j for j, col in enumerate(columns[:-1]) if isPivotCol(col)]
                
                # For each pivot column, find the value of the corresponding variable.
                # This is done by looking at the rightmost entry (the value part of the tableau row) of the row where the 1 in the pivot column is located.
                primal_solution = [(colIndex, variableValueForPivotColumn(tableau, columns[colIndex])) for colIndex in indices]

                # Calculate the objective value. The last entry of the last row of the tableau gives us the negation of the objective function value.
                objective_value = -(tableau[-1][-1])
                
                return tableau, primal_solution, objective_value
        """
    },
    {
        "description": "Case uses the marker or comment to extract the gen_code",
        "before_lines": """
            import heapq

            # Function to calculate the dot product of two vectors
            def dot(a, b):
                # Finish the the current function and stop generation by the end of the function
                return sum(x*y for x, y in zip(a, b))

            # Function to get a specific column from a 2D list (matrix)
            def column(A, j):
                # Finish the the current function and stop generation by the end of the function
                return [row[j] for row in A]

            # Function to transpose a 2D list (matrix)
            def transpose(A):
                # Finish the the current function and stop generation by the end of the function
                return [column(A, j) for j in range(len(A[0]))]

            # Function to check if a column is a pivot column
            def isPivotCol(col):
                # Finish the the current function and stop generation by the end of the function
                return (len([c for c in col if c == 0]) == len(col) - 1) and sum(col) == 1  

            # Function to find the value of a variable for a pivot column
            def variableValueForPivotColumn(tableau, column):
                # Finish the the current function and stop generation by the end of the function
                pivotRow = [i for (i, x) in enumerate(column) if x == 1][0]
                return tableau[pivotRow][-1]

            # Function to check if we can improve the current solution
            def canImprove(tableau):
                # Finish the the current function and stop generation by the end of the function
                lastRow = tableau[-1]
                return any(x > 0 for x in lastRow[:-1])

            # Function to check if there's more than one minimum in a list
            def moreThanOneMin(L):
                if len(L) <= 1:
                    return False
                # Finish the the current function and stop generation by the end of the function
                x, y = heapq.nsmallest(2, L, key=lambda x: x[1])
                return x == y

            # Function to create an identity matrix with certain specifications
            def identity(numRows, numCols, val=1, rowStart=0):
                # Finish the the current function and stop generation by the end of the function
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
                
                ##Given the example of greater than threshold case, complete the case of less than threshold and equal to the threshold.
                if ltThreshold != []:
                    newVars += len(ltThreshold)
                    numRows += len(ltThreshold)
                
                # Equalities don't need slack variables but add to the number of rows
                if eqThreshold != []:
                    numRows += len(eqThreshold)
                
                # If the problem is a minimization, convert it to a maximization by negating the cost vector
                if not maximization:
                    # Finish the current if statement and stop generation by the end of the if statement
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
                
                # Return the new cost function, the modified constraints, and the thresholds
                return newCost, constraints, threshold


            # Main simplex algorithm function
            def simplex(c, A, b):
                # Setup initial tableau from constraints and objective function
                tableau = [row[:] + [x] for row, x in zip(A, b)]
                tableau.append([ci for ci in c] + [0])
                print("Initial tableau:")
                for row in tableau:
                    print(row)
                print()

                # Iterate until no improvements can be made
                while canImprove(tableau):
                    # Choose entering variable (minimum positive index of the last row)
                    column_choices = [(i, x) for (i, x) in enumerate(tableau[-1][:-1]) if x > 0]
                    column = min(column_choices, key=lambda a: a[1])[0]

                    # Check for unboundedness
                    if all(row[column] <= 0 for row in tableau):
                        raise Exception('Linear program is unbounded.')

                    # Implement the code that checks for degeneracy: more than one minimizer of the quotient
                    quotients = [(i, r[-1] / r[column]) for i, r in enumerate(tableau[:-1]) if r[column] > 0]

                    if moreThanOneMin(quotients):
                        raise Exception('Linear program is degenerate.')

                    # Implement the code for chosing leaving variable (row index minimizing the quotient)
                    row = min(quotients, key=lambda x: x[1])[0]

                    # Implement the code that pivots on the chosen row and column
        """,
        "between_lines": """
                    # Implement the code that pivots on the chosen row and column
                    # --BEGIN MISSING CODE--
                    pivot_value = tableau[row][column]
                    tableau[row] = [val / pivot_value if pivot_value != 0 else 0 for val in tableau[row]]
                    for i in range(len(tableau)):
                        if i != row:
                            ratio = tableau[i][column] / tableau[row][column]
                            tableau[i] = [iv - ratio * rv for iv, rv in zip(tableau[i], tableau[row])]
                    # --END MISSING CODE--
                    # Implement the code that pivots on the chosen row and column
                    pivotDenom = tableau[row][column]
                    if pivotDenom == 0:
                        raise Exception('Pivot element is zero.')
                    tableau[row] = [x / pivotDenom for x in tableau[row]]
                    # Zero out the other entries in the pivot column
                    for k in range(len(tableau)):
                        if k != row:
                            coefficient = tableau[k][column]
                            for j in range(len(tableau[k])):
                                if k != row and j != column:
                                    tableau[k][j] -= coefficient * tableau[row][j]
                                elif j == column:
                                    tableau[k][j] = 0
                    print("Tableau after pivot:")
                    for row in tableau:
                        print(row)
                    print()
                # [... the rest of the simplex function code goes here...]

        """,
        "after_lines": """
                tableau[i] = [x / pivotDenom for x in tableau[i]]

                # Zero out the other entries in the pivot column
                for k, row in enumerate(tableau):
                    # Finish the current for loop block and stop generation by the end of the for loop block
                    if k != i:
                        pivotRowMultiple = [y * tableau[k][j] for y in tableau[i]]
                        tableau[k] = [x - y for x, y in zip(tableau[k], pivotRowMultiple)]
                print("Tableau after pivot:")
                for row in tableau:
                    print(row)
                print()
                
            # Transpose the tableau to make it easier to work with columns
            columns = transpose(tableau)
            
            # Identify pivot columns in the tableau. A column is a pivot column if it has a single 1 and the rest of its entries are 0.
            indices = [j for j, col in enumerate(columns[:-1]) if isPivotCol(col)]
            
            # For each pivot column, find the value of the corresponding variable.
            # This is done by looking at the rightmost entry (the value part of the tableau row) of the row where the 1 in the pivot column is located.
            primal_solution = [(colIndex, variableValueForPivotColumn(tableau, columns[colIndex])) for colIndex in indices]

            # Calculate the objective value. The last entry of the last row of the tableau gives us the negation of the objective function value.
            objective_value = -(tableau[-1][-1])
            
            return tableau, primal_solution, objective_value
        """
    },
]

Java_test_clean_cases = [
    {
        "description": "Case with `firstOne = false;",
        "before_lines": """
        """,
        "between_lines": """
            In the provided code for the `PrimeFactorizer` class, there is a section of missing code within the `printPrimeFactorization` method where a flag `firstOne` is used to indicate that the first prime factor is being printed. We need to insert the necessary code to correctly print the first prime factor without a preceding "x" multiplication symbol. Since this is the first factor, we just need to print it directly to the `resultStream`. The following code should be included in the `# --BEGIN MISSING CODE--` section:

            ```java
            resultStream.format ("%d", allPrimes[curPosInAllPrimes]);
            firstOne = False;
            ```

            The completed section of the `printPrimeFactorization` method should look like this:

            ```java
            // if it's the first one, we don't need to print a "x"
            if (firstOne) {
            resultStream.format ("%d", allPrimes[curPosInAllPrimes]);
            firstOne = false;
            // otherwise, print the factor pre-pended with an "x"
            } else {
            resultStream.format (" x %d", allPrimes[curPosInAllPrimes]);
            }
            ```
        """,
        "after_lines": """
        """,
    },
]
def test_post_process(test_cases, program_type, clean_gen_code = True, remove_gen_repeated_lines=True, add_auto_indent=True):
    # Iterate through the test cases, clean the code, and print the results
    for i, test_case in enumerate(test_cases, start=1):
        print(f"Test Case {i}: {test_case['description']}\n")
        gen_code = test_case['between_lines']
        before_lines = test_case['before_lines']
        after_lines = test_case['after_lines']
        print(f"before_lines:\n{before_lines}\n")
        print('-' * 80 + '\n')
        print(f"Original gen Code:\n{gen_code}\n")
        print('-' * 80 + '\n')
        # import pdb; pdb.set_trace()
        if clean_gen_code:
            gen_code = clean_code(gen_code, program_type)
            print(f"Cleaned Code:\n{gen_code}\n")
            import pdb;pdb.set_trace()
        if remove_gen_repeated_lines:
            # Remove any repeated lines from 'before_lines' and 'after_lines'
            # gen_code = remove_repeated_lines(gen_code, before_lines, after_lines)
            before_lines_list = before_lines.strip().split('\n')
            after_lines_list = after_lines.strip().split('\n')
            gen_code = trim_similar_edges(gen_code, before_lines_list, after_lines_list, program_type)
            print(f"Remove duplicated code:\n{gen_code}\n")
            import pdb;pdb.set_trace()
        # Auto-indent code (specifically for Python)
        if add_auto_indent:
            tab_indent = 4
            if program_type == "Java":
                indented_code = auto_bracket_matcher(before_lines, gen_code, after_lines)
                import pdb;pdb.set_trace()
            elif program_type == "Python":
                indented_code = auto_indent(before_lines, gen_code, after_lines)
            else:
                raise ValueError("Auto-indentation is only supported for Python and Java programs.")
            gen_code = indented_code  # Update gen_code to the indented version
            print('-' * 80)
            print("Auto Indented Code:\n")
            print(indented_code.replace('\t', ' ' * tab_indent))

        import pdb; pdb.set_trace()
        print('-' * 80 + '\n')

test_cases_remove_duplicates_on_java = [
    {
        "description": "Case incorrectly tracking the comments at the begining of the after_lines or ending of before_lines",
        "before_lines": """
        import junit.framework.TestCase;
        import java.io.*;
        import java.util.HashMap;
        import java.util.TreeMap;
        import java.util.ArrayList;

        /**
         * This silly little class wraps String, int pairs so they can be sorted.
         * Used by the WordCounter class.
         */
        class WordWrapper implements Comparable <WordWrapper> {

             private String myWord;
             private int myCount;
             private int mySlot;

             /**
                * Creates a new WordWrapper with string "XXX" and count zero.
                */
             public WordWrapper (String myWordIn, int myCountIn, int mySlotIn) {
                 myWord = myWordIn;
                 myCount = myCountIn;
                 mySlot = mySlotIn;
             }

             /**
                * Returns a copy of the string in the WordWrapper.
                *
                * @return    The word contained in this object.  Is copied out: no aliasing.
                */
             public String extractWord () {
                 return new String (myWord);
             }

             /**
                * Returns the count in this WordWrapper.
                *
                * @return    The count contained in the object.
                */
             public int extractCount () {
                 return myCount;
             }

             public void incCount () {
                 myCount++;
             }

             public int getCount () {
                 return myCount;
             }

             public int getPos () {
                 return mySlot;
             }

             public void setPos (int toWhat) {
                     mySlot = toWhat;
             }

             public String toString () {
                 return myWord + " " + mySlot;
             }

             /**
                * Comparison operator so we implement the Comparable interface.
                *
                * @param  w   The word wrapper to compare to
                * @return     a positive, zero, or negative value depending upon whether this
                *             word wrapper is less then, equal to, or greater than param w.  The
                *             relationship is determied by looking at the counts.
                */
             public int compareTo (WordWrapper w) {
                 // Check if the count of the current word (myCount) is equal to the count of the word 'w'
                 // If the counts are equal, compare the words lexicographically
                 if (w.myCount == myCount) {
                     return myWord.compareTo(w.myWord);
                 }
                 // If the counts are not equal, subtract the count of the current word from the count of the word 'w'

        """,
        "between_lines": """
         return myCount - w.myCount;
           }
         }
        """,
        "after_lines": """
                }
            }


            /**
            * This class is used to count occurences of words (strings) in a corpus.  Used by
            * the IndexedDocumentCollection class to find the most frequent words in the corpus.
            */

            class WordCounter {
                
            // this is the map that holds (for each word) the number of times we've seen it
            // note that Integer is a wrapper for the buitin Java int type
            private HashMap <String, WordWrapper> wordsIHaveSeen = new HashMap <String, WordWrapper> ();
            private TreeMap <String, String> ones = new TreeMap <String, String> ();
            
            // the set of words that have been extracted
            private ArrayList<WordWrapper> extractedWords = new ArrayList <WordWrapper> ();

            /**
            * Accepts a word and increments the count of the number of times we have seen it.
            * Note that a deep copy of the param is done (no aliasing).
            *
            * @param addMe   The string to count
            */
            public void insert (String addMe) {
                if (ones.containsKey (addMe)) {
                ones.remove (addMe);
                WordWrapper temp = new WordWrapper (addMe, 1, extractedWords.size ());
                wordsIHaveSeen.put(addMe, temp);
                extractedWords.add (temp);
                }

                // first check to see if the word is alredy in wordsIHaveSeen
                if (wordsIHaveSeen.containsKey (addMe)) {
                // if it is, then remove and increment its count
                WordWrapper temp = wordsIHaveSeen.get (addMe);
                temp.incCount ();

                // find the slot that we go to in the extractedWords list
                // by sorting the 'extractedWords' list in ascending order
                for (int i = temp.getPos () - 1; i >= 0 && 
                    extractedWords.get (i).compareTo (extractedWords.get (i + 1)) > 0; i--) {
                    temp = extractedWords.get (i + 1);
                    temp.setPos (i);
                    
                    WordWrapper temp2 = extractedWords.get (i);
                    temp2.setPos (i + 1);
                    
                    extractedWords.set (i + 1, temp2);
                    extractedWords.set (i, temp);
                }

                // in this case it is not there, so just add it
                } else {
                ones.put (addMe, addMe);
                }
            }

            /**
            * Returns the kth most frequent word in the corpus so far.  A deep copy is returned,
            * so no aliasing is possible.  Returns null if there are not enough words.
            *
            * @param k     Note that the most frequent word is at k = 0
            * @return      The kth most frequent word in the corpus to date
            */
            public String getKthMostFrequent (int k) {

                // if we got here, then the array is all set up, so just return the kth most frequent word
                if (k >= extractedWords.size ()) {
                int which = extractedWords.size ();
                for (String s : ones.navigableKeySet ()) {
                    if (which == k)
                    return s;
                    which++;
                }
                return null;
                } else {
                // If k is less than the size of the extractedWords list,
                // return the kth most frequent word from extractedWords
                return extractedWords.get (k).extractWord ();
                }
            }
            }
            /**
            * A JUnit test case class.
            * Every method starting with the word "test" will be called when running
            * the test with JUnit.
            */
            public class CounterTester extends TestCase {

            /**
            * File object to read words from, with buffering.
            */
            private FileReader     file = null;
            private BufferedReader reader = null;

            /**
            * Close file readers.
            */
            private void closeFiles() {
                try {
                if (file != null) {
                    file.close();
                    file = null;
                }
                if (reader != null) {
                    reader.close();
                    reader = null;
                }
                } catch (Exception e) {
                System.err.println("Problem closing file");
                System.err.println(e);
                e.printStackTrace();
                }
            }

            /**
            * Close files no matter what happens in the test.
            */
            protected void tearDown () {
                closeFiles();
            }

            /**
            * Open file and set up readers.
            *
            * @param fileName  name of file to open
            *   */
            private void openFile(String fileName) {
                try {
                file = new FileReader(fileName);
                reader = new BufferedReader(file);
                } catch (Exception e) {
                System.err.format("Problem opening %s file\n", fileName);
                System.err.println(e);
                e.printStackTrace();
                fail();
                }
            }

            /**
            * Read the next numWords from the file.
            *
            * @param counter   word counter to update
            * @param numWords  number of words to read.
            */
            private void readWords(WordCounter counter, int numWords) {
                try {
                for (int i=0; i<numWords; i++) {
                    if (i % 100000 == 0)
                    System.out.print (".");
                    String word = reader.readLine();
                    if (word == null) {
                    return;
                    }
                    // Insert 'word' into the counter.
                    counter.insert(word);
                }
                } catch (Exception e) {
                System.err.println("Problem reading file");
                System.err.println(e);
                e.printStackTrace();
                fail();
                }
            }

            /**
            * Read the next numWords from the file, mixing with queries
            *
            * @param counter   word counter to update
            * @param numWords  number of words to read.
            */
            private void readWordsMixed (WordCounter counter, int numWords) {
                try {
                int j = 0;
                for (int i=0; i<numWords; i++) {
                    String word = reader.readLine();
                    // If 'word' is null, it means we've reached the end of the file. So, return from the method.
                    if (word == null) {
                    return;
                    }
                    counter.insert(word);

                    // If the current iteration number is a multiple of 10 and greater than 100,000, perform a query of the kth most frequent word.
                    if (i % 10 == 0 && i > 100000) {
                    String myStr = counter.getKthMostFrequent(j++);
                    }

                    // rest j once we get to 100
                    if (j == 100)
                    j = 0;
                }
                } catch (Exception e) {
                System.err.println("Problem reading file");
                System.err.println(e);
                e.printStackTrace();
                fail();
                }
            }

            /**
            * Check that a sequence of words starts at the "start"th most
            * frequent word.
            *
            * @param counter   word counter to lookup
            * @param start     frequency index to start checking at
            * @param expected  array of expected words that start at that frequency
            */
            private void checkExpected(WordCounter counter, int start, String [] expected) {
                for (int i = 0; i<expected.length; i++) {
                String actual = counter.getKthMostFrequent(start);
                System.out.format("k: %d, expected: %s, actual: %s\n",
                                    start, expected[i], actual);
                assertEquals(expected[i], actual);
                start++;
                }
            }

            /**
            * A test method.
            * (Replace "X" with a name describing the test.  You may write as
            * many "testSomething" methods in this class as you wish, and each
            * one will be called when running JUnit over this class.)
            */
            public void testSimple() {
                System.out.println("\nChecking insert");
                WordCounter counter = new WordCounter();
                counter.insert("pizzaz");
                // Insert the word "pizza" into the counter twice
                counter.insert("pizza");
                counter.insert("pizza");
                String [] expected = {"pizza"};
                checkExpected(counter, 0, expected);
            }

            public void testTie() {
                System.out.println("\nChecking tie for 2nd place");
                WordCounter counter = new WordCounter();
                counter.insert("panache");
                counter.insert("pizzaz");
                counter.insert("pizza");
                counter.insert("zebra");
                counter.insert("pizza");
                counter.insert("lion");
                counter.insert("pizzaz");
                counter.insert("panache");
                counter.insert("panache");
                counter.insert("camel");
                // order is important here
                String [] expected = {"pizza", "pizzaz"};
                checkExpected(counter, 1, expected);
            }


            public void testPastTheEnd() {
                System.out.println("\nChecking past the end");
                WordCounter counter = new WordCounter();
                counter.insert("hi");
                counter.insert("hello");
                counter.insert("greetings");
                counter.insert("salutations");
                counter.insert("hi");
                counter.insert("welcome");
                counter.insert("goodbye");
                counter.insert("later");
                counter.insert("hello");
                counter.insert("when");
                counter.insert("hi");
                assertNull(counter.getKthMostFrequent(8));
            }

            public void test100Top5() {
                System.out.println("\nChecking top 5 of 100 words");
                WordCounter counter = new WordCounter();
                // Open the file "allWordsBig"
                openFile("allWordsBig");
                // Read the next 100 words
                readWords(counter, 100);
                String [] expected = {"edu", "comp", "cs", "windows", "cmu"};
                checkExpected(counter, 0, expected);
            }

            public void test300Top5() {
                System.out.println("\nChecking top 5 of first 100 words");
                WordCounter counter = new WordCounter();
                openFile("allWordsBig");
                readWords(counter, 100);
                String [] expected1 = {"edu", "comp", "cs", "windows", "cmu"};
                checkExpected(counter, 0, expected1);

                System.out.println("Adding 100 more words and rechecking top 5");
                readWords(counter, 100);
                String [] expected2 = {"edu", "cmu", "comp", "cs", "state"};
                checkExpected(counter, 0, expected2);

                System.out.println("Adding 100 more words and rechecking top 5");
                readWords(counter, 100);
                String [] expected3 = {"edu", "cmu", "comp", "ohio", "state"};
                checkExpected(counter, 0, expected3);
            }

            public void test300Words14Thru19() {
                System.out.println("\nChecking rank 14 through 19 of 300 words");
                WordCounter counter = new WordCounter();
                openFile("allWordsBig");
                // Read the next 300 words
                readWords(counter, 300);
                String [] expected = {"cantaloupe", "from", "ksu", "okstate", "on", "srv"};
                checkExpected(counter, 14, expected);
            }

            public void test300CorrectNumber() {
                System.out.println("\nChecking correct number of unique words in 300 words");
                WordCounter counter = new WordCounter();
                openFile("allWordsBig");
                readWords(counter, 300);
            /* check the 122th, 123th, and 124th most frequent word in the corpus so far 
                * and check if the result is not null.
                */
                assertNotNull(counter.getKthMostFrequent(122));
                assertNotNull(counter.getKthMostFrequent(123));
                assertNotNull(counter.getKthMostFrequent(124));
                // check the 125th most frequent word in the corpus so far
                // and check if the result is null.
                assertNull(counter.getKthMostFrequent(125));
            }

            public void test300and100() {
                System.out.println("\nChecking top 5 of 100 and 300 words with two counters");
                WordCounter counter1 = new WordCounter();
                openFile("allWordsBig");
                readWords(counter1, 300);
                closeFiles();

                WordCounter counter2 = new WordCounter();
                openFile("allWordsBig");
                readWords(counter2, 100);

                String [] expected1 = {"edu", "cmu", "comp", "ohio", "state"};
                checkExpected(counter1, 0, expected1);

                String [] expected2 = {"edu", "comp", "cs", "windows", "cmu"};
                checkExpected(counter2, 0, expected2);
            }

            public void testAllTop15() {
                System.out.println("\nChecking top 15 of all words");
                WordCounter counter = new WordCounter();
                openFile("allWordsBig");
                // Read the next 6000000 words
                readWords(counter, 6000000);
                String [] expected = {"the", "edu", "to", "of", "and",
                                    "in", "is", "ax", "that", "it",
                                    "cmu", "for", "com", "you", "cs"};
                checkExpected(counter, 0, expected);
            }

            public void testAllTop10000() {
                System.out.println("\nChecking time to get top 10000 of all words");
                WordCounter counter = new WordCounter();
                openFile("allWordsBig");
                readWords(counter, 6000000);

                for (int i=0; i<10000; i++) {
                counter.getKthMostFrequent(i);
                }
            }

            public void testSpeed() {
                System.out.println("\nMixing adding data with finding top k");
                WordCounter counter = new WordCounter();
                openFile("allWordsBig");
                readWordsMixed (counter, 6000000);
                String [] expected = {"the", "edu", "to", "of", "and",
                                    "in", "is", "ax", "that", "it",
                                    "cmu", "for", "com", "you", "cs"};
                checkExpected(counter, 0, expected);
            }

            }
        """
    },
    {
        "description": "Case incorrectly tracking the comments at the begining of the after_lines or ending of before_lines",
        "before_lines": """
        import junit.framework.TestCase;
        import java.io.ByteArrayOutputStream;
        import java.io.PrintStream;
        import java.lang.Math;

        /**
        * This class implements a relatively simple algorithm for computing
        * (and printing) the prime factors of a number.  At initialization,
        * a list of primes is computed.  Given a number, this list is then
        * used to efficiently print the prime factors of the number.
        */
        class PrimeFactorizer {

        // this class will store all primes between 2 and upperBound
        private int upperBound;
        
        // this class will be able to factorize all primes between 1 and maxNumberToFactorize
        private long maxNumberToFactorize;
        
        // this is a list of all of the prime numbers between 2 and upperBound
        private int [] allPrimes;
        
        // this is the number of primes found between 2 and upperBound
        private int numPrimes;
        
        // this is the output stream that the factorization will be printed to
        private PrintStream resultStream;
        
        /**
        * This prints the prime factorization of a number in a "pretty" fashion.
        * If the number passed in exceeds "upperBound", then a nice error message
        * is printed.  
        */
        public void printPrimeFactorization (long numToFactorize) {
            
            // If we are given a number that's too small/big to factor, tell the user
            if (numToFactorize < 1) {
            resultStream.format ("Can't factorize a number less than 1");
            return;
            }
            if (numToFactorize > maxNumberToFactorize) {
            resultStream.format ("%d is too large to factorize", numToFactorize);
            return;
            }
            
            // Now get ready to print the factorization
            resultStream.format ("Prime factorization of %d is: ", numToFactorize);
        
            // our basic tactice will be to find all of the primes that divide evenly into
            // our target.  When we find one, print it out and reduce the target accrordingly.
            // When the target gets down to one, we are done.
            
            // this is the index of the next prime we need to try
            int curPosInAllPrimes = 0;
            
            // tells us we are still waiting to print the first prime... important so we
            // don't print the "x" symbol before the first one
            boolean firstOne = true;
            
            while (numToFactorize > 1 && curPosInAllPrimes < numPrimes) {
            
            // if the current prime divides evenly into the target, we've got a prime factor
            if (numToFactorize % allPrimes[curPosInAllPrimes] == 0) {
                
                // if it's the first one, we don't need to print a "x"
                // print the factor & reset the firstOne flag
                if (firstOne) {
        """,
        "between_lines": """
        // --BEGIN MISSING CODE--
            resultStream.format ("%d", allPrimes[curPosInAllPrimes]);
            firstOne = false;
            }

            // remove that prime factor from the target
            numToFactorize /= allPrimes[curPosInAllPrimes];

        // if the current prime does not divide evenly, try the next one
        } else {
            curPosInAllPrimes++;
        }
        }

        // if we never printed any factors, then display the number itself
        if (firstOne) {
        resultStream.format ("%d", numToFactorize);
        // Otherwise print the factors connected by 'x'
        } else if (numToFactorize > 1) {
        resultStream.format (" x %d", num
        """,
        "after_lines": """
                // otherwise, print the factor pre-pended with an "x"
                } else {
                resultStream.format (" x %d", allPrimes[curPosInAllPrimes]);
                }
                
                // remove that prime factor from the target
                numToFactorize /= allPrimes[curPosInAllPrimes];
                
            // if the current prime does not divde evenly, try the next one
            } else {
                curPosInAllPrimes++;
            }
            }
            
            // if we never printed any factors, then display the number itself
            if (firstOne) {
            resultStream.format ("%d", numToFactorize);
            // Otherwsie print the factors connected by 'x'
            } else if (numToFactorize > 1) {
            resultStream.format (" x %d", numToFactorize);
            }
        }
        
        
        /**
        * This is the constructor.  What it does is to fill the array allPrimes with all
        * of the prime numbers from 2 through sqrt(maxNumberToFactorize).  This array of primes
        * can then subsequently be used by the printPrimeFactorization method to actually
        * compute the prime factorization of a number.  The method also sets upperBound
        * (the largest number we checked for primality) and numPrimes (the number of primes
        * in allPimes).  The algorithm used to compute the primes is the classic "sieve of 
        * Eratosthenes" algorithm.
        */
        public PrimeFactorizer (long maxNumberToFactorizeIn, PrintStream outputStream) {
            
            resultStream = outputStream;
            maxNumberToFactorize = maxNumberToFactorizeIn;
            
            // initialize the two class variables- 'upperBound' and 'allPrimes' note that the prime candidates are
            // from 2 to upperBound, so there are upperBound prime candidates intially
            upperBound = (int) Math.ceil (Math.sqrt (maxNumberToFactorize));
            allPrimes = new int[upperBound - 1];
            
            // numIntsInList is the number of ints currently in the list.
            int numIntsInList = upperBound - 1;
            
            // the number of primes so far is zero
            numPrimes = 0;
            
            // write all of the candidate numbers to the list... this is all of the numbers
            // from two to upperBound
            for (int i = 0; i < upperBound - 1; i++) {
            allPrimes[i] = i + 2;
            }
            
            // now we keep removing numbers from the list until we have only primes
            while (numIntsInList > numPrimes) {
            
            // curPos tells us the last slot that has a "good" (still possibly prime) value.
            int curPos = numPrimes + 1;
            
            // the front of the list is a prime... kill everyone who is a multiple of it
            for (int i = numPrimes + 1; i < numIntsInList; i++) {
                
                // if the dude at position i is not a multiple of the current prime, then
                // we keep him; otherwise, he'll be lost
                if (allPrimes[i] % allPrimes[numPrimes] != 0) {
                allPrimes[curPos] = allPrimes[i];
                curPos++;
                }
            }
            
            // the number of ints in the list is now equal to the last slot we wrote a value to
            numIntsInList = curPos;
            
            // and the guy at the front of the list is now considered a prime
            numPrimes++;
            
            }
        }
        

        }

        /**
        * A JUnit test case class.
        * Every method starting with the word "test" will be called when running
        * the test with JUnit.
        */
        
        public class FactorizationTester extends TestCase {
            /**
        * Stream to hold the result of each test.
        */
        private ByteArrayOutputStream result;

        /**
        * Wrapper to provide printing functionality.
        */
        private PrintStream outputStream;

        /**
        * Setup the output streams before each test.
        */   
        protected void setUp () {
            result = new ByteArrayOutputStream();
            outputStream = new PrintStream(result);
        }    
        
        /**
        * Print a nice message about each test.
        */
        private void printInfo(String description, String expected, String actual) {
            System.out.format ("\n%s\n", description);
            System.out.format ("output:\t%s\n", actual);
            System.out.format ("correct:\t%s\n", expected);
        }

        /**
        * These are the various test methods.  The first 10 test factorizations
        * using the object constructed to factorize numbers up to 100; the latter
        * 7 use the object to factorize numbers up to 100000.
        */
        public void test100MaxFactorize1() {
            // Factorize the number
            PrimeFactorizer myFactorizerMax100 = new PrimeFactorizer(100, outputStream);
            myFactorizerMax100.printPrimeFactorization (1);

            // Print Results
            String expected = new String("Prime factorization of 1 is: 1");
            printInfo("Factorizing 1 using max 100", expected, result.toString());

            // Check if test passed by comparing the expected and actual results
            assertEquals (expected, result.toString());
        }

        // Test the factorization of 7 
        public void test100MaxFactorize7() {
            // Factorize the number
            PrimeFactorizer myFactorizerMax100 = new PrimeFactorizer(100, outputStream);
            myFactorizerMax100.printPrimeFactorization (7);

            // Print Results
            String expected = new String("Prime factorization of 7 is: 7");
            printInfo("Factorizing 7 using max 100", expected, result.toString());

            // Check if test passed
            assertEquals (expected, result.toString());
        }

        // Test the factorization of 5
        public void test100MaxFactorize5() {
            // Factorize the number
            PrimeFactorizer myFactorizerMax100 = new PrimeFactorizer(100, outputStream);
            myFactorizerMax100.printPrimeFactorization (5);

            // Print Results
            String expected = new String("Prime factorization of 5 is: 5");
            printInfo("Factorizing 5 using max 100", expected, result.toString());

            // Check if test passed
            assertEquals (expected, result.toString());
        }
        
        // Test the factorization of 30
        public void test100MaxFactorize30() {
            // Factorize the number
            PrimeFactorizer myFactorizerMax100 = new PrimeFactorizer(100, outputStream);
            myFactorizerMax100.printPrimeFactorization (30);

            // Print Results
            String expected = new String("Prime factorization of 30 is: 2 x 3 x 5");
            printInfo("Factorizing 30 using max 100", expected, result.toString());

            // Check if test passed
            assertEquals (expected, result.toString());
        }

        // Test the factorization of 81
        public void test100MaxFactorize81() {
            // Factorize the number
            PrimeFactorizer myFactorizerMax100 = new PrimeFactorizer(100, outputStream);
            myFactorizerMax100.printPrimeFactorization (81);

            // Print Results
            String expected = new String("Prime factorization of 81 is: 3 x 3 x 3 x 3");
            printInfo("Factorizing 81 using max 100", expected, result.toString());

            // Check if test passed
            assertEquals (expected, result.toString());
        }
        
        // Test the factorization of 71
        public void test100MaxFactorize71() {
            // Factorize the number
            PrimeFactorizer myFactorizerMax100 = new PrimeFactorizer(100, outputStream);
            myFactorizerMax100.printPrimeFactorization (71);

            // Print Results
            String expected = new String("Prime factorization of 71 is: 71");
            printInfo("Factorizing 71 using max 100", expected, result.toString());

            // Check if test passed
            assertEquals (expected, result.toString());
        }
        
        // Test the factorization of 100
        public void test100MaxFactorize100() {
            // Factorize the number
            PrimeFactorizer myFactorizerMax100 = new PrimeFactorizer(100, outputStream);
            myFactorizerMax100.printPrimeFactorization (100);

            // Print Results
            String expected = new String("Prime factorization of 100 is: 2 x 2 x 5 x 5");
            printInfo("Factorizing 100 using max 100", expected, result.toString());

            // Check if test passed
            assertEquals (expected, result.toString());
        }
        
        // Test the factorization of 101
        public void test100MaxFactorize101() {
            // Factorize the number
            PrimeFactorizer myFactorizerMax100 = new PrimeFactorizer(100, outputStream);
            myFactorizerMax100.printPrimeFactorization (101);

            // Print Results
            String expected = new String("101 is too large to factorize");
            printInfo("Factorizing 101 using max 100", expected, result.toString());

            // Check if test passed
            assertEquals (expected, result.toString());
        }
        
        // Test the factorization of 0
        public void test100MaxFactorize0() {
            // Factorize the number
            PrimeFactorizer myFactorizerMax100 = new PrimeFactorizer(100, outputStream);
            myFactorizerMax100.printPrimeFactorization (0);

            // Print Results
            String expected = new String("Can't factorize a number less than 1");
            printInfo("Factorizing 0 using max 100", expected, result.toString());

            // Check if test passed
            assertEquals (expected, result.toString());
        }
        // Test the factorization of 97
        public void test100MaxFactorize97() {
            // Factorize the number
            PrimeFactorizer myFactorizerMax100 = new PrimeFactorizer(100, outputStream);
            myFactorizerMax100.printPrimeFactorization (97);

            // Print Results
            String expected = new String("Prime factorization of 97 is: 97");
            printInfo("Factorizing 97 using max 100", expected, result.toString());

            // Check if test passed
            assertEquals (expected, result.toString());
        }

        // Test the factorization of 34534
        // factorize numbers up to 100000000
        public void test100000000MaxFactorize34534() {
            // Factorize the number
            PrimeFactorizer myFactorizerMax100000000 = new PrimeFactorizer(100000000, outputStream);
            myFactorizerMax100000000.printPrimeFactorization (34534);

            // Print Results
            String expected = new String("Prime factorization of 34534 is: 2 x 31 x 557");
            printInfo("Factorizing 34534 using max 100000000", expected, result.toString());

            // Check if test passed
            assertEquals (expected, result.toString());
        }

        // Test the factorization of 4339
        // factorize numbers up to 100000000
        public void test100000000MaxFactorize4339() {
            // Factorize the number
            PrimeFactorizer myFactorizerMax100000000 = new PrimeFactorizer(100000000, outputStream);
            myFactorizerMax100000000.printPrimeFactorization (4339);

            // Print Results
            String expected = new String("Prime factorization of 4339 is: 4339");
            printInfo("Factorizing 4339 using max 100000000", expected, result.toString());

            // Check if test passed
            assertEquals (expected, result.toString());
        }

        // Test the factorization of 65536
        // factorize numbers up to 100000000
        public void test100000000MaxFactorize65536() {
            // Factorize the number
            PrimeFactorizer myFactorizerMax100000000 = new PrimeFactorizer(100000000, outputStream);
            myFactorizerMax100000000.printPrimeFactorization (65536);

            // Print Results
            String expected = new String("Prime factorization of 65536 is: 2 x 2 x 2 x 2 x 2 x 2 x 2 x 2 x 2 x 2 x 2 x 2 x 2 x 2 x 2 x 2");
            printInfo("Factorizing 65536 using max 100000000", expected, result.toString());

            // Check if test passed
            assertEquals (expected, result.toString());
        }
        
        // Test the factorization of 99797
        // factorize numbers up to 100000000
        public void test100000000MaxFactorize99797() {
            // Factorize the number
            PrimeFactorizer myFactorizerMax100000000 = new PrimeFactorizer(100000000, outputStream);
            myFactorizerMax100000000.printPrimeFactorization (99797);

            // Print Results
            String expected = new String("Prime factorization of 99797 is: 23 x 4339");
            printInfo("Factorizing 99797 using max 100000000", expected, result.toString());

            // Check if test passed
            assertEquals (expected, result.toString());
        }
        
        // Test the factorization of 307
        // factorize numbers up to 100000000
        public void test100000000MaxFactorize307() {
            // Factorize the number
            PrimeFactorizer myFactorizerMax100000000 = new PrimeFactorizer(100000000, outputStream);
            myFactorizerMax100000000.printPrimeFactorization (307);

            // Print Results
            String expected = new String("Prime factorization of 307 is: 307");
            printInfo("Factorizing 307 using max 100000000", expected, result.toString());

            // Check if test passed
            assertEquals (expected, result.toString());
        }

        // Test the factorization of 38845248344
        // factorize numbers up to 100000000000
        public void test100000000000MaxFactorize38845248344() {
            // Factorize the number
            PrimeFactorizer myFactorizerMax100000000000 = new PrimeFactorizer(100000000000L, outputStream);
            myFactorizerMax100000000000.printPrimeFactorization (38845248344L);

            // Print Results
            String expected = new String("Prime factorization of 38845248344 is: 2 x 2 x 2 x 7 x 693665149");
            printInfo("Factorizing 38845248344 using max 100000000000", expected, result.toString());
            
            // Check if test passed
            assertEquals (expected, result.toString());
            
            result.reset ();
            myFactorizerMax100000000000.printPrimeFactorization (24210833220L);
            expected = new String("Prime factorization of 24210833220 is: 2 x 2 x 3 x 3 x 5 x 7 x 17 x 19 x 19 x 31 x 101");
            printInfo("Factorizing 24210833220 using max 100000000000", expected, result.toString());

            // Check if test passed
            assertEquals (expected, result.toString());
        }
        
        public void test100000000000MaxFactorize5387457483444() {
            // Factorize the number
            PrimeFactorizer myFactorizerMax100000000000 = new PrimeFactorizer(100000000000L, outputStream);
            myFactorizerMax100000000000.printPrimeFactorization (5387457483444L);

            // Print Results
            String expected = new String("5387457483444 is too large to factorize");
            printInfo("Factorizing 5387457483444 using max 100000000000", expected, result.toString());

            // Check if test passed
            assertEquals (expected, result.toString());
        }
        
        public void testSpeed() {
            
            // here we factorize 10,000 large numbers; make sure that the constructor correctly sets
            // up the array of primes just once... if it does not, it'll take too long to run
            PrimeFactorizer myFactorizerMax100000000000 = new PrimeFactorizer(100000000000L, outputStream);
            for (long i = 38845248344L; i < 38845248344L + 50000; i++) {
            myFactorizerMax100000000000.printPrimeFactorization (i);
            if (i == 38845248344L) {
                // Compare the Prime factorization of 38845248344 to the expected result
                String expected = new String("Prime factorization of 38845248344 is: 2 x 2 x 2 x 7 x 693665149"); 
                assertEquals (expected, result.toString());
            }
            }
            
            // if we make it here, we pass the test
            assertTrue (true);
        }
        
        }
        """
    }



]
# test_post_process(test_clean_cases, 'Python', clean_gen_code =True, remove_gen_repeated_lines=False, add_auto_indent=False)
# test_post_process(Java_test_clean_cases, 'Java', clean_gen_code =True, remove_gen_repeated_lines=False, add_auto_indent=False)

# test_post_process(test_cases_remove_duplicates_trace_non_comments, 'Python', clean_gen_code =True, remove_gen_repeated_lines=True, add_auto_indent=True)
# test_post_process(test_cases_remove_duplicates, 'Python', clean_gen_code =True, remove_gen_repeated_lines=True, add_auto_indent=True)

test_post_process(test_cases_remove_duplicates_on_java, 'Java', clean_gen_code =True, remove_gen_repeated_lines=True, add_auto_indent=True)

# import parso

# def auto_bracket_matcher(before_lines, between_lines, after_lines):
#     # Concatenate the lines
#     total_lines = before_lines + between_lines + after_lines

#     # Parse the code
#     module = parso.parse(total_lines)

#     # Count the number of open and close brackets
#     open_brackets = total_lines.count('{')
#     close_brackets = total_lines.count('}')

#     # Add or remove brackets from between_lines as necessary
#     if open_brackets > close_brackets:
#         between_lines += '}' * (open_brackets - close_brackets)
#     elif open_brackets < close_brackets:
#         between_lines = between_lines.rsplit('}', close_brackets - open_brackets)[0]

#     return between_lines

# def auto_bracket_matcher(before_lines, between_lines, after_lines):
#     # Define the pairs of brackets
#     brackets = {'{': '}', '[': ']', '(': ')'}

#     # Concatenate the lines
#     total_lines = before_lines + between_lines + after_lines

#     # Initialize a dictionary to count the brackets
#     bracket_counts = {bracket: [0, 0] for bracket in brackets}

#     # Count the number of open and close brackets
#     for char in total_lines:
#         if char in brackets:
#             bracket_counts[char][0] += 1
#         elif char in brackets.values():
#             for bracket, close_bracket in brackets.items():
#                 if char == close_bracket:
#                     bracket_counts[bracket][1] += 1

#     # Add or remove brackets from between_lines as necessary
#     for bracket, counts in bracket_counts.items():
#         open_brackets, close_brackets = counts
#         if open_brackets > close_brackets:
#             between_lines += brackets[bracket] * (open_brackets - close_brackets)
#         elif open_brackets < close_brackets:
#             between_lines = between_lines.rsplit(brackets[bracket], close_brackets - open_brackets)[0]

#     return between_lines



def normalize_string(s):
    return '\n'.join(line.strip() for line in s.split('\n'))

# def auto_bracket_matcher(before_lines, between_lines, after_lines):
#     # Concatenate the lines
#     total_lines = before_lines + between_lines + after_lines

#     # Use a stack to keep track of the brackets
#     stack = []

#     # Process each character in the code
#     for char in total_lines:
#         if char == '{':
#             stack.append(char)
#         elif char == '}':
#             if stack:
#                 stack.pop()
#             else:
#                 # There's an extra closing bracket, remove it from between_lines
#                 between_lines = between_lines.rsplit('}', 1)[0]

#     # If there are extra opening brackets, add closing brackets to between_lines
#     between_lines += '}' * len(stack)

#     return between_lines

# def auto_bracket_matcher(before_lines, between_lines, after_lines):
#     # Concatenate the lines
#     total_lines = before_lines + between_lines + after_lines

#     # Parse the code
#     module = parso.parse(total_lines)

#     # Count the number of open and close brackets
#     open_brackets = total_lines.count('{')
#     close_brackets = total_lines.count('}')

#     # Add or remove brackets from between_lines as necessary
#     if open_brackets > close_brackets:
#         between_lines += '}' * (open_brackets - close_brackets)
#     elif open_brackets < close_brackets:
#         between_lines = between_lines.rsplit('}', close_brackets - open_brackets)[0]

#     # Count the number of open and close brackets in before_lines and after_lines
#     open_brackets_before_after = before_lines.count('{') + after_lines.count('{')
#     close_brackets_before_after = before_lines.count('}') + after_lines.count('}')

#     # Remove extra opening brackets from between_lines
#     if open_brackets_before_after < close_brackets_before_after:
#         between_lines = between_lines.split('{', close_brackets_before_after - open_brackets_before_after)[-1]

#     return between_lines

# def test_auto_bracket_matcher():
#     # Test case 1: No brackets are missing
#     before_lines = """
#     public class Test {
#         public static void main(String[] args) {
#             System.out.println("Hello, world!");
#     """
#     between_lines = """
#             if (true) {
#                 System.out.println("True!");
#             }
#         }
#     """
#     after_lines = """
#     }
#     """
#     assert normalize_string(auto_bracket_matcher(before_lines, between_lines, after_lines)) == normalize_string(between_lines)
#     # Test case 2: One closing bracket is missing in between_lines
#     before_lines = """
#     public class Test {
#         public static void main(String[] args) {
#             System.out.println("Hello, world!");
#     """
#     between_lines = """
#             if (true) {
#                 System.out.println("True!");
#         }
#     """
#     after_lines = """
#     }
#     """
#     expected_between_lines = """
#             if (true) {
#                 System.out.println("True!");
#             }
#         }
#     """
#     print("######### Actual #########")
#     print(normalize_string(auto_bracket_matcher(before_lines, between_lines, after_lines)))
#     print("######### Expected #########")
#     print(normalize_string(expected_between_lines))
#     # assert normalize_string(auto_bracket_matcher(before_lines, between_lines, after_lines)) == normalize_string(expected_between_lines)    
#     import pdb; pdb.set_trace()
#     # Test case 3: One opening bracket is extra in between_lines
#     before_lines = """
#     public class Test {
#         public static void main(String[] args) {
#             System.out.println("Hello, world!");
#     """
#     between_lines = """
#             {
#                 if (true) {
#                 System.out.println("True!");
#             }
#         }
#     """
#     after_lines = """
#     }
#     """
#     expected_between_lines = """
#             if (true) {
#                 System.out.println("True!");
#             }
#         }
#     """
#     # assert auto_bracket_matcher(before_lines, between_lines, after_lines) == expected_between_lines
#     # assert normalize_string(auto_bracket_matcher(before_lines, between_lines, after_lines)) == normalize_string(expected_between_lines)
#     print("######### Actual #########")
#     print(normalize_string(auto_bracket_matcher(before_lines, between_lines, after_lines)))
#     print("######### Expected #########")
#     print(normalize_string(expected_between_lines))
#     import pdb; pdb.set_trace()
#     # Test case 4: Multiple brackets are missing in between_lines
#     before_lines = """
#     public class Test {
#         public static void main(String[] args) {
#             System.out.println("Hello, world!");
#     """
#     between_lines = """
#             if (true) 
#                 System.out.println("True!");
        
#     """
#     after_lines = """
#         }
#     }
#     """
#     expected_between_lines = """
#             if (true) {
#                 System.out.println("True!");
#             }
#     """
#     # assert auto_bracket_matcher(before_lines, between_lines, after_lines) == expected_between_lines
#     # assert normalize_string(auto_bracket_matcher(before_lines, between_lines, after_lines)) == normalize_string(expected_between_lines)
#     print("######### Actual #########")
#     print(normalize_string(auto_bracket_matcher(before_lines, between_lines, after_lines)))
#     print("######### Expected #########")
#     print(normalize_string(expected_between_lines))
#     import pdb; pdb.set_trace()
#     # Test case 5: Multiple brackets are extra in between_lines
#     before_lines = """
#     public class Test {
#         public static void main(String[] args) {
#             System.out.println("Hello, world!");
#     """
#     between_lines = """
#             if (true) {
#                 System.out.println("True!");
#             }
#         }
#     }
#     """
#     after_lines = """
#         }
#     }
#     """
#     expected_between_lines = """
#             if (true) {
#                 System.out.println("True!");
#             }
#     """
#     # assert normalize_string(auto_bracket_matcher(before_lines, between_lines, after_lines)) == normalize_string(expected_between_lines)
#     print("######### Actual #########")
#     print(normalize_string(auto_bracket_matcher(before_lines, between_lines, after_lines)))
#     print("######### Expected #########")
#     print(normalize_string(expected_between_lines))
#     import pdb; pdb.set_trace()
#     # assert auto_bracket_matcher(before_lines, between_lines, after_lines) == expected_between_lines

# test_auto_bracket_matcher()
