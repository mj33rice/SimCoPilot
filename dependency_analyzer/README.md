# Dependency Analyzer

SIMCOPILOT does not simply report the overall pass rate but breaks all of the 1,163 programming tasks into eight overlapping categories and reports results from each.  These categories are as follows: 
1. **Local variable**: The programming task requires the use of a previously-defined local variable.
2. **Global variable**: The task requires the use of a global variable.
3. **Function**: Requires calling a function defined previously in the repository.
4. **Class**: Requires a reference to a class previously defined.
5. **Library**: Requires use of an external library.
6. **If-else condition**: Requires generating the boolean condition in an `if` or `else if` statement.
7. **If-else body**: Requires generating the body of an `if`, `else if`, or `else` statement.
8. **Loop body**: Requires generating a loop body

The number of occurrences of each of these eight different types of tasks is given in the table below:

|                    | Python     |       | Java      |       |
|--------------------|------------|-------|-----------|-------|
|                    | Infill     | Compl.| Infill    | Compl.|
| **Total Count**    | 382        | 212   | 283       | 286   |
| **Local Variable** | 371        | 206   | 232       | 218   |
| **Global Variable**| -          | -     | 157       | 184   |
| **Function**       | 37         | 9     | 110       | 88    |
| **Class**          | 15         | 19    | 54        | 32    |
| **Library**        | 202        | 123   | 6         | 10    |
| **If-Else Cond**   | 19         | 14    | 56        | 33    |
| **If-Else Body**   | 61         | 73    | 79        | 120   |
| **Loop Body**      | 72         | 49    | 102       | 102   |
| **Avg Num Lines**  | 431        |       | 966       |       |

*Table: Task categories and frequencies.*


The number of occurrences of each of these eight different types of tasks is given in Figure

<!-- ## Features

- **Dependency Length Analysis**: Evaluates dependencies based on the length of code preceding a given point, quantifying the distance between the definitions of variables, functions, or classes and their respective calls or implementations in the subsequent sections of code, quantified by the number of lines.
- **Reason and Horizon Categories**: Introduces two crucial analysis categories to better assess
dependency impact and logic handling:
  - **Horizon Category**: Assesses dependency lengths ranging from "Short-Range" to "Cross-Module".
  - **Reason Category**: Evaluates conditional logic, loop terminations, pattern usage, and context awareness.

These categories will offer a comprehensive
analysis of the Language Models’ code synthesis capabilities. Here are some potential categories to
consider: -->

<!-- ### Horizon Category

| Horizon Category | Definition | Characteristics and Examples |
| ---------------- | ---------- | ---------------------------- |
| **Short-Range** | Involves dependencies within a few lines of code. | Local variables, immediate function calls. |
| **Medium-Range** | Covers dependencies over a moderate number of lines in the same file. | References to class variables, methods defined earlier. |
| **Long-Range** | Dependencies extend over a large portion of code. | Understanding of the project structure, including distant components. |
| **Cross-Module** | Relies on elements defined in different modules or libraries. | Requires understanding of external dependencies and library APIs. |


### Reason Category

| Reason Category | Definition | Characteristics and Examples |
| --------------- | ---------- | ---------------------------- |
| **If-else Reasoning** | Involves understanding the logic body of if and else statements. | Assists in developing coherent logic flow in conditional structures. E.g., Suggesting complementary else logic based on the conditions specified in the if statement, or recommending elif branches for multiple conditions. |
| **Define Stop Criteria** | Involves creating the stop condition for loops based on preceding code. | Analyzes the code to determine loop termination conditions. E.g., Suggesting a loop’s stop condition based on the initialization and usage of loop variables | -->

## Example Code's Dependecy Analysis

### Example Code 1

Analyze dependencies within a given segment of code.

```python
1 import math
2 from os import path
3 def example_function():
4     global a
5     a = 1
6     b = 2
7     def inner_function():
8         while i < 10:
9             if i > 2:
10                 print("i is greater than 2")
11             if c > 5:
12                 break
13             else:
14                 c += i
15             i += 1
16         c = a + b
17     d = a + b
18     inner_function()
19     return d
20 ######### Analyze from here #########
21 e = example_function()
22 f = math.sqrt(e)
23 g = path.exists('/example')
24 def add(g, i):
25     a = a + i
26     return g + a
27 list_b = [i for i in range(10)]
28 i = 0
29 while i < 15:
30     if i < 5:
31         print("i < 5")
32     elif i < 10:
33         print("i < 10")
34     else:
35         print("i > 10")
36 ######### Analyze End here #########
37     i += 1
```

#### Output for Example Code 1

```
Reason Categories:
'Define Stop Criteria' detected at line 29.
'If-else Reasoning' detected at line 30.
'If-else Reasoning' detected at line 32.
'Pattern-Based Completion' detected at line 32.
'If-else Reasoning' detected at line 35.

Horizon Categories:
Function 'example_function' used at line 21 is defined at line 3 and has a Medium-Range dependency.
Variable 'e' used at line 22 is locals defined at line 21 and has a Short-Range dependency.
Library 'math' used at line 22 is defined at line 1 and has a Long-Range dependency.
Library 'path' used at line 23 is defined at line 2 and has a Long-Range dependency.
Variable 'i' used at line 25 is locals defined at line 24 and has a Short-Range dependency.
Variable 'g' used at line 26 is locals defined at line 24 and has a Short-Range dependency.
Variable 'a' used at line 26 is globals defined at line 25 and has a Short-Range dependency.
Variable 'i' used at line 27 is part of a list_comp.
Variable 'i' used at line 29 is part of a loop.
Variable 'i' used at line 30 is part of a loop.
Variable 'i' used at line 32 is part of a loop.
```

### Example Code 2

Analyze another segment of code to identify dependencies.

```python
1  global_var = 0
2  ######### Analyze from here #########
3  def outer_function():
4      global global_var
5      global_var = 1
6  
7      class MyClass:
8          def method(self):
9              return global_var
10 
11     return MyClass()
12 
13 instance = outer_function()
14 result = instance.method()
15 print(global_var)
16 print(result)
17 ######### Analyze End here #########
```

#### Output for Example Code 2

```
Reason Categories:
[No reason categories detected in this code segment.]

Horizon Categories:
Variable 'global_var' used at line 9 is globals defined at line 5 and has a Short-Range dependency.
Class 'MyClass' used at line 11 is defined at line 7 and has a Short-Range dependency.
Function 'outer_function' used at line 13 is defined at line 3 and has a Medium-Range dependency.
Variable 'instance' used at line 14 is locals defined at line 13 and has a Short-Range dependency.
Variable 'global_var' used at line 15 is locals defined at line 5 and has a Medium-Range dependency.
Variable 'result' used at line 16 is locals defined at line 14 and has a Short-Range dependency.
```