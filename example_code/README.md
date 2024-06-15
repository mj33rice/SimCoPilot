<!-- # Example Code Analysis Demo -->

<!-- 
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
``` -->