## Dataset Structure
<!-- We gave 11 columns in the datasets, which are:

`task_type`: with four unique values: `infilling_java`, `completion_java`, `infilling_python`, `completion_python`

`code_task`: The Java repositories feature advanced academic projects, including text processing, data structures (AVL, B-tree, M-Tree), and statistical algorithms. The Python repositories range from notebook scripts to object-oriented codes, covering linear programming, computer vision, and reinforcement learning.

`start_line` and `end_line`: Marks the starting and ending line number of code segament to be complete.

`before`, `between`, and `after`: Records the code above the to-be complete code block, the ground truth of to-be complete code block, and code following the code block. 

`reason_categories_output`: A list of dictionaries records the `usage_line` of logic components of the to-be complete code block, such as `If Body`, `If Condition`, `Loop Body`, etc. 

`horizon_categories_output`: Records the programming constucts such as `Global_Variable`, `Function`, `Class` and their `define_line` and `usage_line`

`reason_freq_analysis`: A dictionary records the counts of the  `reason_categories_output`

`horizon_freq_analysis`: A dictionary records the counts of the  `horizon_categories_output` -->

The dataset comprises 11 columns, detailed as follows:

- `task_type`: Identifies the task category, with options including `infilling_java`, `completion_java`, `infilling_python`, and `completion_python`.

- `code_task`: Describes the nature of the coding tasks. For Java, the tasks involve advanced academic projects focusing on text processing, data structures (such as AVL, B-tree, M-Tree), and statistical algorithms. Python tasks span from notebook scripts to object-oriented programming, covering areas like linear programming, computer vision, and reinforcement learning.

- `start_line` and `end_line`: Specify the beginning and ending line numbers of the code segment targeted for completion.

- `before`, `between`, and `after`: Capture the code preceding the target code block, the ground truth of the target code block, and the code following the target block, respectively.

- `reason_categories_output`: A collection of dictionaries detailing the `usage_line` for logical components within the target code block, including elements like `If Body`, `If Condition`, `Loop Body`, etc.

- `horizon_categories_output`: Documents the programming constructs such as `Global_Variable`, `Function`, `Class`, along with their `define_line` and `usage_line`.

- `reason_freq_analysis` and `horizon_freq_analysis`: These dictionaries tally the occurrences within `reason_categories_output` and `horizon_categories_output`, respectively.