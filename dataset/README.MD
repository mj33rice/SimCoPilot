## Dataset Structure
The dataset comprises 11 columns, detailed as follows:

- `task_type`: Identifies the task category, with options including `infilling_java`, `completion_java`, `infilling_python`, and `completion_python`.

- `code_task`: Describes the nature of the coding tasks. For Java, the tasks involve advanced academic projects focusing on text processing, data structures (such as AVL, B-tree, M-Tree), and statistical algorithms. Python tasks span from notebook scripts to object-oriented programming, covering areas like linear programming, computer vision, and reinforcement learning.

- `start_line` and `end_line`: Specify the beginning and ending line numbers of the code segment targeted for completion.

- `before`, `between`, and `after`: Capture the code preceding the target code block, the ground truth of the target code block, and the code following the target block, respectively.

- `reason_categories_output`: A collection of dictionaries detailing the `usage_line` for logical components within the target code block, including elements like `If Body`, `If Condition`, `Loop Body`, etc.

- `horizon_categories_output`: Documents the programming constructs such as `Global_Variable`, `Function`, `Class`, along with their `define_line` and `usage_line`.

- `reason_freq_analysis` and `horizon_freq_analysis`: These dictionaries tally the occurrences within `reason_categories_output` and `horizon_categories_output`, respectively.

## Data Nutrition Label

The Data Nutrition Label for this dataset has been officially submitted to [Data Nutrition Project](https://datanutrition.org/) and is currently pending approval. For those interested, the draft version of the label can be found in `dataset/Data_Nutrition_Label_for_SimCoPilot.pdf`.