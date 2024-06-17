import json
import numpy as np

def get_ckpts_start(task_name, ckpt_file, task_type):
	_data = json.load(open(ckpt_file))
	ckpt_data = None
	for task in _data:
		if task['Task_Name'] == task_name:
			ckpt_data = task[task_type]
			
	if ckpt_data is None:
		raise ValueError("Task not found in ckpt_file")
	
	ckpt_data = [ckpt[0] for ckpt in ckpt_data]
	ckpt_data = sorted(ckpt_data)  # Line number starts from 1
	return ckpt_data


def get_line_comments_start_and_end(python_file):
	with open(python_file, 'r') as f:
		lines = f.readlines()
		
	comments_spans = []
	start = None
	end = None
	for i, line in enumerate(lines):
		if line.strip().startswith("#") or line.strip().startswith("//"):
			if start is None:
				start = i
		else:
			if start is not None:
				end = i - 1  # End of the comment
				comments_spans.append((start + 1, end + 1))  # Line number starts from 1
				start = None
				end = None
	
	# If the file ends with a comment
	if start is not None:
		comments_spans.append((start + 1, len(lines)))
	return comments_spans


def get_block_comments_start_and_end(python_file):
	with open(python_file, 'r') as f:
		lines = f.readlines()
		
	comments_spans = []
	start = None
	end = None
	for i, line in enumerate(lines):
		if line.strip().startswith("'''") or line.strip().startswith('"""') or line.strip().startswith("/*"):
			if start is None:
				start = i  # Start of the comment
			else:
				end = i	 # End of the comment. Case when line starts with """ or ''' to mark the end of comment
				comments_spans.append((start + 1, end + 1))
				start = None
				end = None
		else:
			if start is not None and line.strip().endswith("'''") or line.strip().endswith('"""') or line.strip().endswith("*/"):
				end = i
				comments_spans.append((start + 1, end + 1))
				start = None
				end = None
				
	# If the file ends with a comment
	if start is not None:
		comments_spans.append((start + 1, len(lines)))
		
	return comments_spans


def map_ckpt_to_comments(ckpts_start, comments_end):
	"""
	For each ckpt, get the closest comment before the ckpt
	
		:param ckpts_start: List of checkpoints start line numbers
		:param comments_end: List of comments end line numbers
		:return: List of tuples (ckpt_start, comment_end)
	
	This assumes that the programmers writes comments before writing the corresponding code.
	Limitation: Does not check the semantics of the comments and code.
	"""
	ckpt_comments = []
	for ckpt in ckpts_start:
		comment_end = None
		for comment in comments_end:
			if comment < ckpt:
				comment_end = comment
			else:
				break
		ckpt_comments.append((ckpt, comment_end))
		
	# Remove the ckpts that do not have any comments before them
	ckpt_comments = [ckpt_comment for ckpt_comment in ckpt_comments if ckpt_comment[1] is not None]
	return ckpt_comments
	

def get_avg_comment_to_ckpt_distance(ckpt_comments):
	"""
		:param ckpt_comments: List of tuples (ckpt_start, comment_end)
		:return: Average distance between the comment and the ckpt
	"""
	ckpt_comments = sorted(ckpt_comments, key=lambda x: x[0])
	distances = [ckpt_comment[0] - ckpt_comment[1] for ckpt_comment in ckpt_comments]
	assert all([distance >= 0 for distance in distances]), "Comment after the ckpt"
	
	mean_distance = np.mean(distances)
	std_distance = np.std(distances)
	
	return mean_distance, std_distance

def get_comment_to_ckpt_by_task(task_name, program_type, task_type, ckpt_file=None):
	if program_type == "Python":
		file = f'../example_code/Python/{task_name}/{task_name}.py'
		if ckpt_file is None:
			ckpt_file = '../example_code/Python/Python_tasks_checkpoints.json'
		ckpts = get_ckpts_start(task_name, ckpt_file, task_type)
		
	elif program_type == "Java":
		task_paths = {
			"FactorizationTester": "A0/Test/",
			"CounterTester": "A1/Test/",
			"DoubleVectorTester": "A2/Test/",
			"SparseArrayTester": "A3/Test/",
			"DoubleMatrixTester": "A4/Test/",
			"RNGTester": "A5/Test/",
			"TopKTester": "A6/Test/",
			"MTreeTester": "A7/Test/"
		}
		file = f'../example_code/Java/COMP215/{task_paths[task_name]}{task_name}.java'
		if ckpt_file is None:
			ckpt_file = '../example_code/Java/Java_tasks_checkpoints.json'
		ckpts = get_ckpts_start(task_name, ckpt_file, task_type)
		
	else:
		raise ValueError("Unsupported program type. Please choose 'Python' or 'Java'.")
	
	hash_comments_spans = get_line_comments_start_and_end(file)
	block_comments_spans = get_block_comments_start_and_end(file)

	all_comments = hash_comments_spans + block_comments_spans
	all_comments_end = [comment[1] for comment in all_comments]
	all_comments_end = sorted(all_comments_end)
	
	ckpt_comments = map_ckpt_to_comments(ckpts, all_comments_end)
	# print("Ckpt to comments: ", ckpt_comments)
	return ckpt_comments

def main():
	# task_name = "FactorizationTester"
	# ckpt_comments = get_comment_to_ckpt_by_task(task_name, "Java", "checkpoint_LN_infilling")
	task_name = "RL_Motion_Planning"
	ckpt_comments = get_comment_to_ckpt_by_task(task_name, "Python", "checkpoint_LN_Completion")
	avg_distance, std_dev = get_avg_comment_to_ckpt_distance(ckpt_comments)
	print("--------------------")
	print("Average distance between the comment and the ckpt: ", avg_distance)
	print("Standard deviation: ", std_dev)
	
	
# if __name__ == "__main__":
# 	main()