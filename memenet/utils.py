import torch as T

def pad_epoch(epoch, epochs):
	return str(epoch).rjust(len(str(epochs)), ' ')

def total_param_count(module):
	# return total number of parameters (individual floating-point numbers) in nn.Module
	return sum(int(T.tensor(x.shape).prod()) for x in module.parameters())
