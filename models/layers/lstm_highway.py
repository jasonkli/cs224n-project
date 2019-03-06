import torch
import torch.nn.functional as F
import torch.nn as nn

class LSTMHighway(nn.Module):
	def __init__(self, input_size, output_size):
		super(LSTMHighway, self).__init__()
		self.input_size = input_size
		self.output_size = output_size
		self.downsize = nn.Linear(input_size, output_size)
		self.proj = nn.Linear(input_size, input_size, bias = True)
		self.gate = nn.Linear(input_size, input_size, bias = True)

	def forward(self, x):
		proj_output = F.relu(self.proj(x))
		gate_output = torch.sigmoid(self.gate(x))
		x_highway = (gate_output * proj_output) + ((1 - gate_output) * x)
		x_highway = self.downsize(x_highway)
		return x_highway

### END YOUR CODE 