import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def squash(s, dim=-1, eps=1e-8):
	"""
	"Squashing" non-linearity that shrunks short vectors to almost zero 
	length and long vectors to a length slightly below 1
	v_j = ||s_j||^2 / (1 + ||s_j||^2) * s_j / ||s_j||
	
	Args:
		s: 	 Vector before activation
		dim: Dimension along which to calculate the norm
	
	Returns:
		v:   Squashed vector
	"""
	squared_norm = torch.sum(s**2, dim=dim, keepdim=True)
	v = squared_norm / (1 + squared_norm) * \
		s / (torch.sqrt(squared_norm) + eps)
	return v


class PrimaryCapsules(nn.Module):
	def __init__(self, in_channels, out_channels, 
				 dim_caps, kernel_size=9, stride=2):
		"""
		Primary Capsules layer.

		Args:
			in_channels:  Number of input channels
			out_channels: Number of output channels
			dim_caps:	  length of the output capsule vector
		"""
		super(PrimaryCapsules, self).__init__()
		self.dim_caps = dim_caps
		self._caps_channel = int(out_channels / dim_caps)
		assert self._caps_channel * dim_caps == out_channels #
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

	def forward(self, x):
		out = self.conv(x)
		out = out.view(out.size(0), self._caps_channel, 
			           out.size(2), out.size(3), self.dim_caps) #
		out = out.view(out.size(0), -1, self.dim_caps) #
		return squash(out)


class RoutingCapsules(nn.Module):
	def __init__(self, in_dim, in_caps, num_caps, dim_caps, 
				 num_routing=3, use_cuda=True):
		"""
		Routing Capsules Layer

		Args:
			in_dim: 	 length of input capsule vector
			in_caps: 	 Number of input capsules if digits layer
			num_caps: 	 Number of capsules in the capsule layer
			dim_caps: 	 length of the output capsule vector
			num_routing: Number of iterations during routing algorithm	
		"""
		super(RoutingCapsules, self).__init__()
		self.use_cuda = use_cuda
		self.in_dim = in_dim
		self.in_caps = in_caps
		self.num_caps = num_caps
		self.dim_caps = dim_caps
		self.num_routing = num_routing

		self.W = nn.Parameter(0.01 * torch.randn(1, num_caps, in_caps, 
												 dim_caps, in_dim ))
	
	def __repr__(self):
		"""

		"""
		tab = '  '
		line = '\n'
		next = ' -> '
		res = self.__class__.__name__ + '('
		res = res + line + tab + '(' + str(0) + '): ' + 'CapsuleLinear('
		res = res + str(self.in_dim) + ', ' + str(self.dim_caps) + ')'
		res = res + line + tab + '(' + str(1) + '): ' + 'Routing('
		res = res + 'num_routing=' + str(self.num_routing) + ')'
		res = res + line + ')'
		return res

	def forward(self, x):
		batch_size = x.size(0)
		# (batch_size, in_caps, in_dim) -> (batch_size, 1, in_caps, in_dim, 1)
		x = x.unsqueeze(3).unsqueeze(1)
		#
		# W @ x =
		# (1, num_caps, in_caps, dim_caps, in_dim) 
		# @ 
		# (batch_size, 1, in_caps, in_dim, 1) 
		# =
		# (batch_size, num_caps, in_caps, dim_caps, 1)
		u_hat = torch.matmul(self.W, x)
		# (batch_size, num_caps, in_caps, dim_caps)
		u_hat = u_hat.squeeze(-1)

		'''
		detach u_hat during routing iterations 
		to prevent gradients from flowing, i.e., 
		- In forward pass, u_hat_detached = u_hat;
        - In backward, no gradient can flow from u_hat_detached back to x_hat.
        '''
		u_hat_detached = u_hat.detach()

		# Routing algorithm
		b = Variable(torch.zeros(batch_size, self.num_caps, self.in_caps, 1))
		if self.use_cuda:
			b = b.cuda()

		for route_iter in range(self.num_routing-1):
			# (batch_size, num_caps, in_caps, 1) -> Softmax along num_caps
			c = F.softmax(b, dim=1)

			# element-wise multiplication
			# (batch_size, num_caps, in_caps, 1)
			# * 
			# (batch_size, in_caps, num_caps, dim_caps) 
			# -> (batch_size, num_caps, in_caps, dim_caps) 
			# sum across in_caps ->
			# (batch_size, num_caps, dim_caps)
			s = (c * u_hat_detached).sum(dim=2)
			# apply "squashing" non-linearity along dim_caps
			v = squash(s)
			# dot product agreement 
			# between the current output vj and the prediction uj|i
			# (batch_size, num_caps, in_caps, dim_caps) 
			# @ 
			# (batch_size, num_caps, dim_caps, 1)
			# -> (batch_size, num_caps, in_caps, 1)
			uv = torch.matmul(u_hat_detached, v.unsqueeze(-1))
			b += uv # Note: it seems more appropriate here to use b = uv
		
		'''
		last iteration is done on the original u_hat, without the routing 
		weights update
		use u_hat to compute v in order to backpropagate gradient
		'''
		c = F.softmax(b, dim=1)
		s = (c * u_hat).sum(dim=2)
		v = squash(s)

		return v
