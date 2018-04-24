import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Normal


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
		Primary Capsules Layer
		NIPS 2017

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
		NIPS 2017

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


class PrimaryCaps(nn.Module):
    def __init__(self, A=32, B=32):
        """
	    Primary Capsule Layer
	    ICLR 2018

	    Args:
	        A: input channel
	        B: number of types of capsules.
	    """
        super(PrimaryCaps, self).__init__()
        self.B = B
        self.capsules_pose = nn.ModuleList([nn.Conv2d(in_channels=A,
        									out_channels=4 * 4,
        									kernel_size=1, stride=1) 
        									for _ in range(self.B)])
        self.capsules_activation = nn.ModuleList([nn.Conv2d(in_channels=A, 
        										 out_channels=1,
        										 kernel_size=1, stride=1) 
        										 for _ in range(self.B)])

    def forward(self, x): 
        poses = [self.capsules_pose[i](x) for i in range(self.B)]  
        poses = torch.cat(poses, dim=1)  
        activations = [self.capsules_activation[i](x) for i in range(self.B)]
        activations = F.sigmoid(torch.cat(activations, dim=1))  

        return poses, activations


class ConvCaps(nn.Module):
    def __init__(self, B=32, C=32, K=3, stride=2, iteration=3,
                 coordinate_add=False, transform_share=False, 
                 routing='EM_routing', use_cuda=True):
        """
	    Convolutional Capsule Layer
	    ICLR 2018

	    Args:
	        B: input number of types of capsules.
	        C: output number of types of capsules.
	        K: kernel size of convolution. K = 0 means the capsules in layer L+1's receptive field contain all capsules in layer L, which is used in the final ClassCaps layer.
	        stride: stride of convolution
	        iteration: number of EM iterations
	        coordinate_add: whether to use Coordinate Addition
	        transform_share: whether to share transformation matrix.
	        routing: 'EM_routing' or 'angle_routing'
	    """
        super(ConvCaps, self).__init__()
        self.routing = routing
        self.use_cuda = use_cuda
        self.B = B
        self.C = C
        self.K = K # K = 0 means full receptive field like class capsules
        self.Bkk = None
        self.Cww = None
        self.b = None # batch_size, get it in forword process
        self.stride = stride
        self.coordinate_add = coordinate_add
        # transform_share is also set to True if K = 0
        self.transform_share = transform_share or K == 0
        self.beta_v = None
        self.beta_a = None
        if not transform_share: 
            self.W = nn.Parameter(torch.randn(B, K, K, C, 4, 4))  
        else:
            self.W = nn.Parameter(torch.randn(B, C, 4, 4)) 

        self.iteration = iteration

    def coordinate_addition(self, width_in, votes):
        add = [[i / width_in, j / width_in] for i in range(width_in) for j in range(width_in)]  # K,K,w,w
        add = Variable(torch.Tensor(add))
        if self.use_cuda:
            add = add.cuda()
        add = add.view(1, 1, self.K, self.K, 1, 1, 1, 2)
        add = add.expand(self.b, self.B, self.K, self.K, self.C, 1, 1, 2).contiguous()
        votes[:, :, :, :, :, :, :, :2, -1] = votes[:, :, :, :, :, :, :, :2, -1] + add
        return votes

    def down_w(self, w):
        return range(w * self.stride, w * self.stride + self.K)

    def EM_routing(self, lambda_, a_, V):
        # routing coefficient
        R = Variable(torch.ones([self.b, self.Bkk, self.Cww]), requires_grad=False)
        if self.use_cuda:
            R = R.cuda()
        R /= self.Cww

        for i in range(self.iteration):
            # M-step
            R = (R * a_)[..., None]
            sum_R = R.sum(1)
            mu = ((R * V).sum(1) / sum_R)[:, None, :, :]
            sigma_square = (R * (V - mu) ** 2).sum(1) / sum_R

            # E-step
            if i != self.iteration - 1:
                mu, sigma_square, V_, a__ = mu.data, sigma_square.data, V.data, a_.data
                normal = Normal(mu, sigma_square[:, None, :, :] ** (1 / 2))
                p = torch.exp(normal.log_prob(V_))
                ap = a__ * p.sum(-1)
                R = Variable(ap / torch.sum(ap, -1)[..., None], requires_grad=False)
            else:
                const = (self.beta_v.expand_as(sigma_square) + torch.log(sigma_square)) * sum_R
                a = torch.sigmoid(lambda_ * (self.beta_a.repeat(self.b, 1) - const.sum(2)))

        return a, mu

    def angle_routing(self, lambda_, a_, V):
        # routing coefficient
        R = Variable(torch.zeros([self.b, self.Bkk, self.Cww]), requires_grad=False)
        if self.use_cuda:
            R = R.cuda()

        for i in range(self.iteration):
            R = F.softmax(R, dim=1)
            R = (R * a_)[..., None]
            sum_R = R.sum(1)
            mu = ((R * V).sum(1) / sum_R)[:, None, :, :]

            if i != self.iteration - 1:
                u_v = mu.permute(0, 2, 1, 3) @ V.permute(0, 2, 3, 1)
                u_v = u_v.squeeze().permute(0, 2, 1) / V.norm(2, -1) / mu.norm(2, -1)
                R = R.squeeze() + u_v
            else:
                sigma_square = (R * (V - mu) ** 2).sum(1) / sum_R
                const = (self.beta_v.expand_as(sigma_square) + torch.log(sigma_square)) * sum_R
                a = torch.sigmoid(lambda_ * (self.beta_a.repeat(self.b, 1) - const.sum(2)))

        return a, mu

    def forward(self, x, lambda_):
        poses, activations = x
        width_in = poses.size(2)
        w = int((width_in - self.K) / self.stride + 1) if self.K else 1  # 5
        self.Cww = w * w * self.C
        self.b = poses.size(0) # 

        if self.beta_v is None:
            if self.use_cuda:
                self.beta_v = nn.Parameter(torch.randn(1, self.Cww, 1)).cuda()
                self.beta_a = nn.Parameter(torch.randn(1, self.Cww)).cuda()
            else:
                self.beta_v = nn.Parameter(torch.randn(1, self.Cww, 1))
                self.beta_a = nn.Parameter(torch.randn(1, self.Cww))

        if self.transform_share:
            if self.K == 0:
                self.K = width_in  # class Capsules' kernel = width_in
            W = self.W.view(self.B, 1, 1, self.C, 4, 4).expand(self.B, self.K, self.K, self.C, 4, 4).contiguous()
        else:
            W = self.W  # B,K,K,C,4,4

        self.Bkk = self.K * self.K * self.B

        # used to store every capsule i's poses in each capsule c's receptive field
        pose = poses.contiguous()  # b,16*32,12,12
        pose = pose.view(self.b, 16, self.B, width_in, width_in).permute(0, 2, 3, 4, 1).contiguous()  # b,B,12,12,16
        poses = torch.stack([pose[:, :, self.stride * i:self.stride * i + self.K,
                             self.stride * j:self.stride * j + self.K, :] for i in range(w) for j in range(w)],
                            dim=-1)  # b,B,K,K,w*w,16
        poses = poses.view(self.b, self.B, self.K, self.K, 1, w, w, 4, 4)  # b,B,K,K,1,w,w,4,4
        W_hat = W[None, :, :, :, :, None, None, :, :]  # 1,B,K,K,C,1,1,4,4
        votes = W_hat @ poses  # b,B,K,K,C,w,w,4,4

        if self.coordinate_add:
            votes = self.coordinate_addition(width_in, votes)
            activation = activations.view(self.b, -1)[..., None].repeat(1, 1, self.Cww)
        else:
            activations_ = [activations[:, :, self.down_w(x), :][:, :, :, self.down_w(y)]
                            for x in range(w) for y in range(w)]
            activation = torch.stack(
                activations_, dim=4).view(self.b, self.Bkk, 1, -1) \
                .repeat(1, 1, self.C, 1).view(self.b, self.Bkk, self.Cww)

        votes = votes.view(self.b, self.Bkk, self.Cww, 16)
        activations, poses = getattr(self, self.routing)(lambda_, activation, votes)
        return poses.view(self.b, self.C, w, w, -1), activations.view(self.b, self.C, w, w)
