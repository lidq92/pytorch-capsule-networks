import torch.nn as nn
import torch.nn.functional as F

class MarginLoss(nn.Module):
	def __init__(self, size_average=False, loss_lambda=0.5, 
				 m_plus=0.9, m_minus=0.1):
		"""
		Margin loss:
		L_k = T_k * \max{(0, m^{+} - ||v_k||)}^2 + \lambda * (1 - T_k) * 
		\max{(0, ||v_k|| - m^{-})}^2
		
		Args:
			size_average: should the losses be averaged (True) or summed (False) over observations for each minibatch.
			loss_lambda, m_plus, m_minus correspond to the formula parameters.
		"""
		super(MarginLoss, self).__init__()
		self.size_average = size_average
		self.m_plus = m_plus
		self.m_minus = m_minus
		self.loss_lambda = loss_lambda

	def forward(self, inputs, labels):
		L_k = labels * F.relu(self.m_plus - inputs)**2 + self.loss_lambda * \
			  (1 - labels) * F.relu(inputs - self.m_minus)**2
		L_k = L_k.sum(dim=1)

		if self.size_average:
			return L_k.mean()
		else:
			return L_k.sum()

class CapsuleLoss(nn.Module):
	def __init__(self, loss_lambda=0.5, recon_loss_scale=5e-4, 
				 size_average=False, m_plus=0.9, m_minus=0.1):
		"""
		Combined margin loss and reconstruction loss.
		L = L_m + recon_loss_scale * L_r
		
		Args:
			recon_loss_scale: param for scaling down the reconstruction loss
			size_average: if True, reconstruction loss is MSE instead of SSE
		"""
		super(CapsuleLoss, self).__init__()
		self.size_average = size_average
		self.margin_loss = MarginLoss(size_average=size_average, 
									  loss_lambda=loss_lambda, 
									  m_plus=m_plus, 
									  m_minus=m_minus)
		self.recon_loss = nn.MSELoss(size_average=size_average)
		self.recon_loss_scale = recon_loss_scale

	def forward(self, inputs, labels, images, reconstructions):
		margin_loss = self.margin_loss(inputs, labels)
		recon_loss = self.recon_loss(reconstructions, images)
		caps_loss = margin_loss + self.recon_loss_scale * recon_loss

		return caps_loss
