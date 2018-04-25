import torch.nn as nn
import torch.nn.functional as F


class CapsuleLoss(nn.Module):
    def __init__(self, size_average=False, loss='margin_loss', loss_lambda=0.5, m_plus=0.9, m_minus=0.1, use_recon=True, recon_loss_scale=0.0005):
        """
		Capsule loss by combining margin/spread loss and reconstruction loss.
		L = L_m (or L_s) + recon_loss_scale * L_r
		
		Args:
			size_average: should the losses be averaged (True) or summed 
			(False) over observations for each minibatch
			loss: which loss to use
			loss_lambda, m_plus, m_minus: parameters of margin loss
			use_recon: use reconstruction loss or not
			recon_loss_scale: parameter for scaling down reconstruction loss
    	"""
        super(CapsuleLoss, self).__init__()
        self.size_average        = size_average
        self.loss                = loss
        self.loss_lambda         = loss_lambda
        self.m_plus              = m_plus
        self.m_minus             = m_minus
        self.use_recon           = use_recon
        self.recon_loss_scale    = recon_loss_scale

        self.reconstruction_loss = nn.MSELoss(size_average=size_average)

    @classmethod
    def spread_loss(cls, x, labels, m):
        """
	    Spread Loss:
		L_i = \max{(0, m-(a_t - a_i))}^2, L = \sum_{i\neq t} L_i
		defaut size_average is True
	    """
        L = F.multi_margin_loss(x, labels, p=2, margin=m, size_average=cls().size_average)

        return L

    @classmethod
    def cross_entropy_loss(cls, x, labels, m):
        """
		Cross Entropy Loss
		x size should be (N, C), where C=num_classes, 0<=labels[i]<=C-1
    	"""
        return F.cross_entropy(x, labels, size_average=cls().size_average)

    @classmethod
    def margin_loss(cls, x, labels, m):
        """
		Margin loss:
		L_k = T_k * \max{(0, m^{+} - ||v_k||)}^2 + \lambda * (1 - T_k) * 
		\max{(0, ||v_k|| - m^{-})}^2
		"""
        left = F.relu(cls().m_plus - x, inplace=True) ** 2
        right = F.relu(x - cls().m_minus, inplace=True) ** 2
        L_k = labels * left + cls().loss_lambda* (1. - labels) * right
        L_k = L_k.sum(dim=1)

        if cls().size_average:
            return L_k.mean()
        return L_k.sum()

    def forward(self, inputs, labels, images, reconstructions, m=0.2):
        """m: parameter of spread loss. To avoid dead capsules in the earlier layers,
        start with a small margin of 0.2 and linearly increasing it during training to 0.9."""
        cap_loss = getattr(self, self.loss)(inputs, labels, m)

        if self.use_recon:
            recon_loss = self.reconstruction_loss(reconstructions, images)
            cap_loss  += self.recon_loss_scale * recon_loss

        return cap_loss
