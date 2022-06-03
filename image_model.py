import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from utils import image_int_to_float, image_float_to_int
from arch.ARDM.ARDM_UNet import ARDM_UNet

class MAC(nn.Module):
    def __init__(self, image_dims, cfg):
        super(MAC, self).__init__()

        self.D = 256 # 2**bits in each dimension
        self.image_dims = image_dims
        self.xdim = np.prod(np.array(image_dims)).item()
        self.cfg = cfg

        assert(self.cfg.arch in ['ARDM'])
        assert(self.cfg.mask.strategy in ['none', 'marginal', 'inpainting'])
        assert(self.cfg.mask.order in ['natural', 'spaced', 'random'])

        if self.cfg.arch == 'ARDM':
            self.unet = ARDM_UNet(
                num_classes=256,
                ch=256,
                out_ch=3*256,
                ch_mult=[1],
                num_res_blocks=32,
                full_attn_resolutions=[32, 16, 14, 8, 7, 4],
                num_heads=1,
                dropout=0.,
                max_time=1000.,
                weave_attn=self.cfg.weave_attn)
        else:
            raise NotImplementedError

    def sum_except_batch(self, x):
        return x.reshape(x.shape[0], -1).sum(-1)

    def unet_forward(self, x, mask):
        if self.cfg.arch == 'ARDM':
            logits = self.unet(x, self.sum_except_batch(mask), mask)
        else:
            logits = self.unet(x)
        return logits

    def sample_mask(self, batch, device, strategy, normalize_cardinality=False):
        def get_batch(batch_inner):
            if self.cfg.mask.mixture:
                none_previous_selection, none_current_selection = self._sample_mask(batch_inner//2, device, 'none')
                strategy_previous_selection, strategy_current_selection = self._sample_mask(batch_inner - batch_inner//2, device, strategy)

                previous_selection = torch.cat((none_previous_selection, strategy_previous_selection), dim=0)
                current_selection = torch.cat((none_current_selection, strategy_current_selection), dim=0)

                return previous_selection, current_selection
            else:
                return self._sample_mask(batch_inner, device, strategy)
        
        if normalize_cardinality:
            batch_outer = 100 * batch
            previous_selection, current_selection = get_batch(batch_inner=batch_outer)
            t = self.sum_except_batch(previous_selection)
            weights = (t+1) # 
            idx_select = torch.multinomial(weights.float(), num_samples=batch, replacement=False)
            return previous_selection[idx_select], current_selection[idx_select]
        else:
            return get_batch(batch_inner=batch)


    def _sample_mask(self, batch, device, strategy, shiftup=False):
        if strategy == 'none':
            sigma = torch.rand(size=(batch, self.xdim), device=device)
            sigma = torch.argsort(sigma, dim=-1).reshape(batch, *self.image_dims)
            if shiftup: # can't have the zero mask, but can have the full mask
                t = torch.randint(low=1, high=self.xdim+1, size=(batch,), device=device)
            else:
                t = torch.randint(high=self.xdim, size=(batch,), device=device)
            twrap = t.reshape(batch, 1, 1, 1)
        elif strategy == 'marginal':
            # sample a final mask
            mask, _ = self._sample_mask(batch, device, strategy='none', shiftup=True)
            t = self.sum_except_batch(mask) # t is at least 1 due to shiftup=True

            # sample an intermediate prefix by taking a random int from [0, t)
            batch_arange = torch.arange(self.xdim, device=device).reshape(1, self.xdim).repeat(batch, 1)
            nonzero_weights = batch_arange < t.reshape(batch, 1)
            weights = torch.ones(batch, self.xdim, device=device).float()
            weights = weights * nonzero_weights
            tpre = torch.multinomial(weights.float(), num_samples=1)[:,0]
            twrap = tpre.reshape(batch, 1, 1, 1)

            sigma = self.mask_to_order(mask, order_strategy=self.cfg.mask.order)
        elif strategy == 'inpainting':
            # sample a final mask
            mask, _ = self._sample_mask(batch, device, strategy='none')
            t = self.sum_except_batch(mask)

            # sample an intermediate suffix by taking a random int from [t, self.xdim)
            batch_arange = torch.arange(self.xdim, device=device).reshape(1, self.xdim).repeat(batch, 1)
            nonzero_weights = batch_arange < (self.xdim-t).reshape(batch, 1)
            weights = torch.ones(batch, self.xdim, device=device).float()
            weights = weights * nonzero_weights
            tpost = t + torch.multinomial(weights.float(), num_samples=1)[:,0]
            twrap = tpost.reshape(batch, 1, 1, 1)

            sigma = self.mask_to_order(mask, order_strategy=self.cfg.mask.order) # get natural (stable) ordering, 1s before 0s
        else:
            raise NotImplementedError

        previous_selection = sigma < twrap
        current_selection = sigma == twrap

        return previous_selection, current_selection

    def mask_to_order(self, mask, order_strategy):
        '''
        mask is bitmask of size (batch, *self.image_dims)
        we will mask an img x by doing x*img
        so we call 1-bits 'unmasked', and 0-bits 'masked'
        '''
        batch = mask.shape[0]

        if order_strategy == 'natural':
            # natural ordering
            # using .sort(descending=True, stable=True).indices instead of .argsort() so that we can pass in stable=True
            # uses natural ordering of the unmasked, which might be unsuitable (some cases we should randomize it)
            flat_unmasked_first_order = mask.long().reshape(batch, self.xdim).sort(descending=True, stable=True).indices.argsort()
        elif order_strategy == 'spaced':
            # spaced ordering
            # within the ones, we order by (the reverse of) 0, 48, 96, ..., 1, 49, 97, ..., 2, ...
            # and the same within the zeros
            large_constant = (int)(1e8)
            flat_mask = mask.long().reshape(batch, self.xdim) * large_constant
            for i in range(self.xdim // 48):
                flat_mask[:,i::48] += i
            flat_unmasked_first_order = flat_mask.argsort(descending=True).argsort()
        elif order_strategy == 'random':
            # we want to place all the ones before zeros, but randomize the ordering in each bucket
            # to do so just add a large constant to all the ones, then add random noise to every value, and sort
            large_constant = (int)(1e8)
            flat_mask = mask.long().reshape(batch, self.xdim) * large_constant
            flat_noise_mask = flat_mask + torch.randint(high=self.xdim, size=(batch, self.xdim), device=mask.device)
            flat_unmasked_first_order = flat_noise_mask.argsort(descending=True).argsort()
        else:
            raise NotImplementedError

        unmasked_first_order = flat_unmasked_first_order.reshape(*mask.shape)
        return unmasked_first_order


    def likelihood(self, x, mask, order=None, full=True):
        # mask should have cardinality at least one

        if mask is None: mask = torch.ones(*x.shape, device=x.device).long()

        batch = x.shape[0]
        zeroimg = torch.zeros(batch, *self.image_dims, device=x.device)

        if order is not None:
            sigma = order
        else:
            if self.cfg.mask.strategy == 'none':
                sigma = self.mask_to_order(mask, order_strategy='random')
            elif self.cfg.mask.strategy == 'marginal':
                sigma = self.mask_to_order(mask, order_strategy=self.cfg.mask.order)
            elif self.cfg.mask.strategy == 'inpainting':
                sigma = self.mask_to_order(mask, order_strategy=self.cfg.mask.order)
            else:
                raise NotImplementedError
        T = self.sum_except_batch(mask)

        total_ll = 0

        if not full:
            # instead of doing the full likelihood, just use one timestep as an approximation

            t = T
            # sample an intermediate prefix by taking a random int from [0, t)
            batch_arange = torch.arange(self.xdim, device=x.device).reshape(1, self.xdim).repeat(batch, 1)
            nonzero_weights = batch_arange < t.reshape(batch, 1)
            weights = torch.ones(batch, self.xdim, device=x.device).float()
            weights = weights * nonzero_weights
            tpre = torch.multinomial(weights.float(), num_samples=1)[:,0]
            twrap = tpre.reshape(batch, 1, 1, 1)

            previous_selection = sigma < twrap
            current_selection = sigma == twrap

            xin = x * previous_selection + zeroimg * (~previous_selection)

            logits = self.unet_forward(xin, previous_selection).reshape(batch, self.D, *self.image_dims)
            logits = torch.permute(logits, (0,2,3,4,1))
            distout = torch.distributions.categorical.Categorical(logits=logits)

            ll = distout.log_prob( image_float_to_int(x) )
            ll = self.sum_except_batch(ll * current_selection)

            # importance weight
            ll = ll * t / self.xdim

            return ll.mean()


        for t in range(self.xdim):
            if t > T.max(): break
            #print("%u out of %u steps" % (t, T.max()))
            previous_selection = (sigma < t)
            current_selection = (sigma == t)

            xin = x * previous_selection + zeroimg * (~previous_selection)

            logits = self.unet_forward(xin, previous_selection).reshape(batch, self.D, *self.image_dims)
            logits = torch.permute(logits, (0,2,3,4,1))
            distout = torch.distributions.categorical.Categorical(logits=logits)

            ll = distout.log_prob( image_float_to_int(x) )
            ll = self.sum_except_batch(ll * current_selection)
            ll = ll * (T > t) # stop if we're done with all the unmasked inputs

            total_ll += ll

            if t % 300 == 0:
                print(t)
                print(total_ll.mean() / (t+1))

        return total_ll.mean()

    def forward(self, x):
        batch = x.shape[0]
        
        zeroimg = torch.zeros(batch, *self.image_dims, device=x.device)

        previous_selection, current_selection = self.sample_mask(batch, x.device, strategy=self.cfg.mask.strategy, normalize_cardinality=self.cfg.mask.normalize_cardinality)
        future_selection = ~previous_selection

        xin = x * previous_selection + zeroimg * (~previous_selection)

        logits = self.unet_forward(xin, previous_selection).reshape(batch, self.D, *self.image_dims)
        logits = torch.permute(logits, (0,2,3,4,1))
        distout = torch.distributions.categorical.Categorical(logits=logits)

        ll = distout.log_prob( image_float_to_int(x) )
        ll_final = self.sum_except_batch(ll * future_selection) / self.sum_except_batch(future_selection)

        return ll_final.mean()

    def conditional_sample(self, X, mask, sharpness=1):
        batch = X.shape[0]
        zeroimg = torch.zeros(batch, *self.image_dims, device=X.device)
        mask = mask.bool()
        xin = X * mask

        if self.cfg.mask.strategy == 'none':
            sigma = self.mask_to_order(mask, order_strategy='random')
        elif self.cfg.mask.strategy == 'marginal':
            sigma = self.mask_to_order(mask, order_strategy=self.cfg.mask.order)
        elif self.cfg.mask.strategy == 'inpainting':
            sigma = self.mask_to_order(mask, order_strategy=self.cfg.mask.order)
        else:
            raise NotImplementedError

        start_t = self.sum_except_batch(mask).min()
        for t in range(start_t, self.xdim):
            if t % 10 == 0:
                print("%u out of %u steps" % (t, self.xdim))
            
            previous_selection = (sigma < t)
            current_selection = (sigma == t)

            logits = self.unet_forward(xin, previous_selection).reshape(batch, self.D, *self.image_dims)
            logits = torch.permute(logits, (0,2,3,4,1))

            probs = F.softmax(logits * sharpness, dim=-1)
            probs = (probs * current_selection.unsqueeze(dim=-1)).sum(dim=(1,2,3))

            sample = torch.multinomial(probs, num_samples=1).squeeze()
            sample = sample.reshape(batch, 1, 1, 1)
            sample = image_int_to_float(sample)

            xin = xin * previous_selection + sample * current_selection + zeroimg * (~(previous_selection | current_selection))
            xin = X * mask + xin * (~mask) # make sure each time we reinstate the evidence

        return xin

    def sample(self, batch, device='cuda:0', sharpness=1):
        xin = torch.zeros(batch, *self.image_dims, device=device)
        mask = xin.bool()
        return self.conditional_sample(xin, mask, sharpness)
