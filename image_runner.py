# IMPORTS
import os
import numpy as np
import torch
import torch.optim as optim
from data.dataset import get_dataset
from image_model import MAC
import wandb

class Runner(object):
    def __init__(self, cfg):
        self.cfg = cfg

        self.device_id = 'cuda:{}'.format(cfg.local_rank)
        self.master_node = (self.cfg.local_rank == 0)
        self.distributed = (self.cfg.world_size > 1)

        self.train_loader, self.test_loader = get_dataset(cfg, distributed=self.distributed)

        self.obs = (1, 28, 28) if 'MNIST' in self.cfg.dataset else (3, 32, 32)
        xdim = np.prod(self.obs)
        self.epoch = 0

        self.net = MAC(image_dims=self.obs, cfg=self.cfg)
        self.net.to(self.device_id)

        if self.distributed:
            self.net = torch.nn.parallel.DistributedDataParallel(self.net, device_ids=[cfg.local_rank], output_device=cfg.local_rank)
            self.net_module = self.net.module
        else:
            self.net_module = self.net
        
        self.clip_grad = 100.
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.cfg.lr)

        if self.cfg.loadpath is not None:
            self.load(self.cfg.loadpath)

        self.save_every = 200
        self.eval_every = 5

        if self.cfg.dataset == 'IMAGENET32':
            self.save_every = 8
            self.eval_every = 1

    def load(self, path):
        map_location = {"cuda:0": self.device_id}
        checkpoint = torch.load(path, map_location=map_location)
        self.net_module.load_state_dict(checkpoint['net'])        
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epoch = checkpoint['epoch']
        print("loaded", flush=True)

    def train(self):
        print("training rank %u" % self.cfg.local_rank, flush=True)
        self.net.train()
        dataloader = self.train_loader

        while self.epoch < self.cfg.n_epochs:
            
            epoch_metrics = {
                'log_ll': 0,
                'count': 0,
            }

            bsz = 0
            accum, accumll = 0, 0.0
            self.net.train()

            for it, (X, y) in enumerate(dataloader):
                X = X.cuda(device=self.device_id, non_blocking=True)

                log_ll = self.net(X)
                (-log_ll).backward()

                count = X.shape[0]
                epoch_metrics['log_ll'] += log_ll * count
                epoch_metrics['count'] += count

                bsz += X.shape[0]
                accum += X.shape[0]
                accumll += log_ll * count

                if bsz >= 128 // self.cfg.world_size:
                    total_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip_grad)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    bsz = 0

                if accum >= 12800 // self.cfg.world_size:
                    if self.master_node:
                        print("Iter %u out of %u, log-ll: %.2f" % (it, len(dataloader), log_ll), flush=True)
                        wandb.log({
                            "iter": (it + 1 + len(dataloader)*self.epoch) * self.cfg.batch_size,
                            "batch log_ll": log_ll,
                        })
                        accum = 0
                        accumll = 0.0

            if self.master_node:
                states = {
                    'net': self.net_module.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': self.epoch + 1,
                }
                # if self.config.model.ema:
                #     states.append(ema_helper.state_dict())

                torch.save(states, os.path.join(self.cfg.model_dir, 'checkpoint.pth'))
                if self.epoch % self.save_every == 0:
                    torch.save(states, os.path.join(self.cfg.model_dir, 'checkpoint_{}.pth'.format(self.epoch)))

            if self.epoch % 5 == 0:
                with torch.no_grad():
                    metric_tensor = torch.tensor( [epoch_metrics['log_ll'], epoch_metrics['count'] ] )
                    if self.distributed:
                        torch.distributed.reduce(metric_tensor, dst=0)

                test_epoch_metric_tensor = self.test_marginal()

                if self.master_node:
                    metric_tensor[0] /= metric_tensor[1]
                    wandb.log({
                        "epoch": self.epoch,
                        "train log_ll": metric_tensor[0],
                        "test marg_ll": test_epoch_metric_tensor[0],
                        "test log_ll": test_epoch_metric_tensor[1],
                    })

                    print("Epoch %u out of %u, train log_ll: %.2f, test marg_ll: %.2f, test log_ll: %.2f" % (self.epoch, self.cfg.n_epochs, metric_tensor[0], test_epoch_metric_tensor[0], test_epoch_metric_tensor[1]))

            self.epoch += 1

    def test_marginal(self):
        print("testing")
        self.net.eval()
        dataloader = self.test_loader
        mode = 'test'

        epoch_metrics = {
            'marg_ll': 0,
            'log_ll': 0,
            'count': 0,
        }

        for batch, (X, y) in enumerate(dataloader):
            X = X.cuda(device=self.device_id, non_blocking=True)

            with torch.no_grad():
                mask, _ = self.net_module._sample_mask(X.shape[0], X.device, strategy='none', shiftup=True)
                marg_ll = self.net_module.likelihood(X, mask=mask, full=False)
                log_ll = self.net_module.likelihood(X, mask=None, full=False)

            count = X.shape[0]
            epoch_metrics['marg_ll'] += marg_ll * count
            epoch_metrics['log_ll'] += log_ll * count
            epoch_metrics['count'] += count

        with torch.no_grad():
            metric_tensor = torch.tensor( [epoch_metrics['marg_ll'], epoch_metrics['log_ll'], epoch_metrics['count'] ] )
            if self.distributed:
                torch.distributed.reduce(metric_tensor, dst=0)

            if self.master_node:
                metric_tensor[0] /= metric_tensor[2]
                metric_tensor[1] /= metric_tensor[2]
                print("%s count %u marg_ll: %.4f log_ll: %.4f" % (mode, metric_tensor[2], metric_tensor[0], metric_tensor[1]))

        return metric_tensor