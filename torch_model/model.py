from __future__ import division
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np


class Model(object):

    def __init__(
        self, gpus, ckpt_path, net, opt, loss_fn, lr_sched=None, ckpt_step=None,
    ):
        assert len(gpus) > 0, 'Please specify gpu or gpus to run a job.'

        self.net, self.opt, self.loss_fn, self.gpus = net, opt, loss_fn, gpus
        self.param_init = [param.cpu().detach().numpy() for param in self.net.parameters()]
        self.epoch, self.step = 0, 0
        self.lrs, self.w_decays, self.dps, self.distances = [], [], [], [] # for inspection

        self.device = torch.device('cuda:{}'.format(gpus[0]))
        self.net.to(self.device)
        self.lr_sched = lr_sched if lr_sched is not None else None

        # CHECKPOINT RESTORE
        self.ckpt_path = ckpt_path
        if not os.path.exists(self.ckpt_path):   
            os.makedirs(self.ckpt_path)
        self.load(self.ckpt_path, step=ckpt_step)
    
    def fit(
        self, num_epoch, tr_loader, testloader, save_at=[],
    ):
        while self.epoch < num_epoch:
            self.net.train()
            tr_losses, tr_scales = [], []
            for tr_data in tr_loader:
                if self.lr_sched is not None:
                    self.lr_sched.step()

                # Inspect LR and W_Decay
                lr = self.opt.param_groups[0]['lr']
                self.lrs.append(lr)

                tr_img, tr_lbl = tr_data
                tr_img = tr_img.to(self.device, dtype=torch.float)
                tr_lbl = tr_lbl.to(self.device, dtype=torch.long)
                self.opt.zero_grad()

                logit = self.net(tr_img)
                if type(logit) in [list, tuple]:
                    scale = logit[1]
                    logit = logit[0]
                    scale = torch.mean(scale).item() if isinstance(scale, torch.Tensor) else scale
                loss = self.loss_fn(logit, tr_lbl)
                loss.backward()
                self.opt.step()
                self.step += 1
                tr_losses.append(loss.item())
                if not vars().has_key('scale'):
                    scale = 1
                tr_scales.append(scale)
            self.epoch += 1

            # Validation Step
            self.net.eval()
            with torch.no_grad():
                val_loss, val_acc = self.validate(testloader)

            print("Epoch {} step {} [{:.4f}]: s:{:.3f}, {:.5f}, {:.5f} - V.Acc: {:.2f}%".format(
                self.epoch, self.step, lr, np.mean(tr_scales), np.mean(tr_losses),
                val_loss, val_acc
            ))
            if self.epoch in save_at:
                self.save()

    def validate(self, testloader):
        self.net.eval()
        correct = 0
        total = 0
        losses = []
        with torch.no_grad():
            for data in testloader:
                x, y = data
                x = x.to(self.device, dtype=torch.float)
                y = y.to(self.device, dtype=torch.long)
                val_lg = self.net(x)
                if type(val_lg) in [list, tuple]: val_lg = val_lg[0] # in case network return multiple values
                _, pred = torch.max(val_lg, 1)
                total += y.size(0)
                correct += (pred == y).sum().item()
                loss = self.loss_fn(val_lg, y).item()
                losses.append(loss)
        val_loss = np.mean(losses)
        val_acc = float(correct) / total
        return val_loss, val_acc*100

    def predict(self, dataloader):
        self.net.eval()
        lgs = []
        with torch.no_grad():
            for data in dataloader:
                x, y = data
                x = x.to(self.device, dtype=torch.float)
                y = y.to(self.device, dtype=torch.long)
                lg = self.net(x)
                lgs.append(lg)
        lgs = torch.cat(lgs)
        sms = torch.nn.functional.softmax(lgs, dim=1)
        prob, pred = torch.max(sms, 1)
        return lgs, sms, prob, pred

    def predict_np(self, x_np, batch_size):
        self.net.eval()
        n_step = int(np.ceil(len(x_np) / batch_size))
        outputs = []
        with torch.no_grad():
            for i in range(n_step):
                start, end = batch_size * i, batch_size * (i + 1)
                x = x_np[start: end].transpose([0, 3, 1, 2])
                x = torch.from_numpy(x).to(self.device, dtype=torch.float)
                out = self.net(x, all_pred=True)
                for n, item in enumerate(out):
                    if len(outputs) == n:
                        outputs.append([item.cpu().numpy()])
                    else:
                        outputs[n].append(item.cpu().numpy())
        for n, output in enumerate(outputs):
            outputs[n] = np.concatenate(output, axis=0)
        return outputs

    def save(self, ckpt_path=None, ckpt_data_ext={}):
        ckpt_data = {
            'epoch': self.epoch,
            'step': self.step,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'lrs': self.lrs,
            'w_decays': self.w_decays,
            'dps': self.dps,
            'distances': self.distances,
        }
        ckpt_data.update(ckpt_data_ext)
        if self.lr_sched is not None:
            ckpt_data['lr_sched_state_dict'] = self.lr_sched.state_dict()
        
        ckpt_path = self.ckpt_path if not ckpt_path else ckpt_path
        ckpt_file = os.path.join(ckpt_path, str(self.step))
        torch.save(ckpt_data, ckpt_file)

    def load(self, ckpt_path=None, step=None):
        ckpt_path = self.ckpt_path if not ckpt_path else ckpt_path
        files = os.listdir(ckpt_path)
        if len(files) > 0:
            ckpt_name = max([int(fi) for fi in files]) if step is None else step
            ckpt_file = os.path.join(ckpt_path, str(ckpt_name))
            print("Restore checkpoint at {}".format(ckpt_file))
            self.ckpt = torch.load(ckpt_file, map_location='cuda:{}'.format(self.gpus[0]))
            self.step = self.ckpt['step']
            self.epoch = self.ckpt['epoch']
            self.net.load_state_dict(self.ckpt['model_state_dict'])
            self.opt.load_state_dict(self.ckpt['optimizer_state_dict'])
            self.lrs = self.ckpt['lrs']
            self.w_decays = self.ckpt['w_decays']
            self.dps = self.ckpt['dps']
            self.distances = self.ckpt['distances']
            if self.lr_sched is not None:
                self.lr_sched.load_state_dict(self.ckpt['lr_sched_state_dict'])
        else:
            print("No checkpoint found at {}".format(self.ckpt_path))


# Examine if there is no weight decay effect to the predicted scale.
class SGDNoWeightDecayLast(optim.SGD):
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0 and i != 0: # do not perform weight decay for fc_w
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss
