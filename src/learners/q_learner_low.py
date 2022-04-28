import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
from torch.optim import RMSprop
from controllers import REGISTRY as mac_REGISTRY


class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.mac_all = mac_REGISTRY[args.upper_mac](scheme, args)
        
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        
        # Calculate estimated Q-Values
        mac_out_all = []
        self.mac_all.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac_all.forward(batch, t=t)
            mac_out_all.append(agent_outs)
        mac_out_all = th.stack(mac_out_all, dim=1)  # Concat over time

        # Td-error
        td_error = (mac_out[:, :-1] - mac_out_all[:, :-1].detach())

        mask = mask.unsqueeze(-1).expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.log_stats_t = t_env

    def cuda(self):
        self.mac.cuda()
        self.mac_all.cuda()

    def save_models(self, path):
        self.mac.save_models(path)

    def load_models(self, path):
        self.mac_all.load_models(path)
