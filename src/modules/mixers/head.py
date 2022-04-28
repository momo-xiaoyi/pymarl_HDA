import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HEAD(nn.Module):
    def __init__(self, args):
        super(HEAD, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim+self.n_agents, self.embed_dim )
            self.hyper_w_final = nn.Linear(self.state_dim+self.n_agents, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim+self.n_agents, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim+self.n_agents, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim+self.n_agents, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim+self.n_agents, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states, batch):
        bs = agent_qs.size(0)
        nags = agent_qs.size(1) #n_agents
        nacs = agent_qs.size(2) #n_actions
        states = [states.reshape(bs, self.state_dim).unsqueeze(1).unsqueeze(1).expand(-1, nags, nacs, -1)]
        states.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(2).expand(bs, -1, nacs, -1))
        states = th.cat([x.reshape(bs*nags*nacs, -1) for x in states], dim=1).view(bs, nags, nacs, -1)
        if states.size(3) != self.state_dim+nags:
            raise Exception("Error with dim")
            
        
        agent_qs = agent_qs.view(-1, 1, 1)
        # First layer
        states = states.view(-1, self.state_dim+nags)
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, 1, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q = y.view(bs, nags, nacs)
        return q
