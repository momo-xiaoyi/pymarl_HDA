import torch.nn as nn
import torch.nn.functional as F
import torch as th

class RNNAgent(nn.Module):
    def __init__(self, input_shape, input_shape_all, args):
        super(RNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        
        input_shape2 = input_shape_all + args.n_actions
        self.embed_dim = args.central_mixing_embed_dim
        non_lin = nn.ReLU
        self.net = nn.Sequential(nn.Linear(input_shape2, self.embed_dim),
                                 non_lin(),
                                 nn.Linear(self.embed_dim, self.embed_dim),
                                 non_lin(),
                                 nn.Linear(self.embed_dim, self.embed_dim),
                                 non_lin(),
                                 nn.Linear(self.embed_dim, args.n_actions))
        self.V = nn.Sequential(nn.Linear(input_shape_all, self.embed_dim),
                               non_lin(),
                               nn.Linear(self.embed_dim, args.n_actions))

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, states):
        b, a, e = inputs.size()
        
        x = F.relu(self.fc1(inputs.view(-1, e)), inplace=True)
        if hidden_state is not None:
            hidden_state = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, hidden_state)
        q = self.fc2(h)
        
        states = states.view(b*a, -1)
        inputs2 = th.cat([states, q], dim=1)
        advs = self.net(inputs2)
        vs = self.V(states)
        y = advs + vs
        
        return y.view(b, a, -1), h.view(b, a, -1)
