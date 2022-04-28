import torch as th
import torch.nn as nn
import torch.nn.functional as F

class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args
        self.hidden_com_dim = self.args.rnn_hidden_dim // self.args.n_agents

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        
        
        self.encode = nn.Linear(input_shape, self.hidden_com_dim)
        self.decode = nn.Linear(self.hidden_com_dim * self.args.n_agents + args.rnn_hidden_dim, args.rnn_hidden_dim)

    #def init_hidden(self):
    #    # make hidden states on same device as model
    #    return self.fc1.weight.new(1, self.hidden_com_dim).zero_()
    
    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
    
    #def init_hidden_com(self):
        ##make hidden states on same device as model
        #return self.fc1.weight.new(1, self.args.hidden_com_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        
        x_com = F.relu(self.encode(inputs))
        x_com_get = x_com.unsqueeze(-1).expand(-1, -1, self.args.n_agents).reshape(-1, self.hidden_com_dim*self.args.n_agents)
        
        x = th.cat([x, x_com_get], dim=-1)
        x = F.relu(self.decode(x))
        
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h
    

