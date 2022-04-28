REGISTRY = {}

from .rnn_agent import RNNAgent
from .ff_agent import FFAgent
from .central_rnn_agent import CentralRNNAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["ff"] = FFAgent

REGISTRY["central_rnn"] = CentralRNNAgent

from .rnn_agent_all import RNNAgent as RNNAgent_all
REGISTRY["rnn_all"] = RNNAgent_all

from .rnn_agent_bootstrap import RNNAgent as RNNAgent_bootstrap
REGISTRY["rnn_bootstrap"] = RNNAgent_bootstrap

from .central_rnn_agent_big import CentralRNNAgent as CentralRNNAgent_big
REGISTRY["central_rnn_big"] = CentralRNNAgent_big

from .rnn_agent_explore import RNNAgent as RNNAgent_explore
REGISTRY["rnn_explore"] = RNNAgent_explore

from .rnn_agent_communicate import RNNAgent as RNNAgent_communicate
REGISTRY["rnn_communicate"] = RNNAgent_communicate


from .rnn_ppo_agent import RNNPPOAgent
REGISTRY["rnn_ppo"] = RNNPPOAgent

from .rnn_ppo_agent_communicate import RNNPPOAgent as RNNPPOAgent_communicate
REGISTRY["rnn_ppo_communicate"] = RNNPPOAgent_communicate

from .iqn_rnn_agent import IQNRNNAgent
REGISTRY["iqn_rnn"] = IQNRNNAgent

from .iqn_rnn_agent_communicate import IQNRNNAgent as IQNRNNAgent_communicate
REGISTRY["iqn_rnn_communicate"] = IQNRNNAgent_communicate


from .rnn_agent_communicate2 import RNNAgent as RNNAgent_communicate2
REGISTRY["rnn_communicate2"] = RNNAgent_communicate2
