REGISTRY = {}

from .basic_controller import BasicMAC
from .basic_controller_policy import BasicMAC as PolicyMAC
from .central_basic_controller import CentralBasicMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["policy"] = PolicyMAC
REGISTRY["basic_central_mac"] = CentralBasicMAC

from .basic_controller_teach import BasicMAC as BasicMAC_teach
REGISTRY["basic_mac_teach"] = BasicMAC_teach

from .central_basic_controller_teach import CentralBasicMAC as CentralBasicMAC_teach
REGISTRY["basic_central_mac_teach"] = CentralBasicMAC_teach

from .basic_controller_bootstrap import BasicMAC as BasicMAC_bootstrap
REGISTRY["basic_mac_bootstrap"] = BasicMAC_bootstrap

from .basic_controller_teach_2 import BasicMAC as BasicMAC_teach_2
REGISTRY["basic_mac_teach_2"] = BasicMAC_teach_2

from .basic_controller_explore import BasicMAC as BasicMAC_explore
REGISTRY["basic_mac_explore"] = BasicMAC_explore

from .basic_controller_communicate import BasicMAC as BasicMAC_communicate
REGISTRY["basic_mac_communicate"] = BasicMAC_communicate

from .basic_controller_all import BasicMAC_all
REGISTRY["basic_mac_all"] = BasicMAC_all

from .ppo_controller import PPOMAC
REGISTRY["ppo_mac"] = PPOMAC

from .ppo_controller_communicate import PPOMAC as PPOMAC_communicate
REGISTRY["ppo_mac_communicate"] = PPOMAC_communicate

from .ppo_controller_all import PPOMAC as PPOMAC_all
REGISTRY["ppo_mac_all"] = PPOMAC_all

from .basic_controller_iqn import BasicMAC as BasicMAC_iqn
REGISTRY["basic_mac_iqn"] = BasicMAC_iqn

from .basic_controller_iqn_communicate import BasicMAC as BasicMAC_iqn_communicate
REGISTRY["basic_mac_iqn_communicate"] = BasicMAC_iqn_communicate

from .basic_controller_iqn_all import BasicMAC as BasicMAC_iqn_all
REGISTRY["basic_mac_iqn_all"] = BasicMAC_iqn_all

from .basic_controller_avalible_action import BasicMAC as BasicMAC_avalible_action
REGISTRY["basic_mac_avalible_action"] = BasicMAC_avalible_action

from .basic_controller_iqn_avalible_action import BasicMAC as BasicMAC_iqn_avalible_action
REGISTRY["basic_mac_iqn_avalible_action"] = BasicMAC_iqn_avalible_action
