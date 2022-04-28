from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .max_q_learner import MAXQLearner
from .max_q_learner_ddpg import DDPGQLearner
from .max_q_learner_sac import SACQLearner
from .q_learner_w import QLearner as WeightedQLearner
from .qatten_learner import QattenLearner
from .dmaq_qatten_learner import DMAQ_qattenLearner
from .max_q_learner_teach import MAXQLearner as MAXQLearner_teach
from .max_q_learner_teach_add import MAXQLearner as MAXQLearner_teach_add
from .max_q_learner_addpolicy import MAXQLearner as MAXQLearner_addpolicy
from .max_q_learner_bootstrap import MAXQLearner as MAXQLearner_bootstrap
from .max_q_learner_low import MAXQLearner as MAXQLearner_low
from .max_q_learner_head import MAXQLearner as MAXQLearner_head
from .max_q_learner_head_2 import MAXQLearner as MAXQLearner_head_2
from .max_q_learner_head_little import MAXQLearner as MAXQLearner_head_little
from .dmaq_qatten_learner_ctde import DMAQ_qattenLearner as DMAQ_qattenLearner_ctde
from .q_learner_ctde_bayes import MAXQLearner as MAXQLearner_ctde_bayes
from .dmaq_qatten_learner_ctde_bayes import DMAQ_qattenLearner as DMAQ_qattenLearner_ctde_bayes
REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["w_q_learner"] = WeightedQLearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["max_q_learner"] = MAXQLearner
REGISTRY["sac"] = SACQLearner
REGISTRY["ddpg"] = DDPGQLearner
REGISTRY["qatten_learner"] = QattenLearner
REGISTRY["dmaq_qatten_learner"] = DMAQ_qattenLearner
REGISTRY["max_q_learner_teach"] = MAXQLearner_teach
REGISTRY["max_q_learner_teach_add"] = MAXQLearner_teach_add
REGISTRY["max_q_learner_addpolicy"] = MAXQLearner_addpolicy
REGISTRY["max_q_learner_bootstrap"] = MAXQLearner_bootstrap
REGISTRY["max_q_learner_low"] = MAXQLearner_low
REGISTRY["max_q_learner_head"] = MAXQLearner_head
REGISTRY["max_q_learner_head_2"] = MAXQLearner_head_2
REGISTRY["max_q_learner_head_little"] = MAXQLearner_head_little
REGISTRY["dmaq_qatten_learner_ctde"] = DMAQ_qattenLearner_ctde
REGISTRY["dmaq_qatten_learner_ctde_bayes"] = DMAQ_qattenLearner_ctde_bayes

#from .max_q_learner_explore import MAXQLearner as MAXQLearner_explore
#REGISTRY["max_q_learner_explore"] = MAXQLearner_explore

from .q_learner_low import QLearner as QLearner_low
REGISTRY["q_learner_low"] = QLearner_low

from .q_learner_low2 import QLearner as QLearner_low2
REGISTRY["q_learner_low2"] = QLearner_low2

from .q_learner_low3_orig import QLearner as QLearner_low3
REGISTRY["q_learner_low3"] = QLearner_low3

from .q_learner_teach import QLearner as QLearner_teach
REGISTRY["q_learner_teach"] = QLearner_teach

from .q_learner_teach_bijiao import QLearner as QLearner_teach_bijiao
REGISTRY["q_learner_teach_bijiao"] = QLearner_teach_bijiao


from .dmaq_qatten_learner_teach import DMAQ_qattenLearner as DMAQ_qattenLearner_teach
REGISTRY["dmaq_qatten_learner_teach"] = DMAQ_qattenLearner_teach

from .dmaq_qatten_learner_teach_bijiao import DMAQ_qattenLearner as DMAQ_qattenLearner_teach_bijiao 
REGISTRY["dmaq_qatten_learner_teach_bijiao"] = DMAQ_qattenLearner_teach_bijiao 

from .policy_gradient_v2 import PGLearner_v2
REGISTRY["policy_gradient_v2"] = PGLearner_v2

from .policy_gradient_v2_teach import PGLearner_v2 as PGLearner_v2_teach
REGISTRY["policy_gradient_v2_teach"] = PGLearner_v2_teach


from .offpg_learner import OffPGLearner
REGISTRY["offpg_learner"] = OffPGLearner


from .iqn_learner import IQNLearner
REGISTRY["iqn_learner"] = IQNLearner

from .iqn_learner_teach import IQNLearner as IQNLearner_teach
REGISTRY["iqn_learner_teach"] = IQNLearner_teach


from .q_learner_teach_edit_p import QLearner as QLearner_teach_edit_p
REGISTRY["q_learner_teach_edit_p"] = QLearner_teach_edit_p
