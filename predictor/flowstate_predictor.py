import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "forecasting_bench", "granite-tsfm")))

from tsfm_public import FlowStateForPrediction
from notebooks.hfdemo.flowstate.gift_wrapper import FlowState_Gift_Wrapper



def LoadFlowState(model_name, prediction_length,context_length, target_dim, freq, batch_size,device='cuda', domain=None, nd=False):
    flowstate = FlowStateForPrediction.from_pretrained(model_name).to(device)

    config = flowstate.config
    config.min_context = 0
    config.device = device
    flowstate = FlowState_Gift_Wrapper(flowstate, prediction_length, context_length=context_length, n_ch=target_dim, batch_size=batch_size, 
                                 f=freq, device=device, domain=domain, no_daily=nd)
    return flowstate




    




