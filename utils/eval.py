import torch 
import numpy as np

tf = lambda x: torch.FloatTensor(x).to(_dev[0])

def logq(path_list, actions_list, model, env, device):
    # TODO: this method is probably suboptimal, since it may repeat forward calls for
    # the same nodes.
    log_q = torch.tensor(1.0)
    for path, actions in zip(path_list, actions_list):
        path = path[::-1]
        actions = actions[::-1]
        path_obs = np.asarray([env.state2obs(state) for state in path])
        with torch.no_grad():
            # TODO: potentially mask invalid actions next_q
            logits_path = model(torch.FloatTensor(path_obs).to(device))
        logsoftmax = torch.nn.LogSoftmax(dim=1)
        logprobs_path = logsoftmax(logits_path)
        log_q_path = torch.tensor(0.0)
        for s, a, logprobs in zip(*[path, actions, logprobs_path]):
            log_q_path = log_q_path + logprobs[a]
        # Accumulate log prob of path
        if torch.le(log_q, 0.0):
            log_q = torch.logaddexp(log_q, log_q_path)
        else:
            log_q = log_q_path
    return log_q.item()