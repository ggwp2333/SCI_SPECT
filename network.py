import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(64, 64), output_dim=None):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = h
        if output_dim is not None:
            layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ActorCriticNet(nn.Module):
    def __init__(self, use_step_feature: bool = True, hidden_dims=(64, 64), n_actions: int = 54, max_steps: int = 22):
        super().__init__()
        self.use_step_feature = use_step_feature
        self.max_steps = float(max_steps)
        input_dim = n_actions + (1 if use_step_feature else 0)

        # shared MLP
        self.trunk = MLP(input_dim, hidden_dims, output_dim=hidden_dims[-1])
        feat_dim = hidden_dims[-1]

        # actor-critic heads
        self.policy_head = nn.Linear(feat_dim, n_actions)
        self.value_head = nn.Linear(feat_dim, 1)

    def forward(self, action_map: torch.Tensor, step_t: torch.Tensor | None = None):
        B = action_map.shape[0]
        x = action_map.view(B, -1).float()

        if self.use_step_feature:
            if step_t is None:
                raise ValueError("step_t must be provided when use_step_feature=True")
            step_feat = (step_t.float() / self.max_steps).unsqueeze(-1)
            x = torch.cat([x, step_feat], dim=-1)

        h = self.trunk(x)
        logits = self.policy_head(h)          
        values = self.value_head(h).squeeze(-1)  
        return logits, values

