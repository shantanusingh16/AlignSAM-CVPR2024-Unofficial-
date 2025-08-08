import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ImplicitAgent(nn.Module):
    def __init__(self, envs, agent_cfg):
        super().__init__()
        self.debug_mode = agent_cfg.get('debug_mode', False)
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(256, 64, 8, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 128, 5, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(128, 256, 3, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(256, 512, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(512 * 4 * 4, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def parse_obs(self, obs):
        sam_image_embeddings = obs["sam_image_embeddings"] # (b, c, h, w)
        sam_pred_mask_prob = obs["sam_pred_mask_prob"].unsqueeze(dim=1)  # (b, 1, h, w)

        embedding_shape = tuple(sam_image_embeddings.size())
        resized_sam_mask_prob = nn.functional.interpolate(
            sam_pred_mask_prob, size=(embedding_shape[2], embedding_shape[3]), 
            mode="bilinear", align_corners=False)
        resized_sam_mask_prob = resized_sam_mask_prob.repeat(1, embedding_shape[1], 1, 1)

        x = sam_image_embeddings * resized_sam_mask_prob 
        x += sam_image_embeddings # skip connection
        return x

    def get_value(self, obs):
        x = self.parse_obs(obs)
        hidden = self.network(x)
        return self.critic(hidden)

    def get_action_and_value(self, obs, action=None):
        x = self.parse_obs(obs)
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
