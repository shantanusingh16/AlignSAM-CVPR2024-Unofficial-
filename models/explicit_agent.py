import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
from torchvision.transforms.functional import to_pil_image

from CLIP_Surgery import clip
from CLIP_Surgery.clip.clip_model import ResidualAttentionBlock

import mlflow
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ExplicitAgent(nn.Module):
    def __init__(self, envs, agent_cfg):
        super().__init__()

        self.setup_clip(
            agent_cfg['clip_model_name'], 
            agent_cfg['clip_image_size'],
            agent_cfg['clip_text_prompt']
        )

        self.similarity_scale_temperature = agent_cfg['similarity_scale_temperature']
        self.debug_mode = agent_cfg.get('debug_mode', False)

        self.sam_network = nn.Sequential(
            layer_init(nn.Conv2d(256, 128, 3, stride=2, padding=1, padding_mode='zeros')), # (b, 256, 64, 64)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            layer_init(nn.Conv2d(128, 64, 3, stride=2, padding=1, padding_mode='zeros')), # (b, 128, 32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 5, stride=3)), # (b, 64, 16, 16)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(), # (b, 1024)
        )

        self.clip_network = nn.Sequential(
            layer_init(nn.Conv2d(len(self.clip_text_prompt), 16, 3, stride=2, padding=1, padding_mode='zeros')), # (b, num_prompts, 64, 64)
            nn.BatchNorm2d(16),
            nn.ReLU(),  
            layer_init(nn.Conv2d(16, 32, 3, stride=2, padding=1, padding_mode='zeros')), # (b, 16, 32, 32)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 5, stride=3)), # (b, 64, 4, 4)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(), # (b, 1024)
        )

        self.combined_attention = ResidualAttentionBlock(
            d_model=2048,
            n_head=4,
            attn_mask=None,
            need_weights=False
        ) 

        self.head = nn.Sequential(
            layer_init(nn.Linear(2048, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
        )

        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

        def get_base_env(env):
            while hasattr(env, 'env'):
                env = env.env
            return env
        
        base_env = get_base_env(envs.envs[0])
        self.max_steps = base_env.max_steps if hasattr(base_env, 'max_steps') else None


    def setup_clip(self, clip_model_name, clip_image_size, clip_text_prompt):
        clip_model, _ = clip.load(clip_model_name, device='cpu')
        self.clip_model = clip_model
        self.clip_model.eval()

        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.clip_preprocess = Compose([
            Resize((clip_image_size[1], clip_image_size[0]), interpolation=InterpolationMode.BICUBIC), 
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

        self.clip_text_prompt = clip_text_prompt

        # Prompt ensemble for text features with normalization
        text_features = clip.encode_text_with_prompt_ensemble(self.clip_model, clip_text_prompt, 'cpu')
        self.clip_text_features = nn.Parameter(text_features, requires_grad=False)


        # Extract redundant features from an empty string
        redundant_features = clip.encode_text_with_prompt_ensemble(self.clip_model, [""], 'cpu')
        self.clip_redundant_features = nn.Parameter(redundant_features, requires_grad=False)
        

    def get_clip_surgery_features(self, obs):
        with torch.no_grad():
            obs_image = obs["image"] # (b, h, w, c)
            embedding_shape = tuple(obs["sam_image_embeddings"].size()) # (b, c, h, w)
            
            image = obs_image.float().permute(0, 3, 1, 2) / 255.0 # (b, h, w, c) -> (b, c, h, w)
            image = self.clip_preprocess(image)

            image_features = self.clip_model.encode_image(image)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)

            # Apply feature surgery
            similarity = clip.clip_feature_surgery(image_features, 
                                                self.clip_text_features, 
                                                self.clip_redundant_features)
            similarity_map = clip.get_similarity_map(similarity[:, 1:, :], 
                                                    (embedding_shape[2], embedding_shape[3]))
            
            similarity_map = similarity_map.permute(0, 3, 1, 2) # (b, h, w, c) -> (b, c, h, w)

            # Scale the similarity map
            channel_scale_map = torch.zeros(len(self.clip_text_prompt), 
                                            dtype=torch.float32, 
                                            device=similarity_map.device,
                                            requires_grad=False)
            for batch_idx, cat in enumerate(obs["target_category"]):
                if cat in self.clip_text_prompt:
                    cat_idx = self.clip_text_prompt.index(cat)
                    channel_scale_map[cat_idx] = 1 / self.similarity_scale_temperature
                else:
                    continue # Skip, since this would just scale down values of all channels
                
                # Use soft-max style normalization
                channel_scale_map = torch.exp(channel_scale_map - torch.logsumexp(channel_scale_map, dim=0))
                
                # Add offset to boost the highest channel to scale to 1
                channel_scale_map += 1 - channel_scale_map.max()

                similarity_map[batch_idx] *= channel_scale_map.view(-1, 1, 1)  # Scale each channel

            if self.debug_mode:
                # Debugging: visualize the similarity map
                ncols = len(self.clip_text_prompt) + 1
                nrows = min(similarity_map.size(0), 2)  # Show at most 2 rows
                fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(5*ncols, 3*nrows))
                for i in range(nrows):
                    ax[i, 0].imshow(obs["image"][i].cpu().numpy().astype(np.uint8))
                    ax[i, 0].set_title(obs["target_category"][i])
                    for j, cat in enumerate(self.clip_text_prompt):
                        sim_map = similarity_map[i, j].cpu().numpy()
                        ax[i, j+1].imshow(sim_map, cmap='hot', vmin=0, vmax=1)
                        ax[i, j+1].set_title(cat)
                plt.tight_layout()
                mlflow.log_figure(fig, f"similarity_map_{i}.png")
                plt.close(fig)

            return similarity_map


    def get_sam_features(self, obs):
        sam_image_embeddings = obs["sam_image_embeddings"] # (b, c, h, w)
        sam_pred_mask_prob = obs["sam_pred_mask_prob"].unsqueeze(dim=1)  # (b, 1, h, w)

        embedding_shape = tuple(sam_image_embeddings.size())
        resized_sam_mask_prob = nn.functional.interpolate(
            sam_pred_mask_prob, size=(embedding_shape[2], embedding_shape[3]), 
            mode="bilinear", align_corners=False)

        # Rescale the mask probabilities based on number of steps
        if ("num_steps" in obs) and self.max_steps is not None:
            eps = 1e-6
            temp = torch.exp(self.max_steps - obs["num_steps"])  # Exponential decay
            p = torch.clamp(resized_sam_mask_prob, min=eps, max=1-eps)
            logit = torch.logit(p)
            resized_sam_mask_prob = torch.sigmoid(logit / temp.view(-1, 1, 1, 1))

        if self.debug_mode:
            # Debugging: visualize the sam embedding and prob map

            fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(6, 6))
            ax = ax.flatten()
            for i in range(2):
                ax[2*i].imshow(obs["image"][i].cpu().numpy().astype(np.uint8))
                ax[2*i].set_title(obs["target_category"][i])
                prob_map = resized_sam_mask_prob[i, 0].cpu().numpy()
                ax[2*i + 1].imshow(prob_map, cmap='hot', vmin=0, vmax=1)
                ax[2*i +1].set_title("Steps: {}".format(obs.get("num_steps", ["NA", "NA"])[i]))
            plt.tight_layout()
            plt.axis('off')
            mlflow.log_figure(fig, f"sam_prob_map{i}.png")
            plt.close(fig)
        
        resized_sam_mask_prob = resized_sam_mask_prob.repeat(1, embedding_shape[1], 1, 1)
        x = sam_image_embeddings * resized_sam_mask_prob 
        x += sam_image_embeddings # skip connection

        return x
    

    def merge_clip_sam_features(self, obs):
        x_sam = self.get_sam_features(obs)
        x_clip = self.get_clip_surgery_features(obs)
        hidden_x = self.sam_network(x_sam)
        hidden_clip = self.clip_network(x_clip)
        combined_hidden = torch.cat([hidden_x, hidden_clip], dim=1)
        out = self.combined_attention(combined_hidden)
        return out


    def get_value(self, obs):
        x = self.merge_clip_sam_features(obs)
        hidden = self.head(x)
        return self.critic(hidden)


    def get_action_and_value(self, obs, action=None):
        x = self.merge_clip_sam_features(obs)
        hidden = self.head(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
