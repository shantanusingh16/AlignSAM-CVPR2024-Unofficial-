"""Integration tests for the full pipeline."""

import pytest
import torch
import numpy as np
import tempfile
import os
import yaml
from unittest.mock import Mock, patch
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from models import make_agent
from custom_gym_implns.envs.sam_seg_env import SamSegEnv
from datasets.coco_dataset import CocoDataset


@pytest.mark.integration
class TestFullPipeline:
    """Integration tests for the complete training pipeline."""
    
    @pytest.fixture
    def integration_config(self):
        """Configuration for integration testing."""
        return {
            'agent_config': {
                'type': 'explicit',
                'clip_model_name': 'CS-ViT-B/16',
                'clip_image_size': [224, 224],
                'clip_text_prompt': ['person', 'car'],
                'similarity_scale_temperature': 0.33,
                'debug_mode': False
            },
            'env_config': {
                'img_shape': [100, 100, 3],
                'embedding_shape': [256, 16, 16],
                'mask_shape': [100, 100],
                'render_frame_shape': [80, 80],
                'max_steps': 3,
                'target_categories': ['person', 'car'],
                'dataset_config': {
                    'type': 'coco',
                    'data_dir': 'test_data',
                    'data_type': 'val2017',
                    'seed': 42
                },
                'sam_ckpt_fp': 'test_sam.pt',
                'num_patches': 4,
                'penalize_for_wrong_input': False,
                'use_dice_score': True,
                'render_mode': 'rgb_array'
            }
        }
    
    def test_agent_environment_interaction(self, integration_config, mock_dependencies):
        """Test agent-environment interaction loop."""
        # Create mock environments wrapper
        mock_envs = Mock()
        mock_envs.single_action_space = Mock()
        mock_envs.single_action_space.n = 32  # 4x4 patches * 2 labels
        mock_envs.envs = [Mock()]
        mock_envs.envs[0].env = Mock()
        mock_envs.envs[0].env.max_steps = 3
        
        # Create agent
        agent = make_agent(integration_config['agent_config'], mock_envs)
        
        # Create environment
        env = SamSegEnv(**integration_config['env_config'])
        
        # Run interaction loop
        obs, _ = env.reset()
        
        for step in range(3):
            # Convert observation to batch format for agent
            batch_obs = {}
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    batch_obs[key] = torch.from_numpy(value).unsqueeze(0)
                elif isinstance(value, str):
                    batch_obs[key] = [value]
                else:
                    batch_obs[key] = torch.tensor([value])
            
            # Get action from agent
            with torch.no_grad():
                action, log_prob, entropy, value = agent.get_action_and_value(batch_obs)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action.item())
            
            # Check outputs are valid
            assert isinstance(reward, (int, float, np.number))
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert np.isfinite(reward)
            
            if terminated or truncated:
                break
        
        # Test value function
        final_value = agent.get_value(batch_obs)
        assert final_value.shape == (1, 1)
        assert torch.isfinite(final_value).all()
    
    def test_implicit_agent_environment_interaction(self, integration_config, mock_dependencies):
        """Test implicit agent with environment."""
        # Modify config for implicit agent
        implicit_config = integration_config.copy()
        implicit_config['agent_config'] = {
            'type': 'implicit',
            'debug_mode': False
        }
        
        # Create mock environments wrapper
        mock_envs = Mock()
        mock_envs.single_action_space = Mock()
        mock_envs.single_action_space.n = 32
        mock_envs.envs = [Mock()]
        mock_envs.envs[0].env = Mock()
        mock_envs.envs[0].env.max_steps = 3
        
        # Create agent and environment
        agent = make_agent(implicit_config['agent_config'], mock_envs)
        env = SamSegEnv(**integration_config['env_config'])
        
        # Run brief interaction
        obs, _ = env.reset()
        
        # Convert to batch format
        batch_obs = {
            'sam_image_embeddings': torch.from_numpy(obs['sam_image_embeddings']).unsqueeze(0),
            'sam_pred_mask_prob': torch.from_numpy(obs['sam_pred_mask_prob']).unsqueeze(0)
        }
        
        # Get action from agent
        with torch.no_grad():
            action, log_prob, entropy, value = agent.get_action_and_value(batch_obs)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action.item())
        
        # Check outputs
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    
    def test_batch_processing(self, integration_config, mock_dependencies):
        """Test processing batches of observations."""
        # Create mock environments wrapper
        mock_envs = Mock()
        mock_envs.single_action_space = Mock()
        mock_envs.single_action_space.n = 32
        mock_envs.envs = [Mock()]
        mock_envs.envs[0].env = Mock()
        mock_envs.envs[0].env.max_steps = 3
        
        agent = make_agent(integration_config['agent_config'], mock_envs)
        
        # Create batch of observations
        batch_size = 4
        batch_obs = {
            'image': torch.randint(0, 255, (batch_size, 100, 100, 3), dtype=torch.uint8),
            'target_category': ['person', 'car', 'person', 'car'],
            'sam_image_embeddings': torch.randn(batch_size, 256, 16, 16),
            'sam_pred_mask_prob': torch.rand(batch_size, 100, 100),
            'num_steps': torch.tensor([0, 1, 2, 0])
        }
        
        # Process batch
        with torch.no_grad():
            actions, log_probs, entropies, values = agent.get_action_and_value(batch_obs)
        
        # Check batch outputs
        assert actions.shape == (batch_size,)
        assert log_probs.shape == (batch_size,)
        assert entropies.shape == (batch_size,)
        assert values.shape == (batch_size, 1)
        
        # Check all outputs are finite
        assert torch.isfinite(actions).all()
        assert torch.isfinite(log_probs).all()
        assert torch.isfinite(entropies).all()
        assert torch.isfinite(values).all()
    
    @pytest.mark.slow
    def test_training_step_simulation(self, integration_config, mock_dependencies):
        """Simulate a training step with gradient computation."""
        # Create mock environments wrapper
        mock_envs = Mock()
        mock_envs.single_action_space = Mock()
        mock_envs.single_action_space.n = 32
        mock_envs.envs = [Mock()]
        mock_envs.envs[0].env = Mock()
        mock_envs.envs[0].env.max_steps = 3
        
        agent = make_agent(integration_config['agent_config'], mock_envs)
        agent.train()
        
        # Simulate training batch
        batch_size = 8
        batch_obs = {
            'image': torch.randint(0, 255, (batch_size, 100, 100, 3), dtype=torch.uint8),
            'target_category': ['person'] * batch_size,
            'sam_image_embeddings': torch.randn(batch_size, 256, 16, 16),
            'sam_pred_mask_prob': torch.rand(batch_size, 100, 100),
            'num_steps': torch.randint(0, 3, (batch_size,))
        }
        
        # Forward pass
        actions, log_probs, entropies, values = agent.get_action_and_value(batch_obs)
        
        # Simulate PPO loss computation
        advantages = torch.randn(batch_size)  # Mock advantages
        returns = torch.randn(batch_size, 1)   # Mock returns
        old_log_probs = torch.randn(batch_size) # Mock old log probs
        
        # Policy loss (simplified PPO)
        ratio = torch.exp(log_probs - old_log_probs)
        policy_loss = -torch.min(ratio * advantages, 
                                torch.clamp(ratio, 0.9, 1.1) * advantages).mean()
        
        # Value loss
        value_loss = ((values.squeeze() - returns.squeeze()) ** 2).mean()
        
        # Entropy loss
        entropy_loss = -entropies.mean()
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
        
        # Backward pass
        total_loss.backward()
        
        # Check gradients exist
        has_gradients = False
        for param in agent.parameters():
            if param.requires_grad and param.grad is not None:
                has_gradients = True
                break
        
        assert has_gradients, "No gradients computed during training step"
        
        # Check loss is finite
        assert torch.isfinite(total_loss), "Loss is not finite"
    
    def test_config_consistency(self, temp_config_dir):
        """Test consistency between loaded configs and created objects."""
        # Load configs from files
        agent_config_path = os.path.join(temp_config_dir, 'agents', 'test_agent.yaml')
        env_config_path = os.path.join(temp_config_dir, 'envs', 'test_env.yaml')
        
        with open(agent_config_path, 'r') as f:
            agent_config = yaml.safe_load(f)
        
        with open(env_config_path, 'r') as f:
            env_config = yaml.safe_load(f)
        
        # Create mock environments wrapper
        mock_envs = Mock()
        mock_envs.single_action_space = Mock()
        mock_envs.single_action_space.n = 32
        mock_envs.envs = [Mock()]
        mock_envs.envs[0].env = Mock()
        mock_envs.envs[0].env.max_steps = env_config['max_steps']
        
        # Create objects
        with patch('datasets.get_dataset'), \
             patch('custom_gym_implns.envs.utils.repvit_sam_wrapper.RepVITSamWrapper'):
            
            agent = make_agent(agent_config, mock_envs)
            env = SamSegEnv(**env_config)
        
        # Check consistency
        assert agent.similarity_scale_temperature == agent_config['similarity_scale_temperature']
        assert agent.debug_mode == agent_config['debug_mode']
        assert env.max_steps == env_config['max_steps']
        assert env.target_categories == env_config['target_categories']
    
    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_gpu_compatibility(self, integration_config, mock_dependencies):
        """Test GPU device compatibility."""
        device = torch.device('cuda')
        
        # Create mock environments wrapper
        mock_envs = Mock()
        mock_envs.single_action_space = Mock()
        mock_envs.single_action_space.n = 32
        mock_envs.envs = [Mock()]
        mock_envs.envs[0].env = Mock()
        mock_envs.envs[0].env.max_steps = 3
        
        agent = make_agent(integration_config['agent_config'], mock_envs)
        agent.to(device)
        
        # Create GPU batch
        batch_obs = {
            'image': torch.randint(0, 255, (2, 100, 100, 3), dtype=torch.uint8).to(device),
            'target_category': ['person', 'car'],
            'sam_image_embeddings': torch.randn(2, 256, 16, 16).to(device),
            'sam_pred_mask_prob': torch.rand(2, 100, 100).to(device),
            'num_steps': torch.tensor([0, 1]).to(device)
        }
        
        # Forward pass on GPU
        with torch.no_grad():
            actions, log_probs, entropies, values = agent.get_action_and_value(batch_obs)
        
        # Check outputs are on GPU
        assert actions.device.type == 'cuda'
        assert log_probs.device.type == 'cuda'
        assert entropies.device.type == 'cuda'
        assert values.device.type == 'cuda'
    
    def test_state_dict_save_load_consistency(self, integration_config, mock_dependencies):
        """Test state dict saving and loading preserves model behavior."""
        # Create mock environments wrapper
        mock_envs = Mock()
        mock_envs.single_action_space = Mock()
        mock_envs.single_action_space.n = 32
        mock_envs.envs = [Mock()]
        mock_envs.envs[0].env = Mock()
        mock_envs.envs[0].env.max_steps = 3
        
        # Create two identical agents
        agent1 = make_agent(integration_config['agent_config'], mock_envs)
        agent2 = make_agent(integration_config['agent_config'], mock_envs)
        
        # Get initial outputs from agent1
        batch_obs = {
            'image': torch.randint(0, 255, (1, 100, 100, 3), dtype=torch.uint8),
            'target_category': ['person'],
            'sam_image_embeddings': torch.randn(1, 256, 16, 16),
            'sam_pred_mask_prob': torch.rand(1, 100, 100),
            'num_steps': torch.tensor([0])
        }
        
        with torch.no_grad():
            action1, log_prob1, entropy1, value1 = agent1.get_action_and_value(batch_obs)
        
        # Save and load state dict
        state_dict = agent1.state_dict()
        agent2.load_state_dict(state_dict, strict=False)
        
        # Get outputs from agent2 after loading
        with torch.no_grad():
            action2, log_prob2, entropy2, value2 = agent2.get_action_and_value(batch_obs)
        
        # Outputs should be identical (within numerical precision)
        assert torch.allclose(log_prob1, log_prob2, atol=1e-6)
        assert torch.allclose(entropy1, entropy2, atol=1e-6)
        assert torch.allclose(value1, value2, atol=1e-6)
        # Note: actions might differ due to sampling, so we don't check them