import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from models.implicit_agent import ImplicitAgent, layer_init


class TestImplicitAgent:
    """Test ImplicitAgent class."""
    
    def test_init(self, mock_envs):
        """Test agent initialization."""
        config = {'debug_mode': False}
        agent = ImplicitAgent(mock_envs, config)
        
        # Check basic attributes
        assert agent.debug_mode == False
        
        # Check network components
        assert hasattr(agent, 'network')
        assert hasattr(agent, 'actor')
        assert hasattr(agent, 'critic')
        
        # Check network architecture
        assert isinstance(agent.network, nn.Sequential)
        assert isinstance(agent.actor, nn.Linear)
        assert isinstance(agent.critic, nn.Linear)
        
        # Check output dimensions
        assert agent.actor.out_features == mock_envs.single_action_space.n
        assert agent.critic.out_features == 1
    
    def test_init_with_debug(self, mock_envs):
        """Test agent initialization with debug mode."""
        config = {'debug_mode': True}
        agent = ImplicitAgent(mock_envs, config)
        
        assert agent.debug_mode == True
    
    def test_parse_obs(self, mock_envs):
        """Test observation parsing."""
        config = {'debug_mode': False}
        agent = ImplicitAgent(mock_envs, config)
        
        # Create sample observation
        sample_obs = {
            "sam_image_embeddings": torch.randn(2, 256, 64, 64),
            "sam_pred_mask_prob": torch.rand(2, 256, 256)
        }
        
        parsed = agent.parse_obs(sample_obs)
        
        # Check output shape matches input embeddings
        expected_shape = sample_obs["sam_image_embeddings"].shape
        assert parsed.shape == expected_shape
        
        # Check that features are not all zeros (due to skip connection)
        assert not torch.allclose(parsed, torch.zeros_like(parsed))
    
    def test_parse_obs_mask_resizing(self, mock_envs):
        """Test that mask probabilities are correctly resized."""
        config = {'debug_mode': False}
        agent = ImplicitAgent(mock_envs, config)
        
        # Create observation with different mask and embedding sizes
        sample_obs = {
            "sam_image_embeddings": torch.randn(1, 256, 32, 32),  # Smaller spatial size
            "sam_pred_mask_prob": torch.rand(1, 512, 512)  # Larger mask
        }
        
        parsed = agent.parse_obs(sample_obs)
        
        # Check that output matches embedding spatial dimensions
        assert parsed.shape == (1, 256, 32, 32)
    
    def test_get_value(self, mock_envs):
        """Test value function computation."""
        config = {'debug_mode': False}
        agent = ImplicitAgent(mock_envs, config)
        
        sample_obs = {
            "sam_image_embeddings": torch.randn(3, 256, 64, 64),
            "sam_pred_mask_prob": torch.rand(3, 256, 256)
        }
        
        values = agent.get_value(sample_obs)
        
        # Check output shape
        batch_size = sample_obs["sam_image_embeddings"].shape[0]
        expected_shape = (batch_size, 1)
        assert values.shape == expected_shape
        
        # Check that values are finite
        assert torch.all(torch.isfinite(values))
    
    def test_get_action_and_value(self, mock_envs):
        """Test action sampling and value computation."""
        config = {'debug_mode': False}
        agent = ImplicitAgent(mock_envs, config)
        
        sample_obs = {
            "sam_image_embeddings": torch.randn(2, 256, 64, 64),
            "sam_pred_mask_prob": torch.rand(2, 256, 256)
        }
        
        # Test without providing action
        action, log_prob, entropy, value = agent.get_action_and_value(sample_obs)
        
        batch_size = sample_obs["sam_image_embeddings"].shape[0]
        
        # Check shapes
        assert action.shape == (batch_size,)
        assert log_prob.shape == (batch_size,)
        assert entropy.shape == (batch_size,)
        assert value.shape == (batch_size, 1)
        
        # Check that actions are in valid range
        assert torch.all(action >= 0)
        assert torch.all(action < mock_envs.single_action_space.n)
        
        # Check that log probabilities are negative or zero
        assert torch.all(log_prob <= 0)
        
        # Check that entropy is non-negative
        assert torch.all(entropy >= 0)
    
    def test_get_action_and_value_with_action(self, mock_envs):
        """Test action evaluation with provided actions."""
        config = {'debug_mode': False}
        agent = ImplicitAgent(mock_envs, config)
        
        sample_obs = {
            "sam_image_embeddings": torch.randn(2, 256, 64, 64),
            "sam_pred_mask_prob": torch.rand(2, 256, 256)
        }
        
        # Provide specific actions
        test_actions = torch.tensor([10, 20])
        
        action, log_prob, entropy, value = agent.get_action_and_value(
            sample_obs, action=test_actions
        )
        
        # Check that returned actions match input
        assert torch.equal(action, test_actions)
        
        # Check other shapes
        batch_size = sample_obs["sam_image_embeddings"].shape[0]
        assert log_prob.shape == (batch_size,)
        assert entropy.shape == (batch_size,)
        assert value.shape == (batch_size, 1)
    
    def test_forward_pass_gradient_flow(self, mock_envs):
        """Test that gradients flow through the network."""
        config = {'debug_mode': False}
        agent = ImplicitAgent(mock_envs, config)
        agent.train()
        
        sample_obs = {
            "sam_image_embeddings": torch.randn(1, 256, 64, 64),
            "sam_pred_mask_prob": torch.rand(1, 256, 256)
        }
        
        # Forward pass
        action, log_prob, entropy, value = agent.get_action_and_value(sample_obs)
        
        # Compute dummy loss
        loss = -(log_prob.mean() + value.mean() + entropy.mean())
        loss.backward()
        
        # Check that gradients exist for all parameters
        for name, param in agent.named_parameters():
            assert param.grad is not None, f"Gradient not computed for {name}"
            assert not torch.allclose(param.grad, torch.zeros_like(param.grad)), \
                f"Zero gradient for {name}"
    
    def test_network_architecture_shapes(self, mock_envs):
        """Test that network layers have correct input/output dimensions."""
        config = {'debug_mode': False}
        agent = ImplicitAgent(mock_envs, config)
        
        # Test with known input size
        test_input = torch.randn(1, 256, 64, 64)
        
        # Should not raise errors
        output = agent.network(test_input)
        
        # Check output shape before actor/critic
        assert output.shape == (1, 512)  # Final linear layer output
    
    def test_different_batch_sizes(self, mock_envs):
        """Test agent with different batch sizes."""
        config = {'debug_mode': False}
        agent = ImplicitAgent(mock_envs, config)
        
        for batch_size in [1, 2, 4, 8]:
            sample_obs = {
                "sam_image_embeddings": torch.randn(batch_size, 256, 64, 64),
                "sam_pred_mask_prob": torch.rand(batch_size, 256, 256)
            }
            
            action, log_prob, entropy, value = agent.get_action_and_value(sample_obs)
            
            # Check shapes scale with batch size
            assert action.shape == (batch_size,)
            assert log_prob.shape == (batch_size,)
            assert entropy.shape == (batch_size,)
            assert value.shape == (batch_size, 1)
    
    def test_eval_mode(self, mock_envs):
        """Test agent in evaluation mode."""
        config = {'debug_mode': False}
        agent = ImplicitAgent(mock_envs, config)
        agent.eval()
        
        sample_obs = {
            "sam_image_embeddings": torch.randn(2, 256, 64, 64),
            "sam_pred_mask_prob": torch.rand(2, 256, 256)
        }
        
        with torch.no_grad():
            action, log_prob, entropy, value = agent.get_action_and_value(sample_obs)
        
        # Should work fine in eval mode
        assert action.shape == (2,)
        assert log_prob.shape == (2,)
        assert entropy.shape == (2,)
        assert value.shape == (2, 1)
    
    def test_state_dict_save_load(self, mock_envs):
        """Test saving and loading state dict."""
        config = {'debug_mode': False}
        agent1 = ImplicitAgent(mock_envs, config)
        agent2 = ImplicitAgent(mock_envs, config)
        
        # Save state from first agent
        state_dict = agent1.state_dict()
        
        # Modify second agent to have different weights
        with torch.no_grad():
            agent2.actor.weight.fill_(999.0)
            agent2.critic.weight.fill_(999.0)
        
        # Load state into second agent
        agent2.load_state_dict(state_dict)
        
        # Check that weights are now the same
        for name, param1 in agent1.named_parameters():
            param2 = dict(agent2.named_parameters())[name]
            assert torch.allclose(param1, param2), f"Parameters {name} not equal after load"