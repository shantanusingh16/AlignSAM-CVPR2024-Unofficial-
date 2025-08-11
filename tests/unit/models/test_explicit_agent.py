import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from models.explicit_agent import ExplicitAgent, layer_init


class TestLayerInit:
    """Test layer initialization utility."""
    
    def test_layer_init_linear(self):
        """Test layer initialization for Linear layers."""
        layer = nn.Linear(10, 5)
        initialized_layer = layer_init(layer, std=1.0, bias_const=0.1)
        
        # Check that it returns the same layer
        assert initialized_layer is layer
        
        # Check that weights and bias are initialized
        assert layer.weight.data is not None
        assert layer.bias.data is not None
        
        # Check bias initialization
        assert torch.allclose(layer.bias.data, torch.tensor(0.1))
    
    def test_layer_init_conv2d(self):
        """Test layer initialization for Conv2d layers."""
        layer = nn.Conv2d(3, 16, 3)
        initialized_layer = layer_init(layer, std=0.5, bias_const=-0.1)
        
        assert initialized_layer is layer
        assert torch.allclose(layer.bias.data, torch.tensor(-0.1))


class TestExplicitAgent:
    """Test ExplicitAgent class."""
    
    def test_init(self, mock_envs, agent_config):
        """Test agent initialization."""
        agent = ExplicitAgent(mock_envs, agent_config)
        
        # Check basic attributes
        assert agent.similarity_scale_temperature == 0.33
        assert agent.debug_mode == False
        assert agent.max_steps == 5
        
        # Check network components
        assert hasattr(agent, 'sam_network')
        assert hasattr(agent, 'clip_network') 
        assert hasattr(agent, 'combined_attention')
        assert hasattr(agent, 'head')
        assert hasattr(agent, 'actor')
        assert hasattr(agent, 'critic')
        
        # Check CLIP setup
        assert hasattr(agent, 'clip_model')
        assert hasattr(agent, 'clip_preprocess')
        assert hasattr(agent, 'clip_text_prompt')
        assert hasattr(agent, 'clip_text_features')
        assert hasattr(agent, 'clip_redundant_features')
    
    def test_init_implicit_config(self, mock_envs):
        """Test agent initialization with implicit agent config."""
        config = {
            'type': 'implicit',
            'debug_mode': True
        }
        
        # This should not create an ExplicitAgent, but test error handling
        with pytest.raises((KeyError, AttributeError)):
            # Should fail because explicit agent requires CLIP config
            ExplicitAgent(mock_envs, config)
    
    def test_setup_clip(self, mock_envs, agent_config):
        """Test CLIP model setup."""
        agent = ExplicitAgent(mock_envs, agent_config)
        
        # Check that CLIP text prompts are set
        assert agent.clip_text_prompt == agent_config['clip_text_prompt']
        
        # Check that text features are parameters
        assert isinstance(agent.clip_text_features, nn.Parameter)
        assert isinstance(agent.clip_redundant_features, nn.Parameter)
        
        # Check shapes
        expected_num_prompts = len(agent_config['clip_text_prompt'])
        assert agent.clip_text_features.shape[0] == expected_num_prompts
        assert agent.clip_redundant_features.shape[0] == 1
    
    def test_get_clip_surgery_features(self, mock_envs, agent_config, sample_observation):
        """Test CLIP Surgery feature extraction."""
        agent = ExplicitAgent(mock_envs, agent_config)
        
        with torch.no_grad():
            features = agent.get_clip_surgery_features(sample_observation)
        
        # Check output shape
        batch_size = sample_observation["image"].shape[0]
        embedding_h, embedding_w = sample_observation["sam_image_embeddings"].shape[2:4]
        num_prompts = len(agent_config['clip_text_prompt'])
        
        expected_shape = (batch_size, num_prompts, embedding_h, embedding_w)
        assert features.shape == expected_shape
        
        # Check that features are in valid range (after similarity processing)
        assert torch.all(features >= 0)
        assert torch.all(features <= 1.5)  # Allow some headroom for scaled similarities
    
    def test_get_sam_features(self, mock_envs, agent_config, sample_observation):
        """Test SAM feature processing."""
        agent = ExplicitAgent(mock_envs, agent_config)
        
        features = agent.get_sam_features(sample_observation)
        
        # Check output shape (should match input embeddings)
        expected_shape = sample_observation["sam_image_embeddings"].shape
        assert features.shape == expected_shape
        
        # Check that features are not all zeros (skip connection should ensure this)
        assert not torch.allclose(features, torch.zeros_like(features))
    
    def test_get_sam_features_with_steps(self, mock_envs, agent_config, sample_observation):
        """Test SAM feature processing with step-based temperature scaling."""
        agent = ExplicitAgent(mock_envs, agent_config)
        
        # Add num_steps to observation
        obs_with_steps = sample_observation.copy()
        obs_with_steps["num_steps"] = torch.tensor([1, 3])
        
        features = agent.get_sam_features(obs_with_steps)
        
        # Check that features shape is preserved
        expected_shape = sample_observation["sam_image_embeddings"].shape
        assert features.shape == expected_shape
    
    def test_merge_clip_sam_features(self, mock_envs, agent_config, sample_observation):
        """Test feature fusion between CLIP and SAM."""
        agent = ExplicitAgent(mock_envs, agent_config)
        
        merged_features = agent.merge_clip_sam_features(sample_observation)
        
        # Check output dimensions
        batch_size = sample_observation["image"].shape[0] 
        expected_shape = (batch_size, 2048)  # Combined feature dimension
        assert merged_features.shape == expected_shape
    
    def test_get_value(self, mock_envs, agent_config, sample_observation):
        """Test value function computation."""
        agent = ExplicitAgent(mock_envs, agent_config)
        
        values = agent.get_value(sample_observation)
        
        # Check output shape
        batch_size = sample_observation["image"].shape[0]
        expected_shape = (batch_size, 1)
        assert values.shape == expected_shape
        
        # Check that values are finite
        assert torch.all(torch.isfinite(values))
    
    def test_get_action_and_value(self, mock_envs, agent_config, sample_observation):
        """Test action sampling and value computation."""
        agent = ExplicitAgent(mock_envs, agent_config)
        
        # Test without providing action
        action, log_prob, entropy, value = agent.get_action_and_value(sample_observation)
        
        batch_size = sample_observation["image"].shape[0]
        
        # Check shapes
        assert action.shape == (batch_size,)
        assert log_prob.shape == (batch_size,)
        assert entropy.shape == (batch_size,)
        assert value.shape == (batch_size, 1)
        
        # Check that actions are in valid range
        assert torch.all(action >= 0)
        assert torch.all(action < mock_envs.single_action_space.n)
        
        # Check that log probabilities are negative
        assert torch.all(log_prob <= 0)
        
        # Check that entropy is positive
        assert torch.all(entropy >= 0)
    
    def test_get_action_and_value_with_action(self, mock_envs, agent_config, sample_observation):
        """Test action evaluation with provided actions."""
        agent = ExplicitAgent(mock_envs, agent_config)
        
        # Provide specific actions
        batch_size = sample_observation["image"].shape[0]
        test_actions = torch.tensor([5, 10])[:batch_size]
        
        action, log_prob, entropy, value = agent.get_action_and_value(
            sample_observation, action=test_actions
        )
        
        # Check that returned actions match input
        assert torch.equal(action, test_actions)
        
        # Check other shapes
        assert log_prob.shape == (batch_size,)
        assert entropy.shape == (batch_size,)
        assert value.shape == (batch_size, 1)
    
    def test_state_dict_excludes_clip(self, mock_envs, agent_config):
        """Test that state dict excludes frozen CLIP weights."""
        agent = ExplicitAgent(mock_envs, agent_config)
        
        state_dict = agent.state_dict()
        
        # Check that no CLIP model keys are present
        clip_keys = [k for k in state_dict.keys() if k.startswith("clip_model.")]
        assert len(clip_keys) == 0
        
        # Check that other components are present
        expected_keys = ["sam_network", "clip_network", "combined_attention", "head", "actor", "critic"]
        for key in expected_keys:
            matching_keys = [k for k in state_dict.keys() if k.startswith(key)]
            assert len(matching_keys) > 0, f"No keys found for {key}"
    
    def test_load_state_dict_robustness(self, mock_envs, agent_config):
        """Test loading state dict with missing keys."""
        agent = ExplicitAgent(mock_envs, agent_config)
        
        # Create incomplete state dict
        partial_state_dict = {
            "actor.weight": torch.randn(64, 512),
            "actor.bias": torch.randn(64),
        }
        
        # Should not raise error with strict=False (default)
        agent.load_state_dict(partial_state_dict, strict=False)
        
        # Check that partial weights were loaded
        assert torch.equal(agent.actor.weight, partial_state_dict["actor.weight"])
        assert torch.equal(agent.actor.bias, partial_state_dict["actor.bias"])
    
    def test_forward_pass_gradient_flow(self, mock_envs, agent_config, sample_observation):
        """Test that gradients flow through the network."""
        agent = ExplicitAgent(mock_envs, agent_config)
        agent.train()
        
        # Forward pass
        action, log_prob, entropy, value = agent.get_action_and_value(sample_observation)
        
        # Compute dummy loss
        loss = -(log_prob.mean() + value.mean() + entropy.mean())
        loss.backward()
        
        # Check that gradients exist for trainable parameters
        trainable_params = [p for p in agent.parameters() if p.requires_grad]
        assert len(trainable_params) > 0
        
        for param in trainable_params:
            assert param.grad is not None, "Gradient not computed for trainable parameter"
    
    def test_debug_mode_visualization(self, mock_envs, agent_config, sample_observation):
        """Test debug mode functionality (without actually creating plots)."""
        config_debug = agent_config.copy()
        config_debug['debug_mode'] = True
        
        agent = ExplicitAgent(mock_envs, config_debug)
        
        # Mock matplotlib and mlflow to avoid actual plotting
        with patch('matplotlib.pyplot.subplots'), \
             patch('matplotlib.pyplot.tight_layout'), \
             patch('matplotlib.pyplot.close'), \
             patch('mlflow.log_figure'):
            
            # Should not raise errors in debug mode
            features = agent.get_clip_surgery_features(sample_observation)
            sam_features = agent.get_sam_features(sample_observation)
            
            assert features is not None
            assert sam_features is not None