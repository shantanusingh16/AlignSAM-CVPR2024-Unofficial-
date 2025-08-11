import pytest
import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from custom_gym_implns.envs.sam_seg_env import SamSegEnv


class TestSamSegEnv:
    """Test SamSegEnv Gymnasium environment."""
    
    @pytest.fixture
    def env_kwargs(self):
        """Basic environment configuration."""
        return {
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
            'num_patches': 4,  # 4x4 grid
            'penalize_for_wrong_input': True,
            'use_dice_score': False,
            'render_mode': 'rgb_array'
        }
    
    @pytest.fixture
    def mock_dependencies(self, monkeypatch, mock_coco_dataset, mock_sam_wrapper):
        """Mock external dependencies for environment testing."""
        # Mock dataset getter
        def mock_get_dataset(config):
            return mock_coco_dataset
        
        # Mock SAM wrapper  
        def mock_sam_wrapper_init(sam_ckpt_fp):
            return mock_sam_wrapper
            
        monkeypatch.setattr("datasets.get_dataset", mock_get_dataset)
        monkeypatch.setattr("custom_gym_implns.envs.utils.repvit_sam_wrapper.RepVITSamWrapper", 
                          lambda x: mock_sam_wrapper)
    
    def test_init_valid_config(self, env_kwargs, mock_dependencies):
        """Test environment initialization with valid config."""
        env = SamSegEnv(**env_kwargs)
        
        # Check basic attributes
        assert env.img_shape == env_kwargs['img_shape']
        assert env.embedding_shape == env_kwargs['embedding_shape'] 
        assert env.mask_shape == env_kwargs['mask_shape']
        assert env.max_steps == env_kwargs['max_steps']
        assert env.target_categories == env_kwargs['target_categories']
        
        # Check observation space
        assert isinstance(env.observation_space, spaces.Dict)
        assert 'image' in env.observation_space.spaces
        assert 'target_category' in env.observation_space.spaces
        assert 'sam_image_embeddings' in env.observation_space.spaces
        assert 'sam_pred_mask_prob' in env.observation_space.spaces
        assert 'num_steps' in env.observation_space.spaces
        
        # Check action space computation
        expected_actions = 4 * 4 * 2  # num_patches^2 * 2 labels
        assert env.action_space.n == expected_actions
    
    def test_init_with_patch_size(self, env_kwargs, mock_dependencies):
        """Test initialization with img_patch_size instead of num_patches."""
        env_kwargs_patch = env_kwargs.copy()
        del env_kwargs_patch['num_patches']
        env_kwargs_patch['img_patch_size'] = 25  # 100/25 = 4 patches per dim
        
        env = SamSegEnv(**env_kwargs_patch)
        
        # Should result in same action space
        expected_actions = 4 * 4 * 2
        assert env.action_space.n == expected_actions
    
    def test_init_invalid_configs(self, env_kwargs, mock_dependencies):
        """Test initialization with invalid configurations."""
        # Both patch_size and num_patches
        invalid_kwargs = env_kwargs.copy()
        invalid_kwargs['img_patch_size'] = 25
        
        with pytest.raises(AssertionError):
            SamSegEnv(**invalid_kwargs)
        
        # Neither patch_size nor num_patches
        invalid_kwargs2 = env_kwargs.copy()
        del invalid_kwargs2['num_patches']
        
        with pytest.raises(ValueError):
            SamSegEnv(**invalid_kwargs2)
        
        # Invalid render mode
        invalid_kwargs3 = env_kwargs.copy()
        invalid_kwargs3['render_mode'] = 'invalid_mode'
        
        with pytest.raises(AssertionError):
            SamSegEnv(**invalid_kwargs3)
    
    def test_action_to_input_mapping(self, env_kwargs, mock_dependencies):
        """Test action to input coordinate mapping."""
        env = SamSegEnv(**env_kwargs)
        
        # Check that action mapping is created
        assert hasattr(env, '_action_to_input')
        assert len(env._action_to_input) == env.action_space.n
        
        # Check some mappings
        for action_id, (coords, label) in env._action_to_input.items():
            assert isinstance(coords, tuple)
            assert len(coords) == 2
            assert label in [0, 1]  # negative, positive
            
            # Check coordinates are within image bounds
            x, y = coords
            assert 0 <= x < env_kwargs['img_shape'][1]  # width
            assert 0 <= y < env_kwargs['img_shape'][0]  # height
    
    def test_reset(self, env_kwargs, mock_dependencies):
        """Test environment reset."""
        env = SamSegEnv(**env_kwargs)
        
        obs, info = env.reset()
        
        # Check observation structure
        assert isinstance(obs, dict)
        assert 'image' in obs
        assert 'target_category' in obs 
        assert 'sam_image_embeddings' in obs
        assert 'sam_pred_mask_prob' in obs
        assert 'num_steps' in obs
        
        # Check observation shapes
        assert obs['image'].shape == tuple(env_kwargs['img_shape'])
        assert obs['sam_image_embeddings'].shape == tuple(env_kwargs['embedding_shape'])
        assert obs['sam_pred_mask_prob'].shape == tuple(env_kwargs['mask_shape'])
        
        # Check initial state
        assert obs['num_steps'] == 0
        assert obs['target_category'] in env_kwargs['target_categories']
        
        # Check info
        assert isinstance(info, dict)
    
    def test_step_valid_action(self, env_kwargs, mock_dependencies):
        """Test stepping with valid actions."""
        env = SamSegEnv(**env_kwargs)
        env.reset()
        
        # Take a valid action
        action = 5
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check return types
        assert isinstance(obs, dict)
        assert isinstance(reward, (int, float, np.number))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        
        # Check observation structure unchanged
        assert 'image' in obs
        assert 'target_category' in obs
        assert 'sam_image_embeddings' in obs
        assert 'sam_pred_mask_prob' in obs
        assert 'num_steps' in obs
        
        # Check step counter incremented
        assert obs['num_steps'] == 1
    
    def test_step_max_steps(self, env_kwargs, mock_dependencies):
        """Test truncation at max steps."""
        env = SamSegEnv(**env_kwargs)
        env.reset()
        
        # Take max_steps actions
        for step in range(env_kwargs['max_steps']):
            obs, reward, terminated, truncated, info = env.step(0)
            
            if step == env_kwargs['max_steps'] - 1:
                # Should be truncated on final step
                assert truncated == True
                assert obs['num_steps'] == env_kwargs['max_steps']
            else:
                # Should not be truncated before final step
                assert truncated == False
                assert obs['num_steps'] == step + 1
    
    def test_step_invalid_action(self, env_kwargs, mock_dependencies):
        """Test stepping with invalid actions."""
        env = SamSegEnv(**env_kwargs)
        env.reset()
        
        # Action out of bounds
        invalid_action = env.action_space.n + 10
        
        with pytest.raises(KeyError):
            env.step(invalid_action)
    
    def test_reward_computation_iou(self, env_kwargs, mock_dependencies):
        """Test reward computation using IoU."""
        env_kwargs_iou = env_kwargs.copy()
        env_kwargs_iou['use_dice_score'] = False
        
        env = SamSegEnv(**env_kwargs_iou)
        env.reset()
        
        # Mock SAM predictor to return specific mask
        with patch.object(env.sam_predictor, 'predict') as mock_predict:
            # Return mask with some overlap with ground truth
            mock_mask = np.random.rand(100, 100) > 0.5
            mock_embeddings = np.random.randn(256, 16, 16)
            mock_predict.return_value = (mock_mask.astype(np.float32), mock_embeddings)
            
            obs, reward, _, _, _ = env.step(0)
            
            # Should return a valid reward
            assert isinstance(reward, (int, float, np.number))
            assert np.isfinite(reward)
    
    def test_reward_computation_dice(self, env_kwargs, mock_dependencies):
        """Test reward computation using Dice score.""" 
        env_kwargs_dice = env_kwargs.copy()
        env_kwargs_dice['use_dice_score'] = True
        
        env = SamSegEnv(**env_kwargs_dice)
        env.reset()
        
        with patch.object(env.sam_predictor, 'predict') as mock_predict:
            mock_mask = np.random.rand(100, 100) > 0.5
            mock_embeddings = np.random.randn(256, 16, 16)
            mock_predict.return_value = (mock_mask.astype(np.float32), mock_embeddings)
            
            obs, reward, _, _, _ = env.step(0)
            
            # Should return a valid reward
            assert isinstance(reward, (int, float, np.number))
            assert np.isfinite(reward)
    
    def test_penalty_for_wrong_input(self, env_kwargs, mock_dependencies):
        """Test penalty system for wrong input."""
        env_kwargs_penalty = env_kwargs.copy()
        env_kwargs_penalty['penalize_for_wrong_input'] = True
        
        env = SamSegEnv(**env_kwargs_penalty)
        env.reset()
        
        # This would require more complex mocking to test penalty logic
        # For now, just ensure it doesn't crash
        obs, reward, _, _, _ = env.step(0)
        assert np.isfinite(reward)
    
    def test_render_rgb_array(self, env_kwargs, mock_dependencies):
        """Test rendering in rgb_array mode."""
        env = SamSegEnv(**env_kwargs)
        env.reset()
        
        frame = env.render()
        
        # Check frame properties
        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.uint8
        assert len(frame.shape) == 3
        assert frame.shape[2] == 3  # RGB channels
        
        # Check frame size matches render_frame_shape
        expected_h, expected_w = env_kwargs['render_frame_shape']
        assert frame.shape[0] == expected_h
        assert frame.shape[1] == expected_w
    
    def test_multiple_resets(self, env_kwargs, mock_dependencies):
        """Test multiple resets work correctly."""
        env = SamSegEnv(**env_kwargs)
        
        # Reset multiple times
        for _ in range(3):
            obs, info = env.reset()
            
            # Check reset state
            assert obs['num_steps'] == 0
            assert obs['target_category'] in env_kwargs['target_categories']
            
            # Take one step
            obs, _, _, _, _ = env.step(0)
            assert obs['num_steps'] == 1
    
    def test_observation_space_compliance(self, env_kwargs, mock_dependencies):
        """Test that observations comply with declared space."""
        env = SamSegEnv(**env_kwargs)
        obs, _ = env.reset()
        
        # Check each observation component
        for key, space in env.observation_space.spaces.items():
            assert key in obs, f"Missing observation key: {key}"
            
            if isinstance(space, spaces.Box):
                obs_array = obs[key] if isinstance(obs[key], np.ndarray) else np.array(obs[key])
                assert space.contains(obs_array), f"Observation {key} out of bounds"
            elif isinstance(space, spaces.Discrete):
                assert space.contains(obs[key]), f"Discrete observation {key} out of bounds"
            elif isinstance(space, spaces.Text):
                assert isinstance(obs[key], str), f"Text observation {key} not string"
    
    def test_action_space_compliance(self, env_kwargs, mock_dependencies):
        """Test that all actions in action space are valid."""
        env = SamSegEnv(**env_kwargs)
        env.reset()
        
        # Test a sample of actions
        num_test_actions = min(10, env.action_space.n)
        test_actions = np.linspace(0, env.action_space.n - 1, num_test_actions, dtype=int)
        
        for action in test_actions:
            # Should not raise errors
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Reset if truncated/terminated
            if terminated or truncated:
                env.reset()
    
    def test_deterministic_reset_with_seed(self, env_kwargs, mock_dependencies):
        """Test that reset with same seed gives consistent results."""
        env1 = SamSegEnv(**env_kwargs)
        env2 = SamSegEnv(**env_kwargs)
        
        # Reset with same seed
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        
        # Images might be different due to dataset sampling randomness
        # But target categories and other deterministic elements should match
        assert obs1['target_category'] == obs2['target_category']
        assert obs1['num_steps'] == obs2['num_steps']