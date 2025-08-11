import pytest
import yaml
import tempfile
import os
from unittest.mock import Mock, patch
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from models import make_agent


class TestConfigLoading:
    """Test configuration file loading and validation."""
    
    def test_load_agent_config_explicit(self, temp_config_dir):
        """Test loading explicit agent configuration."""
        config_path = os.path.join(temp_config_dir, 'agents', 'test_agent.yaml')
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required fields are present
        assert 'type' in config
        assert config['type'] == 'explicit'
        assert 'clip_model_name' in config
        assert 'clip_image_size' in config
        assert 'clip_text_prompt' in config
        assert 'similarity_scale_temperature' in config
        assert 'debug_mode' in config
        
        # Check data types
        assert isinstance(config['clip_image_size'], list)
        assert len(config['clip_image_size']) == 2
        assert isinstance(config['clip_text_prompt'], list)
        assert isinstance(config['similarity_scale_temperature'], (int, float))
        assert isinstance(config['debug_mode'], bool)
    
    def test_load_env_config(self, temp_config_dir):
        """Test loading environment configuration."""
        config_path = os.path.join(temp_config_dir, 'envs', 'test_env.yaml')
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required fields
        required_fields = [
            'img_shape', 'embedding_shape', 'mask_shape', 'render_frame_shape',
            'max_steps', 'num_patches', 'target_categories', 'dataset_config', 'sam_ckpt_fp'
        ]
        
        for field in required_fields:
            assert field in config, f"Missing required field: {field}"
        
        # Check data types and shapes
        assert isinstance(config['img_shape'], list) and len(config['img_shape']) == 3
        assert isinstance(config['embedding_shape'], list) and len(config['embedding_shape']) == 3
        assert isinstance(config['mask_shape'], list) and len(config['mask_shape']) == 2
        assert isinstance(config['render_frame_shape'], list) and len(config['render_frame_shape']) == 2
        assert isinstance(config['max_steps'], int) and config['max_steps'] > 0
        assert isinstance(config['num_patches'], int) and config['num_patches'] > 0
        assert isinstance(config['target_categories'], list)
        assert isinstance(config['dataset_config'], dict)
        assert isinstance(config['sam_ckpt_fp'], str)
    
    def test_dataset_config_structure(self, temp_config_dir):
        """Test dataset configuration structure."""
        config_path = os.path.join(temp_config_dir, 'envs', 'test_env.yaml')
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        dataset_config = config['dataset_config']
        
        # Check required dataset fields
        assert 'type' in dataset_config
        assert 'data_dir' in dataset_config
        assert 'data_type' in dataset_config
        assert 'seed' in dataset_config
        
        # Check data types
        assert isinstance(dataset_config['type'], str)
        assert isinstance(dataset_config['data_dir'], str)
        assert isinstance(dataset_config['data_type'], str)
        assert isinstance(dataset_config['seed'], int)
    
    def test_make_agent_explicit(self, mock_envs):
        """Test agent creation with explicit configuration."""
        config = {
            'type': 'explicit',
            'clip_model_name': 'CS-ViT-B/16',
            'clip_image_size': [224, 224],
            'clip_text_prompt': ['person', 'car'],
            'similarity_scale_temperature': 0.33,
            'debug_mode': False
        }
        
        agent = make_agent(config, mock_envs)
        
        # Check that correct agent type was created
        from models.explicit_agent import ExplicitAgent
        assert isinstance(agent, ExplicitAgent)
    
    def test_make_agent_implicit(self, mock_envs):
        """Test agent creation with implicit configuration."""
        config = {
            'type': 'implicit',
            'debug_mode': False
        }
        
        agent = make_agent(config, mock_envs)
        
        # Check that correct agent type was created
        from models.implicit_agent import ImplicitAgent
        assert isinstance(agent, ImplicitAgent)
    
    def test_make_agent_invalid_type(self, mock_envs):
        """Test agent creation with invalid type."""
        config = {
            'type': 'invalid_type',
            'debug_mode': False
        }
        
        with pytest.raises(ValueError) as exc_info:
            make_agent(config, mock_envs)
        
        assert "Unknown agent type" in str(exc_info.value)
    
    def test_yaml_syntax_validation(self, temp_config_dir):
        """Test YAML file syntax validation."""
        # Test valid YAML files load without errors
        agent_config_path = os.path.join(temp_config_dir, 'agents', 'test_agent.yaml')
        env_config_path = os.path.join(temp_config_dir, 'envs', 'test_env.yaml')
        
        # Should not raise exceptions
        with open(agent_config_path, 'r') as f:
            agent_config = yaml.safe_load(f)
        
        with open(env_config_path, 'r') as f:
            env_config = yaml.safe_load(f)
        
        assert agent_config is not None
        assert env_config is not None
    
    def test_invalid_yaml_handling(self):
        """Test handling of invalid YAML syntax."""
        invalid_yaml = """
        type: explicit
        clip_model_name: CS-ViT-B/16
        clip_image_size: [224, 224
        # Missing closing bracket
        """
        
        with pytest.raises(yaml.YAMLError):
            yaml.safe_load(invalid_yaml)
    
    def test_config_path_validation(self):
        """Test configuration file path validation."""
        nonexistent_path = "/path/that/does/not/exist.yaml"
        
        with pytest.raises(FileNotFoundError):
            with open(nonexistent_path, 'r') as f:
                yaml.safe_load(f)
    
    def test_config_type_validation(self):
        """Test configuration type validation."""
        # Test explicit agent config validation
        explicit_configs = [
            # Missing clip_model_name
            {
                'type': 'explicit',
                'clip_image_size': [224, 224],
                'clip_text_prompt': ['person'],
                'similarity_scale_temperature': 0.33,
                'debug_mode': False
            },
            # Invalid clip_image_size type
            {
                'type': 'explicit',
                'clip_model_name': 'CS-ViT-B/16',
                'clip_image_size': 224,  # Should be list
                'clip_text_prompt': ['person'],
                'similarity_scale_temperature': 0.33,
                'debug_mode': False
            },
            # Invalid temperature type
            {
                'type': 'explicit',
                'clip_model_name': 'CS-ViT-B/16',
                'clip_image_size': [224, 224],
                'clip_text_prompt': ['person'],
                'similarity_scale_temperature': "0.33",  # Should be float
                'debug_mode': False
            }
        ]
        
        # These should cause errors when creating agents
        for config in explicit_configs:
            with pytest.raises((KeyError, TypeError, ValueError)):
                # Mock environment
                mock_envs = Mock()
                mock_envs.single_action_space = Mock()
                mock_envs.single_action_space.n = 64
                mock_envs.envs = [Mock()]
                mock_envs.envs[0].env = Mock()
                mock_envs.envs[0].env.max_steps = 5
                
                make_agent(config, mock_envs)
    
    def test_environment_config_validation(self, temp_config_dir):
        """Test environment configuration validation."""
        config_path = os.path.join(temp_config_dir, 'envs', 'test_env.yaml')
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Test shape validations
        assert len(config['img_shape']) == 3, "img_shape should have 3 dimensions"
        assert len(config['embedding_shape']) == 3, "embedding_shape should have 3 dimensions"
        assert len(config['mask_shape']) == 2, "mask_shape should have 2 dimensions"
        assert len(config['render_frame_shape']) == 2, "render_frame_shape should have 2 dimensions"
        
        # Test positive integer constraints
        assert config['max_steps'] > 0, "max_steps should be positive"
        assert config['num_patches'] > 0, "num_patches should be positive"
        
        # Test list constraints
        assert len(config['target_categories']) > 0, "target_categories should not be empty"
        
        # Test nested dataset config
        dataset_config = config['dataset_config']
        assert dataset_config['type'] in ['coco'], "dataset type should be valid"
        assert dataset_config['seed'] >= 0, "seed should be non-negative"
    
    def test_config_defaults_handling(self):
        """Test handling of default values in configurations."""
        # Test agent config with minimal required fields
        minimal_explicit_config = {
            'type': 'explicit',
            'clip_model_name': 'CS-ViT-B/16',
            'clip_image_size': [224, 224],
            'clip_text_prompt': ['person'],
            'similarity_scale_temperature': 0.33
            # debug_mode should default to False
        }
        
        # Should work with defaults
        mock_envs = Mock()
        mock_envs.single_action_space = Mock()
        mock_envs.single_action_space.n = 64
        mock_envs.envs = [Mock()]
        mock_envs.envs[0].env = Mock()
        mock_envs.envs[0].env.max_steps = 5
        
        agent = make_agent(minimal_explicit_config, mock_envs)
        assert agent.debug_mode == False  # Should use default
    
    def test_config_file_permissions(self, temp_config_dir):
        """Test configuration file access permissions.""" 
        config_path = os.path.join(temp_config_dir, 'agents', 'test_agent.yaml')
        
        # Should be readable
        assert os.access(config_path, os.R_OK), "Config file should be readable"
        
        # Test loading with different file modes
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            assert config is not None
    
    def test_config_cross_validation(self, temp_config_dir):
        """Test cross-validation between agent and environment configs."""
        agent_config_path = os.path.join(temp_config_dir, 'agents', 'test_agent.yaml')
        env_config_path = os.path.join(temp_config_dir, 'envs', 'test_env.yaml')
        
        with open(agent_config_path, 'r') as f:
            agent_config = yaml.safe_load(f)
        
        with open(env_config_path, 'r') as f:
            env_config = yaml.safe_load(f)
        
        # Check that agent clip_text_prompt matches env target_categories
        agent_prompts = set(agent_config['clip_text_prompt'])
        env_categories = set(env_config['target_categories'])
        
        # Agent prompts should be subset or equal to environment categories
        assert agent_prompts.issubset(env_categories), \
            "Agent prompts should match environment categories"