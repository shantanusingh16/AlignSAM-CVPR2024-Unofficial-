import pytest
import torch
import numpy as np
import gymnasium as gym
from unittest.mock import Mock, MagicMock
import tempfile
import os
import yaml

# Test fixtures for common objects

@pytest.fixture
def device():
    """Return CPU device for testing."""
    return torch.device('cpu')

@pytest.fixture
def mock_envs():
    """Mock Gymnasium environment for agent testing."""
    mock_env = Mock()
    mock_env.single_action_space = Mock()
    mock_env.single_action_space.n = 64  # 32x32 patches * 2 labels
    mock_env.envs = [Mock()]
    mock_env.envs[0].env = Mock()  # Mock nested env structure
    mock_env.envs[0].env.max_steps = 5
    return mock_env

@pytest.fixture
def sample_observation():
    """Sample observation dictionary for testing."""
    return {
        "image": torch.randint(0, 255, (2, 375, 500, 3), dtype=torch.uint8),
        "target_category": ["person", "car"],
        "sam_image_embeddings": torch.randn(2, 256, 64, 64),
        "sam_pred_mask_prob": torch.rand(2, 256, 256),
        "num_steps": torch.tensor([1, 2])
    }

@pytest.fixture
def agent_config():
    """Sample agent configuration."""
    return {
        'type': 'explicit',
        'clip_model_name': 'CS-ViT-B/16',
        'clip_image_size': [224, 224],
        'clip_text_prompt': ['person', 'cat', 'dog', 'car', 'bicycle', 'bus'],
        'similarity_scale_temperature': 0.33,
        'debug_mode': False
    }

@pytest.fixture
def env_config():
    """Sample environment configuration."""
    return {
        'img_shape': [375, 500, 3],
        'embedding_shape': [256, 64, 64],
        'mask_shape': [256, 256],
        'render_frame_shape': [320, 426],
        'max_steps': 5,
        'num_patches': 32,
        'penalize_for_wrong_input': False,
        'use_dice_score': True,
        'render_mode': 'rgb_array',
        'target_categories': ['person', 'cat', 'dog', 'car', 'bicycle', 'bus'],
        'dataset_config': {
            'type': 'coco',
            'data_dir': 'data/coco-dataset',
            'data_type': 'val2017',
            'seed': 42,
            'max_instances': 5
        },
        'sam_ckpt_fp': 'RepViT/sam/weights/repvit_sam.pt'
    }

@pytest.fixture
def mock_coco_dataset():
    """Mock COCO dataset for testing."""
    dataset = Mock()
    dataset.target_categories = ['person', 'car']
    dataset.coco = Mock()
    dataset.coco.getCatIds.return_value = [1, 3]
    dataset.coco.getImgIds.return_value = [100, 200, 300]
    dataset.coco.loadImgs.return_value = [{'file_name': 'test.jpg'}]
    
    # Mock get_sample method
    sample_image = np.random.randint(0, 255, (375, 500, 3), dtype=np.uint8)
    sample_masks = np.random.randint(0, 2, (375, 500), dtype=np.uint8)
    sample_cat_masks = np.random.randint(0, 3, (375, 500), dtype=np.uint8)
    
    dataset.get_sample.return_value = (sample_image, sample_masks, sample_cat_masks, 'person')
    return dataset

@pytest.fixture
def mock_sam_wrapper():
    """Mock SAM wrapper for testing."""
    wrapper = Mock()
    wrapper.set_image = Mock()
    wrapper.predict = Mock()
    wrapper.predict.return_value = (
        torch.rand(256, 256),  # mask_prob
        torch.randn(256, 64, 64)  # image_embeddings
    )
    return wrapper

@pytest.fixture
def temp_config_dir():
    """Temporary directory with config files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create agent config
        agent_config = {
            'type': 'explicit',
            'clip_model_name': 'CS-ViT-B/16',
            'clip_image_size': [224, 224],
            'clip_text_prompt': ['person', 'car'],
            'similarity_scale_temperature': 0.33,
            'debug_mode': False
        }
        
        env_config = {
            'img_shape': [375, 500, 3],
            'embedding_shape': [256, 64, 64],
            'mask_shape': [256, 256],
            'render_frame_shape': [320, 426],
            'max_steps': 5,
            'num_patches': 32,
            'target_categories': ['person', 'car'],
            'dataset_config': {
                'type': 'coco',
                'data_dir': 'data/test',
                'data_type': 'val2017',
                'seed': 42
            },
            'sam_ckpt_fp': 'test_sam.pt'
        }
        
        # Write config files
        os.makedirs(os.path.join(tmpdir, 'agents'))
        os.makedirs(os.path.join(tmpdir, 'envs'))
        
        with open(os.path.join(tmpdir, 'agents', 'test_agent.yaml'), 'w') as f:
            yaml.dump(agent_config, f)
            
        with open(os.path.join(tmpdir, 'envs', 'test_env.yaml'), 'w') as f:
            yaml.dump(env_config, f)
            
        yield tmpdir

@pytest.fixture(autouse=True)
def mock_external_dependencies(monkeypatch):
    """Mock external dependencies that are expensive or unavailable in tests."""
    # Mock CLIP loading
    mock_clip = Mock()
    mock_clip_model = Mock()
    mock_clip_model.eval.return_value = None
    mock_clip_model.parameters.return_value = []
    mock_clip_model.encode_image.return_value = torch.randn(2, 512)
    
    def mock_load(model_name, device):
        return mock_clip_model, None
        
    monkeypatch.setattr("CLIP_Surgery.clip.load", mock_load)
    
    # Mock CLIP Surgery functions
    def mock_encode_text(model, prompts, device):
        return torch.randn(len(prompts), 512)
    
    def mock_clip_feature_surgery(image_features, text_features, redundant_features):
        batch_size = image_features.shape[0]
        num_classes = text_features.shape[0] + 1
        return torch.rand(batch_size, num_classes, 196)  # 14x14 patches
    
    def mock_get_similarity_map(similarity, target_size):
        batch_size, num_classes, num_patches = similarity.shape
        h, w = target_size
        return torch.rand(batch_size, h, w, num_classes)
    
    monkeypatch.setattr("CLIP_Surgery.clip.encode_text_with_prompt_ensemble", mock_encode_text)
    monkeypatch.setattr("CLIP_Surgery.clip.clip_feature_surgery", mock_clip_feature_surgery)
    monkeypatch.setattr("CLIP_Surgery.clip.get_similarity_map", mock_get_similarity_map)
    
    # Mock pycocotools COCO
    mock_coco_class = Mock()
    monkeypatch.setattr("pycocotools.coco.COCO", mock_coco_class)
    
    # Mock cv2 imread to avoid file system dependencies
    def mock_imread(filepath, flag):
        return np.random.randint(0, 255, (375, 500, 3), dtype=np.uint8)
    
    monkeypatch.setattr("cv2.imread", mock_imread)
    
    # Mock os.path.exists for file checks
    monkeypatch.setattr("os.path.exists", lambda x: True)