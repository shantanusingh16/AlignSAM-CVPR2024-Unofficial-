# Project Title: AlignSAM Reinforcement Learning Framework

## Authors
- [Shantanu Singh](mailto:shantanusingh10@gmail.com)

## Overview
This project implements a reinforcement learning framework for interactive segmentation using the Segment Anything Model (SAM) and CLIP Surgery, inspired by the AlignSAM paper (CVPR 2024). The system trains RL agents using Proximal Policy Optimization (PPO) to learn optimal keypoint prompt sequences for object segmentation in images from the COCO dataset.

## Background and Motivation
Interactive segmentation requires intelligent placement of positive and negative clicks to guide segmentation models toward accurate object boundaries. Traditional approaches rely on heuristic strategies using detection or segmentation models that cannot learn, but this project explores using reinforcement learning to learn optimal click placement policies.

The AlignSAM approach demonstrates that RL agents can learn to interact with foundational models more effectively than traditional methods. This unofficial implementation provides an open-source framework for experimenting with RL-based interactive segmentation strategies.

## Goals 
- Implement a complete RL training pipeline for interactive segmentation using SAM
- Support both explicit (CLIP-guided) and implicit agent architectures
- Provide flexible configuration system for experiments and hyperparameter tuning
- Enable comprehensive logging and monitoring for training loops
- Support checkpointing and resumable training for long experiments
- Generate video visualizations of agent learning progress
- Real-time inference optimization or deployment-ready models


## Detailed Design

### System Architecture
The system follows a modular architecture with clear separation between RL training, environment simulation, agent policies, and dataset management.

```
                        ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
                        │   PPO Training  │────│  Gymnasium Env   │────│   SAM Wrapper   │
                        │     Loop        │    │  (SamSegEnv)     │    │  (RepViT-SAM)   │
                        └─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                       │              
                                │                       │                 
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  CLIP features  │    │  Agent Models    │    │  COCO Dataset    │
│                 │    │    (Implict)     │    │    Wrapper       │
│                 │────│    (Explicit)    │    │                  │
└─────────────────┘    └──────────────────┘    └──────────────────┘
```

### Components

#### 1. Training Loop (`train_sam_align_ppo.py`)
**Responsibilities:**
- Orchestrates PPO training with configurable hyperparameters
- Manages experiment logging via MLflow and TensorBoard
- Handles checkpointing and training resumption
- Captures video episodes for visualization

**Key Features:**
- Dataclass-based configuration using `tyro`
- Support for GPU/MPS acceleration
- Configurable video capture frequency
- Automatic experiment tracking and artifact logging

#### 2. Agent Architectures (`models/`)

##### ExplicitAgent (`models/explicit_agent.py`)
**Responsibilities:**
- Combines SAM image embeddings with CLIP Surgery semantic features
- Uses attention mechanisms to fuse multi-modal information
- Provides semantic guidance through text prompts

**Architecture:**
- SAM feature processing network (conv layers + pooling)
- CLIP feature processing network (conv layers + pooling) 
- Cross-attention fusion using ResidualAttentionBlock
- Shared policy/value head for action/value prediction

**Key Features:**
- CLIP Surgery integration with prompt ensembles
- Temperature-scaled similarity mapping
- Debug mode with visualization logging
- Efficient state dict handling (excludes frozen CLIP weights)

##### ImplicitAgent (`models/implicit_agent.py`)  
**Responsibilities:**
- Processes SAM embeddings without external semantic guidance
- Simpler architecture for baseline comparisons
- Direct mapping from SAM features to actions

**Architecture:**
- Conv2D feature extraction from SAM embeddings
- Mask probability modulation of embeddings
- Linear layers for policy/value prediction

#### 3. Environment (`custom_gym_implns/envs/sam_seg_env.py`)
**Responsibilities:**
- Implements Gymnasium interface for RL training
- Manages COCO dataset sampling and target category selection
- Coordinates SAM predictions based on agent actions
- Computes rewards using IoU or Dice scores

**Key Features:**
- Configurable action spaces (patch-based click placement)
- Dynamic reward computation with penalty options
- Support for multiple target categories
- Rendering capabilities for visualization

**Action Space:**
- Discrete actions mapping to (x,y) coordinates and positive/negative labels
- Patch-based discretization for manageable action space size
- Configurable patch sizes or patch counts

**Observation Space:**
- RGB images with target category labels
- SAM image embeddings (256-channel feature maps)
- Current mask predictions from SAM
- Step counters for temporal information

#### 4. SAM Integration (`custom_gym_implns/envs/utils/repvit_sam_wrapper.py`)
**Responsibilities:**
- Wraps RepViT-SAM model for efficient inference
- Manages SAM image encoding and mask prediction
- Handles coordinate transformations between action space and SAM input

#### 5. Dataset Management (`datasets/coco_dataset.py`)
**Responsibilities:**
- COCO dataset loading and preprocessing
- Category-based image filtering
- Random sampling with configurable seeds
- Instance mask extraction and processing

### Data Models

#### Observation Dictionary
```python
{
    "image": np.ndarray,              # RGB image (H, W, 3)
    "target_category": str,           # Target object category name
    "sam_image_embeddings": Tensor,   # SAM encoder output (256, H/4, W/4)
    "sam_pred_mask_prob": Tensor,     # Current mask predictions (H, W)
    "num_steps": int                  # Current step count in episode
}
```

#### Configuration Schema
- **Agent Config**: Model type, CLIP settings, debug flags
- **Environment Config**: Image shapes, dataset paths, reward parameters
- **Training Config**: PPO hyperparameters, logging settings, hardware preferences

### APIs

#### Agent Interface
```python
class Agent(nn.Module):
    def get_action_and_value(obs, action=None) -> (action, log_prob, entropy, value)
    def get_value(obs) -> value
```

#### Environment Interface  
```python
class SamSegEnv(gym.Env):
    def reset() -> (observation, info)
    def step(action) -> (observation, reward, terminated, truncated, info)
    def render() -> np.ndarray
```

#### Dataset Interface
```python
class CocoDataset:
    def get_sample(target_categories) -> (image, masks, metadata)
    def configure_targets(categories) -> (cat_ids, img_ids)
```

### User Interface
The system primarily operates through command-line interfaces with YAML configuration files. Web-based monitoring is available through:
- MLflow UI for experiment tracking and comparison
- TensorBoard for detailed training metrics
- Generated video files for visual progress assessment

## Implementation Strategy

### Phase 1: Core Infrastructure ✓
- ✅ Implement PPO training loop with logging
- ✅ Create Gymnasium environment wrapper
- ✅ Integrate RepViT-SAM model
- ✅ Set up COCO dataset pipeline

### Phase 2: Agent Architectures ✓  
- ✅ Implement ImplicitAgent baseline
- ✅ Develop ExplicitAgent with CLIP Surgery
- ✅ Add attention-based feature fusion
- ✅ Integrate debugging and visualization tools

### Phase 3: Training Pipeline ✓
- ✅ Configure experiment tracking and logging
- ✅ Implement checkpointing and resumption
- ✅ Add video capture capabilities
- ✅ Validate training stability and convergence

### Phase 4: Future Enhancements
- [ ] Add support for additional datasets (ADE20K, CityScapes)
- [ ] Implement distributed training capabilities  
- [ ] Optimize inference performance for deployment
- [ ] Add mixed-precision training support

## Risks and Mitigations

### 1. Training Instability
**Risk**: PPO training may suffer from instability due to sparse rewards and high-dimensional action spaces.

**Mitigation**: 
- Careful hyperparameter tuning with clipping and learning rate scheduling
- Reward shaping with intermediate feedback signals
- Comprehensive logging to detect and diagnose instability early

### 2. Memory Constraints
**Risk**: SAM embeddings and CLIP features consume significant GPU memory, limiting batch sizes.

**Mitigation**:
- Gradient checkpointing for memory-intensive operations
- Configurable batch sizes and environment counts
- Efficient state dict handling to skip frozen CLIP params and reduce checkpoint sizes

### 3. Dataset Bias
**Risk**: COCO dataset limitations may not generalize to other domains or object types.

**Mitigation**:
- Extensible dataset interface for easy integration of new datasets
- Category-balanced sampling strategies
- Comprehensive evaluation across diverse object categories

### 4. Reproducibility
**Risk**: RL training inherently involves stochasticity that can affect result reproducibility.

**Mitigation**:
- Comprehensive seed management across all random components
- Detailed logging of hyperparameters and environment settings
- Multiple training runs with statistical analysis

## Testing Strategy

### Unit Testing
- Agent forward pass consistency and gradient flow
- Environment state transitions and reward computation  
- Dataset sampling and category filtering correctness
- Configuration loading and validation

### Integration Testing
- End-to-end training loop execution
- Checkpoint saving and loading functionality
- Multi-environment parallel execution
- Cross-platform compatibility (CUDA/MPS/CPU)

### Performance Testing (PENDING)
- Memory usage profiling during training
- Training throughput benchmarking
- Inference latency measurement
- Scalability testing with different batch sizes

### Validation Testing (PENDING)
- Agent learning curve analysis and convergence validation
- Reward signal correctness across different scenarios  
- Robustness testing for multi-node and multi-gpu configurations

## Dependencies

### Submodules
- **RepViT**: Efficient vision transformer backbone for SAM
- **CLIP Surgery**: Modified CLIP model for semantic feature extraction

### Core Framework
- **PyTorch**: Deep learning framework with CUDA support
- **Gymnasium**: RL environment interface standard

### Computer Vision
- **OpenCV**: Image processing and manipulation
- **PyCocoTools**: COCO dataset interface and evaluation metrics

### Machine Learning
- **TorchVision**: Computer vision utilities and transforms
- **Timm**: Pre-trained pytorch image models and utilities

### Experiment Management  
- **MLflow**: Experiment tracking and model registry
- **TensorBoard**: Dashboard for training visualization and monitoring
- **Tyro**: Command-line interface generation


## Timeline

### Completed (Baseline Implementation)
- ✅ Core infrastructure and PPO training loop
- ✅ Environment implementation and SAM integration
- ✅ Agent architectures and CLIP Surgery integration
- ✅ Training pipeline validation and debugging tools

### Future Development (Enhancement Phase)
- **1-2 Weeks**: Performance optimization and memory efficiency improvements
- **1 Week**: Additional dataset integration and evaluation frameworks
- **1-2 Weeks**: Distributed training implementation and scalability testing
- **1 Week**: Model export capabilities and inference optimization
- **2-3 Weeks**: Adding tests for performance and validation testing. 

## Conclusion

This AlignSAM implementation provides a comprehensive framework for researching RL-based interactive segmentation. The modular architecture supports experimentation with different agent designs, datasets, and training strategies. The integration of modern ML tools (MLflow, TensorBoard) ensures reproducible research and effective experiment management.

The system successfully demonstrates that RL agents can learn effective click placement strategies for interactive segmentation, opening avenues for further research in human-AI collaborative image editing and automated annotation tools. The extensible design facilitates future enhancements and integration with emerging segmentation models and datasets.