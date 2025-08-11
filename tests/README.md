# AlignSAM Test Suite

This directory contains comprehensive pytest-based unit and integration tests for the AlignSAM reinforcement learning framework.

## Test Structure

```
tests/
├── conftest.py                    # Test fixtures and configuration
├── test_integration.py            # Integration tests for full pipeline
├── unit/                          # Unit tests
│   ├── models/                    # Agent model tests
│   │   ├── test_explicit_agent.py # ExplicitAgent tests
│   │   └── test_implicit_agent.py # ImplicitAgent tests
│   ├── envs/                      # Environment tests
│   │   └── test_sam_seg_env.py    # SamSegEnv tests
│   ├── datasets/                  # Dataset tests
│   │   └── test_coco_dataset.py   # CocoDataset tests
│   └── test_config_loading.py     # Configuration loading tests
└── README.md                      # This file
```

## Running Tests

### Prerequisites

Install test dependencies (if not already installed):
```bash
pip install pytest pytest-mock
```

### Run All Tests

```bash
# From project root directory
pytest
```

### Run Specific Test Categories

```bash
# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/test_integration.py

# Run tests for specific component
pytest tests/unit/models/
pytest tests/unit/envs/
pytest tests/unit/datasets/
```

### Run with Specific Markers

```bash
# Run only fast tests (exclude slow ones)
pytest -m "not slow"

# Run only integration tests
pytest -m integration

# Run only GPU tests (requires CUDA)
pytest -m gpu
```

### Run with Coverage

```bash
# Install coverage plugin
pip install pytest-cov

# Run with coverage report
pytest --cov=. --cov-report=html --cov-report=term-missing
```

## Test Configuration

Tests are configured via `pytest.ini` in the project root with the following settings:

- **Test Discovery**: Automatically finds `test_*.py` and `*_test.py` files
- **Output**: Verbose output with short tracebacks and colored output
- **Markers**: Custom markers for categorizing tests (slow, integration, unit, gpu, dataset)
- **Warnings**: Filters out common warnings from dependencies
- **Timeouts**: 5-minute timeout for individual tests
- **Logging**: Enabled CLI logging at INFO level

## Test Fixtures

Common test fixtures are defined in `conftest.py`:

- `device`: CPU device for testing
- `mock_envs`: Mock Gymnasium environment wrapper
- `sample_observation`: Sample observation dictionary
- `agent_config`: Sample agent configuration
- `env_config`: Sample environment configuration
- `mock_coco_dataset`: Mock COCO dataset
- `mock_sam_wrapper`: Mock SAM wrapper
- `temp_config_dir`: Temporary directory with config files
- `mock_external_dependencies`: Mocks for CLIP, COCO API, etc.

## Test Coverage

The test suite covers:

### Unit Tests
- **Agent Models**: Forward pass, gradient flow, state dict handling
- **Environment**: Reset/step mechanics, observation spaces, reward computation
- **Dataset**: COCO data loading, category filtering, sample generation
- **Configuration**: YAML loading, validation, agent creation

### Integration Tests
- **Agent-Environment Interaction**: Full interaction loops
- **Batch Processing**: Multi-sample processing
- **Training Simulation**: Gradient computation and loss calculation
- **GPU Compatibility**: CUDA device testing (if available)
- **State Management**: Save/load consistency

## Mocking Strategy

Tests extensively mock external dependencies to:
- **Avoid Heavy Dependencies**: CLIP models, SAM weights, COCO dataset files
- **Ensure Reproducibility**: Consistent test behavior across environments
- **Speed Up Execution**: Fast test runs without model loading
- **Enable Testing**: Test without requiring large model files or datasets

Key mocked components:
- CLIP Surgery model loading and inference
- RepViT-SAM model wrapper
- COCO API and dataset files
- File system operations (cv2.imread, os.path.exists)

## Running Without Dependencies

Tests can run without downloading model weights or datasets thanks to comprehensive mocking:

```bash
# Run tests without any external files
pytest tests/unit/

# Run integration tests with mocked dependencies
pytest tests/test_integration.py
```

## Custom Markers

Use pytest markers to categorize and filter tests:

- `@pytest.mark.slow`: Long-running tests (>30s)
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.gpu`: Requires CUDA GPU
- `@pytest.mark.dataset`: Requires dataset files

## Debugging Tests

### Run Single Test
```bash
pytest tests/unit/models/test_explicit_agent.py::TestExplicitAgent::test_init -v
```

### Drop into Debugger on Failure
```bash
pytest --pdb
```

### Show Local Variables on Failure
```bash
pytest -l
```

### Show Print Statements
```bash
pytest -s
```

## Contributing Tests

When adding new functionality:

1. **Add Unit Tests**: Test individual components in isolation
2. **Mock Dependencies**: Use fixtures to mock external dependencies
3. **Test Edge Cases**: Include error conditions and boundary cases
4. **Add Integration Tests**: Test component interactions for complex features
5. **Use Appropriate Markers**: Mark slow or GPU-dependent tests
6. **Update Fixtures**: Add new fixtures to `conftest.py` for reuse

## Common Issues

### Import Errors
Ensure the project root is in Python path:
```python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
```

### CUDA Errors  
GPU tests are skipped automatically if CUDA is not available:
```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
```

### Mock Issues
Use `patch` decorators or context managers for proper mocking:
```python
with patch('module.function') as mock_func:
    # test code
```

### Fixture Scope
Use appropriate fixture scopes (`function`, `class`, `module`, `session`) based on resource requirements.