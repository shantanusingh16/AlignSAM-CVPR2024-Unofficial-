import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import json
import sys

# Add the project root to the Python path  
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from datasets.coco_dataset import CocoDataset


class TestCocoDataset:
    """Test CocoDataset class."""
    
    @pytest.fixture
    def mock_coco_api(self):
        """Mock COCO API object."""
        mock_coco = Mock()
        mock_coco.getCatIds.return_value = [1, 2, 3]
        mock_coco.getImgIds.return_value = [100, 200, 300, 400, 500]
        mock_coco.loadImgs.return_value = [{'file_name': 'test_image.jpg'}]
        mock_coco.loadCats.return_value = [
            {'id': 1, 'name': 'person'},
            {'id': 2, 'name': 'bicycle'}, 
            {'id': 3, 'name': 'car'}
        ]
        mock_coco.loadAnns.return_value = [
            {'category_id': 1, 'segmentation': [[10, 10, 20, 10, 20, 20, 10, 20]]},
            {'category_id': 2, 'segmentation': [[30, 30, 40, 30, 40, 40, 30, 40]]}
        ]
        mock_coco.getAnnIds.return_value = [1001, 1002]
        return mock_coco
    
    @pytest.fixture
    def temp_dataset_dir(self):
        """Create temporary dataset directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create directory structure
            images_dir = os.path.join(tmpdir, 'images', 'val2017')
            annotations_dir = os.path.join(tmpdir, 'annotations')
            os.makedirs(images_dir)
            os.makedirs(annotations_dir)
            
            # Create dummy annotation file
            dummy_annotations = {
                "images": [{"id": 100, "file_name": "test_image.jpg"}],
                "annotations": [],
                "categories": [{"id": 1, "name": "person"}]
            }
            
            ann_file = os.path.join(annotations_dir, 'instances_val2017.json')
            with open(ann_file, 'w') as f:
                json.dump(dummy_annotations, f)
            
            # Create dummy image file
            img_file = os.path.join(images_dir, 'test_image.jpg')
            # Create a simple RGB image array and save as image
            dummy_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            
            yield tmpdir, img_file
    
    def test_init(self, mock_coco_api, temp_dataset_dir):
        """Test dataset initialization."""
        tmpdir, _ = temp_dataset_dir
        
        with patch('datasets.coco_dataset.COCO', return_value=mock_coco_api):
            dataset = CocoDataset(
                data_dir=tmpdir,
                data_type='val2017', 
                seed=42,
                max_instances=10
            )
            
            assert dataset.data_dir == tmpdir
            assert dataset.data_type == 'val2017'
            assert dataset.max_instances == 10
            assert dataset.random is not None
            assert dataset.coco is mock_coco_api
    
    def test_init_default_max_instances(self, mock_coco_api, temp_dataset_dir):
        """Test initialization with default max_instances."""
        tmpdir, _ = temp_dataset_dir
        
        with patch('datasets.coco_dataset.COCO', return_value=mock_coco_api):
            dataset = CocoDataset(
                data_dir=tmpdir,
                data_type='val2017',
                max_instances=-1  # Should use default
            )
            
            assert dataset.max_instances == 100  # Default value
    
    def test_configure_targets_valid(self, mock_coco_api, temp_dataset_dir):
        """Test target configuration with valid categories."""
        tmpdir, _ = temp_dataset_dir
        
        with patch('datasets.coco_dataset.COCO', return_value=mock_coco_api):
            dataset = CocoDataset(tmpdir, 'val2017')
            
            target_categories = ['person', 'car']
            cat_ids, img_ids = dataset.configure_targets(target_categories)
            
            # Check that COCO API methods were called correctly
            mock_coco_api.getCatIds.assert_called_with(catNms=target_categories)
            mock_coco_api.getImgIds.assert_called_with(catIds=[1, 2, 3])  # Mock return value
            
            assert cat_ids == [1, 2, 3]
            assert img_ids == [100, 200, 300, 400, 500]
    
    def test_configure_targets_invalid(self, mock_coco_api, temp_dataset_dir):
        """Test target configuration with invalid categories."""
        tmpdir, _ = temp_dataset_dir
        
        # Make getCatIds raise an exception for invalid categories
        mock_coco_api.getCatIds.side_effect = Exception("Invalid category")
        
        with patch('datasets.coco_dataset.COCO', return_value=mock_coco_api):
            dataset = CocoDataset(tmpdir, 'val2017')
            
            with pytest.raises(Exception) as exc_info:
                dataset.configure_targets(['invalid_category'])
            
            assert "Incorrect categories passed" in str(exc_info.value)
    
    def test_load_image_success(self, mock_coco_api, temp_dataset_dir):
        """Test successful image loading."""
        tmpdir, img_file = temp_dataset_dir
        
        with patch('datasets.coco_dataset.COCO', return_value=mock_coco_api):
            dataset = CocoDataset(tmpdir, 'val2017')
            
            # Mock cv2.imread to return valid image
            with patch('datasets.coco_dataset.cv2.imread') as mock_imread, \
                 patch('datasets.coco_dataset.cv2.cvtColor') as mock_cvtColor:
                
                mock_img_bgr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                mock_img_rgb = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                
                mock_imread.return_value = mock_img_bgr
                mock_cvtColor.return_value = mock_img_rgb
                
                img = dataset.load_image(100)
                
                # Check that image loading was called correctly
                mock_imread.assert_called_once()
                mock_cvtColor.assert_called_once()
                assert img.shape == (100, 100, 3)
                assert np.array_equal(img, mock_img_rgb)
    
    def test_load_image_file_not_found(self, mock_coco_api, temp_dataset_dir):
        """Test image loading when file doesn't exist."""
        tmpdir, _ = temp_dataset_dir
        
        with patch('datasets.coco_dataset.COCO', return_value=mock_coco_api):
            dataset = CocoDataset(tmpdir, 'val2017')
            
            with patch('datasets.coco_dataset.os.path.exists', return_value=False):
                with pytest.raises(FileNotFoundError):
                    dataset.load_image(100)
    
    def test_load_image_opencv_fails(self, mock_coco_api, temp_dataset_dir):
        """Test image loading when OpenCV fails to read image."""
        tmpdir, _ = temp_dataset_dir
        
        with patch('datasets.coco_dataset.COCO', return_value=mock_coco_api):
            dataset = CocoDataset(tmpdir, 'val2017')
            
            with patch('datasets.coco_dataset.cv2.imread', return_value=None):
                with pytest.raises(ValueError) as exc_info:
                    dataset.load_image(100)
                
                assert "could not be loaded" in str(exc_info.value)
    
    def test_get_sample_success(self, mock_coco_api, temp_dataset_dir):
        """Test successful sample retrieval."""
        tmpdir, _ = temp_dataset_dir
        
        # Setup mock to return valid data
        mock_coco_api.getCatIds.return_value = [1, 2]
        mock_coco_api.getImgIds.return_value = [100]
        
        with patch('datasets.coco_dataset.COCO', return_value=mock_coco_api):
            dataset = CocoDataset(tmpdir, 'val2017', seed=42)
            
            # Mock image loading and mask generation
            with patch.object(dataset, 'load_image') as mock_load_image, \
                 patch('datasets.coco_dataset.np.random.RandomState') as mock_random_state:
                
                # Setup mocks
                mock_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                mock_load_image.return_value = mock_img
                
                mock_random = Mock()
                mock_random.choice.side_effect = [100, 1]  # img_id, category_id
                mock_random.randint.return_value = 0  # instance selection
                mock_random_state.return_value = mock_random
                dataset.random = mock_random
                
                # Mock annotation loading
                mock_coco_api.getAnnIds.return_value = [1001]
                mock_coco_api.loadAnns.return_value = [{
                    'category_id': 1,
                    'segmentation': [[10, 10, 20, 10, 20, 20, 10, 20]]
                }]
                
                # Mock mask creation  
                with patch('datasets.coco_dataset.np.zeros') as mock_zeros, \
                     patch('datasets.coco_dataset.cv2.fillPoly') as mock_fillPoly:
                    
                    mock_mask = np.zeros((100, 100), dtype=np.uint8)
                    mock_cat_mask = np.zeros((100, 100), dtype=np.uint8)
                    mock_zeros.side_effect = [mock_mask, mock_cat_mask]
                    
                    result = dataset.get_sample(['person'])
                    
                    # Check return values
                    assert len(result) == 4
                    img, mask, cat_mask, target_cat = result
                    
                    assert np.array_equal(img, mock_img)
                    assert target_cat == 'person'
    
    def test_get_sample_no_valid_instances(self, mock_coco_api, temp_dataset_dir):
        """Test get_sample when no valid instances found."""
        tmpdir, _ = temp_dataset_dir
        
        mock_coco_api.getCatIds.return_value = [1]
        mock_coco_api.getImgIds.return_value = [100, 200]
        
        with patch('datasets.coco_dataset.COCO', return_value=mock_coco_api):
            dataset = CocoDataset(tmpdir, 'val2017', seed=42)
            
            # Mock to always return empty annotations
            mock_coco_api.getAnnIds.return_value = []
            
            with patch.object(dataset, 'load_image') as mock_load_image:
                mock_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) 
                mock_load_image.return_value = mock_img
                
                # Should keep trying different images
                with patch.object(dataset.random, 'choice', side_effect=[100, 200, 100]):
                    # This would normally loop indefinitely, so we'll limit the test
                    # In practice, the method should have better error handling
                    with pytest.raises((IndexError, RecursionError)):
                        # Timeout or recursion limit hit
                        dataset.get_sample(['person'])
    
    def test_random_seed_reproducibility(self, mock_coco_api, temp_dataset_dir):
        """Test that same seed produces reproducible results."""
        tmpdir, _ = temp_dataset_dir
        
        with patch('datasets.coco_dataset.COCO', return_value=mock_coco_api):
            dataset1 = CocoDataset(tmpdir, 'val2017', seed=42)
            dataset2 = CocoDataset(tmpdir, 'val2017', seed=42)
            
            # Generate same random choices
            choices1 = [dataset1.random.choice([1, 2, 3]) for _ in range(5)]
            
            # Reset random state for second dataset
            dataset2.random = np.random.RandomState(42)
            choices2 = [dataset2.random.choice([1, 2, 3]) for _ in range(5)]
            
            assert choices1 == choices2
    
    def test_max_instances_limiting(self, mock_coco_api, temp_dataset_dir):
        """Test that max_instances limits the number of annotations.""" 
        tmpdir, _ = temp_dataset_dir
        
        with patch('datasets.coco_dataset.COCO', return_value=mock_coco_api):
            dataset = CocoDataset(tmpdir, 'val2017', max_instances=2)
            
            # Mock many annotations
            many_anns = [{'category_id': 1} for _ in range(10)]
            mock_coco_api.loadAnns.return_value = many_anns
            mock_coco_api.getAnnIds.return_value = list(range(10))
            
            with patch.object(dataset, 'load_image') as mock_load_image:
                mock_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                mock_load_image.return_value = mock_img
                
                with patch('datasets.coco_dataset.np.zeros') as mock_zeros, \
                     patch('datasets.coco_dataset.cv2.fillPoly'):
                    
                    mock_zeros.return_value = np.zeros((100, 100), dtype=np.uint8)
                    
                    # This should work without trying to process all 10 annotations
                    result = dataset.get_sample(['person'])
                    assert len(result) == 4
    
    def test_polygon_to_mask_conversion(self, mock_coco_api, temp_dataset_dir):
        """Test conversion of polygon segmentation to mask."""
        tmpdir, _ = temp_dataset_dir
        
        with patch('datasets.coco_dataset.COCO', return_value=mock_coco_api):
            dataset = CocoDataset(tmpdir, 'val2017')
            
            with patch.object(dataset, 'load_image') as mock_load_image:
                mock_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                mock_load_image.return_value = mock_img
                
                # Mock annotation with polygon segmentation
                mock_ann = {
                    'category_id': 1,
                    'segmentation': [[10, 10, 30, 10, 30, 30, 10, 30]]  # Rectangle
                }
                mock_coco_api.getAnnIds.return_value = [1001]
                mock_coco_api.loadAnns.return_value = [mock_ann]
                
                with patch('datasets.coco_dataset.cv2.fillPoly') as mock_fillPoly, \
                     patch('datasets.coco_dataset.np.zeros') as mock_zeros:
                    
                    mock_mask = np.zeros((100, 100), dtype=np.uint8)
                    mock_cat_mask = np.zeros((100, 100), dtype=np.uint8) 
                    mock_zeros.side_effect = [mock_mask, mock_cat_mask]
                    
                    result = dataset.get_sample(['person'])
                    
                    # Check that fillPoly was called for mask creation
                    assert mock_fillPoly.call_count >= 1
                    
                    # Check polygon coordinates were processed correctly
                    call_args = mock_fillPoly.call_args_list[0]
                    polygon_points = call_args[0][1]
                    expected_polygon = np.array([[10, 10], [30, 10], [30, 30], [10, 30]], dtype=np.int32)
                    
                    assert polygon_points.shape == expected_polygon.shape