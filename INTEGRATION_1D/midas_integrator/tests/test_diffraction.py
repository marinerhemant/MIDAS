#!/usr/bin/env python3
"""
Unit tests for the midas_integrator package.

Author: Hemant Sharma
Date: 2025/03/06
"""

import os
import unittest
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt

# Import the module to be tested
from midas_integrator import (
    DiffractionConfig, VoigtFitter, BinaryUtils, 
    ImageUtils, GPUUtils, ImageIntegrator, DiffractionProcessor
)

class TestVoigtFitter(unittest.TestCase):
    """Test cases for the VoigtFitter class."""
    
    def test_func_voigt(self):
        """Test that the Voigt function produces expected values."""
        # Create a simple x array
        x = np.linspace(0, 10, 100)
        
        # Test with known parameters
        amp = 10.0
        bg = 1.0
        mix = 0.5
        cen = 5.0
        width = 1.0
        
        # Calculate the Voigt profile
        y = VoigtFitter.func_voigt(x, amp, bg, mix, cen, width)
        
        # Basic checks
        self.assertEqual(len(y), len(x))
        self.assertGreater(y.max(), bg)  # Max value should be above background
        self.assertAlmostEqual(x[np.argmax(y)], cen, delta=0.1)  # Peak should be near center
    
    def test_fit_single_voigt(self):
        """Test fitting a single Voigt profile to synthetic data."""
        # Create synthetic data with known parameters
        x = np.linspace(0, 10, 100)
        true_params = [10.0, 1.0, 0.5, 5.0, 1.0]
        y_true = VoigtFitter.func_voigt(x, *true_params)
        
        # Add some noise
        np.random.seed(42)
        y_noisy = y_true + np.random.normal(0, 0.1, len(x))
        
        # Fit the data
        fitted_params, _ = VoigtFitter.fit_single_voigt(x, y_noisy)
        
        # Check that fitted parameters are close to true values
        for i, (true, fitted) in enumerate(zip(true_params, fitted_params)):
            self.assertAlmostEqual(true, fitted, delta=true*0.2)  # Within 20%
    
    def test_multi_voigt(self):
        """Test that the multi-Voigt function works correctly."""
        # Create a simple x array
        x = np.linspace(0, 20, 200)
        
        # Parameters for two peaks
        params = [
            10.0, 1.0, 0.5, 5.0, 1.0,  # First peak
            5.0, 1.0, 0.3, 15.0, 2.0   # Second peak
        ]
        
        # Calculate the multi-Voigt profile
        y = VoigtFitter.multi_voigt(x, *params)
        
        # Check that there are two distinct peaks
        peaks = []
        for i in range(1, len(x)-1):
            if y[i] > y[i-1] and y[i] > y[i+1]:
                peaks.append(i)
        
        self.assertEqual(len(peaks), 2)
        
        # Check peak positions are approximately correct
        self.assertAlmostEqual(x[peaks[0]], 5.0, delta=1.0)
        self.assertAlmostEqual(x[peaks[1]], 15.0, delta=1.0)

class TestImageUtils(unittest.TestCase):
    """Test cases for the ImageUtils class."""
    
    def test_load_image_data_missing_file(self):
        """Test that load_image_data raises an error for missing files."""
        # Test with a non-existent file
        with self.assertRaises(FileNotFoundError):
            ImageUtils.load_image_data("nonexistent_file.tif")
    
    @patch('PIL.Image.open')
    def test_load_image_data_with_dark(self, mock_open):
        """Test loading an image with dark correction."""
        # Mock the image data
        mock_img = MagicMock()
        mock_img.__enter__.return_value = mock_img
        mock_img.return_value = mock_img
        
        # Create synthetic image arrays
        image_array = np.ones((10, 10)) * 10
        dark_array = np.ones((10, 10)) * 2
        
        # Set up the mock to return our synthetic arrays
        mock_img.return_value = mock_img
        mock_img.__enter__.return_value = mock_img
        mock_img.array.side_effect = [image_array, dark_array]
        
        # Mock np.array to return our test arrays
        with patch('numpy.array', side_effect=[image_array, dark_array]):
            # Test with mocked files
            result, nr_pixels_y = ImageUtils.load_image_data("test.tif", "dark.tif")
            
            # Check results
            self.assertEqual(result.shape, (10, 10))
            self.assertEqual(nr_pixels_y, 10)
            # The result should be image - dark with clipping
            self.assertTrue(np.all(result == 8))


class TestDiffractionConfig(unittest.TestCase):
    """Test cases for the DiffractionConfig class."""
    
    def test_to_dict_from_dict(self):
        """Test that config can be converted to dict and back."""
        # Create a config
        config = DiffractionConfig(
            image_path="test.tif",
            dark_path="dark.tif",
            r_min=20.0,
            r_max=100.0
        )
        
        # Convert to dict
        config_dict = config.to_dict()
        
        # Check dict contains expected values
        self.assertEqual(config_dict['image_path'], "test.tif")
        self.assertEqual(config_dict['dark_path'], "dark.tif")
        self.assertEqual(config_dict['r_min'], 20.0)
        
        # Convert back to config
        config2 = DiffractionConfig.from_dict(config_dict)
        
        # Check values match
        self.assertEqual(config.image_path, config2.image_path)
        self.assertEqual(config.dark_path, config2.dark_path)
        self.assertEqual(config.r_min, config2.r_min)
    
    def test_save_load(self):
        """Test saving and loading config to/from a file."""
        # Create a config
        config = DiffractionConfig(
            image_path="test.tif",
            dark_path="dark.tif",
            r_min=20.0,
            r_max=100.0
        )
        
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp:
            temp_path = temp.name
        
        try:
            config.save(temp_path)
            
            # Load from the file
            config2 = DiffractionConfig.load(temp_path)
            
            # Check values match
            self.assertEqual(config.image_path, config2.image_path)
            self.assertEqual(config.dark_path, config2.dark_path)
            self.assertEqual(config.r_min, config2.r_min)
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)


@unittest.skipIf(not GPUUtils.cuda_available(), "CUDA not available")
class TestGPUIntegration(unittest.TestCase):
    """
    Test cases for GPU-accelerated integration.
    These tests are skipped if CUDA is not available.
    """
    
    def test_gpu_utils(self):
        """Test GPU utility functions."""
        # Test CUDA availability
        self.assertTrue(GPUUtils.cuda_available())
        
        # Test getting optimal block size
        block_size = GPUUtils.get_optimal_block_size()
        self.assertGreater(block_size, 0)
        self.assertLessEqual(block_size, 1024)  # Max allowed by CUDA
    
    def test_gpu_vs_cpu_integration(self):
        """Test that GPU and CPU integration give similar results."""
        # Create synthetic test data
        np.random.seed(42)
        image = np.random.rand(100, 100).astype(np.float32)
        px_list = np.zeros((1000, 4), dtype=np.int32)
        frac_values = np.ones(1000, dtype=np.float64)
        n_px_list = np.zeros(100, dtype=np.int32)
        
        # Fill with test values
        for i in range(1000):
            px_list[i] = [i % 100, i // 100, 0, 0]
            
        for i in range(50):
            n_px_list[i*2] = 20
            n_px_list[i*2+1] = i * 20
        
        # Run CPU integration
        cpu_result = ImageIntegrator.integrate_image_cpu(
            image, px_list, n_px_list, frac_values, 
            50, 1, 0, 1, -1, -2, 100
        )
        
        # Run GPU integration
        gpu_result = ImageIntegrator.integrate_image_cuda(
            image, px_list, n_px_list, frac_values, 
            50, 1, 0, 1, -1, -2, 100
        )
        
        # Check that results are similar
        # Note: Due to floating point precision differences, 
        # we expect some small differences
        self.assertEqual(cpu_result.shape, gpu_result.shape)
        np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-5, atol=1e-5)


class TestDiffractionProcessor(unittest.TestCase):
    """Test cases for the DiffractionProcessor class."""
    
    @patch('midas_integrator.core.ImageUtils.load_image_data')
    @patch('midas_integrator.core.BinaryUtils.load_pixel_maps')
    @patch('midas_integrator.core.ImageIntegrator.integrate_image')
    @patch('midas_integrator.core.VoigtFitter.fit_single_voigt')
    @patch('midas_integrator.core.PlotUtils.plot_results')
    def test_process_single_peak(self, mock_plot, mock_fit, mock_integrate, mock_load_maps, mock_load_image):
        """Test processing a diffraction image with a single peak."""
        # Create synthetic data
        mock_load_image.return_value = (np.ones((100, 100)), 100)
        mock_load_maps.return_value = (np.zeros((100, 4)), np.ones(100), np.zeros(100))
        
        # Mock the integration result
        mock_result = np.zeros((50, 2))
        mock_result[:, 0] = np.linspace(0, 10, 50)  # radius
        mock_result[:, 1] = np.sin(mock_result[:, 0]) + 1  # intensity with a peak
        mock_integrate.return_value = mock_result
        
        # Mock the fitting result
        mock_params = np.array([10.0, 1.0, 0.5, 5.0, 1.0])
        mock_fit.return_value = (mock_params, None)
        
        # Create a processor with a simple configuration
        config = DiffractionConfig(
            image_path="test.tif",
            output_file="output.png",
            num_peaks=1
        )
        processor = DiffractionProcessor(config)
        
        # Process the image
        result, params = processor.process()
        
        # Check results
        self.assertIs(result, mock_result)
        self.assertIs(params, mock_params)
        
        # Verify mocks were called
        mock_load_image.assert_called_once()
        mock_load_maps.assert_called_once()
        mock_integrate.assert_called_once()
        mock_fit.assert_called_once()
        mock_plot.assert_called_once()


if __name__ == '__main__':
    unittest.main()