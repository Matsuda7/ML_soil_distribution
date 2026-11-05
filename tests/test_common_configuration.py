"""Shared configuration tests, written but not executed for source-only work."""
import unittest
from unittest.mock import patch
from module.configuration import load_prediction_settings


class CommonConfigurationTests(unittest.TestCase):
    def settings(self):
        return {
            'target.json': {'input_chunk_size': 1000},
            'config/spatial.json': {'grid_spacing_m': 10},
            'config/validation.json': {'mode':'borehole','test_size':.2,'validation_size':.2,'random_state':42,'n_splits':5},
            'config/model.json': {'prediction_depth_m':5,'observed_depth_tolerance_m':1},
        }

    def test_reads_spatial_and_validation_directly(self):
        settings = self.settings()
        with patch('module.configuration._read_json', side_effect=lambda p: settings[p]) as read:
            config = load_prediction_settings('target.json')
        self.assertEqual(config['grid_spacing_m'], 10)
        self.assertEqual(config['validation'], settings['config/validation.json'])
        self.assertEqual(config['validation_config'], 'config/validation.json')
        self.assertEqual({call.args[0] for call in read.call_args_list}, set(settings))

    def test_legacy_target_validation_override_is_preserved(self):
        settings = self.settings()
        settings['target.json']['validation'] = {'random_state':7}
        with patch('module.configuration._read_json', side_effect=lambda p: settings[p]):
            config = load_prediction_settings('target.json')
        self.assertEqual(config['validation']['random_state'], 7)
        self.assertEqual(config['validation']['test_size'], .2)

    def test_legacy_custom_validation_file_is_preserved(self):
        settings = self.settings()
        settings['target.json']['validation_config'] = 'custom.json'
        settings['custom.json'] = {'mode':'row','test_size':.1,'validation_size':.3,'random_state':1}
        with patch('module.configuration._read_json', side_effect=lambda p: settings[p]):
            config = load_prediction_settings('target.json')
        self.assertEqual(config['validation'], settings['custom.json'])
