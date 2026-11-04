# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import TRANSFORMS
import mmcv
import numpy as np


@TRANSFORMS.register_module()
class LoadPairedImagesFromFile:
    """Load paired RGB and IR images for cross-modal distillation.

    This transform loads an IR image into the 'img' key (for the student) and
    an RGB image into the 'img_rgb' key (for the teacher).

    It expects the input results dict to have:
    - 'img_path': path to the RGB image.
    - 'img2_path': path to the IR image.
    """

    def __init__(self, to_float32=False, color_type='color', backend_args=None):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.backend_args = backend_args

    def __call__(self, results):
        """
        Args:
            results (dict): The results dict from the dataset.
        Returns:
            dict: The results dict updated with loaded images.
        """
        # After KAISTDataset.prepare_data() swaps paths:
        #   img_path  → IR image (for student)
        #   img2_path → RGB image (for teacher)
        ir_path = results['img_path']
        rgb_path = results['img2_path']

        try:
            ir_img = mmcv.imread(
                ir_path, channel_order='bgr', backend_args=self.backend_args)
            rgb_img = mmcv.imread(
                rgb_path, channel_order='bgr', backend_args=self.backend_args)
        except Exception as e:
            raise ValueError(f"Error loading images. IR: {ir_path}, RGB: {rgb_path}. Error: {e}")

        if rgb_img is None:
            raise ValueError(f'Failed to load RGB image: {rgb_path}')
        if ir_img is None:
            raise ValueError(f'Failed to load IR image: {ir_path}')

        if self.to_float32:
            rgb_img = rgb_img.astype(np.float32)
            ir_img = ir_img.astype(np.float32)

        # The IR image is the primary input for the student model.
        # It goes into the standard 'img' key.
        results['img'] = ir_img

        # The RGB image is the auxiliary input for the teacher model.
        # It goes into a new 'img_rgb' key.
        results['img_rgb'] = rgb_img

        # The pipeline uses the shape of the primary 'img' key.
        results['img_shape'] = ir_img.shape[:2]
        results['ori_shape'] = ir_img.shape[:2]

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'backend_args={self.backend_args})')
        return repr_str


from mmdet.datasets.transforms import Resize as MMDetectionResize
from mmdet.datasets.transforms import RandomFlip as MMDetectionRandomFlip

@TRANSFORMS.register_module()
class PairedResize(MMDetectionResize):
    """Resize paired images (img and img_rgb) with the same parameters."""
    
    def transform(self, results):
        # Apply standard resize to 'img'
        results = super().transform(results)
        
        if 'img_rgb' in results:
            # Ensure we use the exact same new shape
            new_h, new_w = results['img_shape'][:2]
            
            img_rgb = results['img_rgb']
            # Use mmcv.imresize to match the primary image size exactly
            # This implicitly respects the keep_ratio/scale logic decided by the parent class
            img_rgb = mmcv.imresize(img_rgb, (new_w, new_h), interpolation=self.interpolation)
            results['img_rgb'] = img_rgb
            
        return results

@TRANSFORMS.register_module()
class PairedRandomFlip(MMDetectionRandomFlip):
    """Randomly flip paired images (img and img_rgb) with the same parameters."""
    
    def transform(self, results):
        # Apply to 'img' (decides flip, direction, updates keys)
        results = super().transform(results)
        
        if 'img_rgb' in results and results.get('flip', False):
            # Apply same flip
            direction = results['flip_direction']
            # mmcv.imflip handles the flipping
            results['img_rgb'] = mmcv.imflip(results['img_rgb'], direction=direction)
            
        return results

@TRANSFORMS.register_module()
class Rename:
    """A simple transform to rename keys in the results dictionary.

    This is used as a workaround for the buggy TransformBroadcaster.

    Args:
        mapping (dict): A dictionary mapping old keys to new keys.
    """

    def __init__(self, mapping):
        self.mapping = mapping

    def transform(self, results):
        """Transform function to rename keys."""
        for old_key, new_key in self.mapping.items():
            if old_key in results:
                results[new_key] = results.pop(old_key)
        return results

    def __call__(self, results):
        """Call function to rename keys in results."""
        return self.transform(results)

    def __repr__(self):
        return f'{self.__class__.__name__}(mapping={self.mapping})'