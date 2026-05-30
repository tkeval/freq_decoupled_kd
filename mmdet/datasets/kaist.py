from mmdet.registry import DATASETS
from .coco import CocoDataset
import os.path as osp
import json
import copy


@DATASETS.register_module()
class KAISTDataset(CocoDataset):
    """KAIST Dataset for thermal-RGB detection."""
    METAINFO = {
        'classes': ('cyclist', 'person', 'people'),  # Default classes
        'palette': [
            (220, 20, 60),  # cyclist
            (119, 11, 32),  # person
            (0, 0, 142),  # people
        ]
    }

    def __init__(self, *args, include_empty_images=False, **kwargs):
        # Get classes from metainfo if provided in config
        if 'metainfo' in kwargs and 'classes' in kwargs['metainfo']:
            self.METAINFO = {
                'classes': kwargs['metainfo']['classes'],
                'palette': self.METAINFO['palette'][:len(kwargs['metainfo']['classes'])]
            }

        # Create category name to label mapping BEFORE super().__init__
        self.cat2label = {cat: i for i, cat in enumerate(self.METAINFO['classes'])}

        # Create category IDs
        self.cat_ids = [i for i in range(len(self.METAINFO['classes']))]

        # Flag to include images without annotations (for feature distillation)
        self._include_empty_images = include_empty_images

        super().__init__(*args, **kwargs)

    def load_data_list(self):
        """Load annotations from ann_file."""
        ann_file = osp.join(self.data_root, self.ann_file)
        # print(f"\nLoading annotations from: {ann_file}")

        with open(ann_file) as f:
            self.coco = json.load(f)

        # Create category name to id mapping from COCO annotations
        self.coco_cat_name_to_id = {cat['name']: cat['id']
                                    for cat in self.coco['categories']}

        # Create category to image mapping
        self.cat_img_map = {i: [] for i in range(len(self.METAINFO['classes']))}

        data_list = []
        for img_info in self.coco['images']:
            data_info = {}

            # Get file paths
            visible_path = img_info['file_name']
            lwir_path = img_info['file_name2']

            # Join with data root and img prefix
            if self.data_prefix.get('img_path', None):
                img_prefix = self.data_prefix['img_path']
            else:
                img_prefix = self.data_prefix.get('img', '')

            data_info['img_path'] = osp.join(self.data_root, img_prefix, visible_path)
            data_info['img2_path'] = osp.join(self.data_root, img_prefix, lwir_path)

            data_info['height'] = img_info['height']
            data_info['width'] = img_info['width']
            data_info['img_id'] = img_info['id']

            # Get annotations
            gt_bboxes = []
            gt_labels = []
            gt_ignore_flags = []

            # Get all annotations for this image
            img_anns = [ann for ann in self.coco['annotations']
                        if ann['image_id'] == img_info['id']]

            for ann in img_anns:
                # Get category name from original annotation
                category_name = next(cat['name'] for cat in self.coco['categories']
                                     if cat['id'] == ann['category_id'])

                # Skip if category is not in our selected classes
                if category_name not in self.METAINFO['classes']:
                    continue

                x1, y1, w, h = ann['bbox']
                bbox = [x1, y1, x1 + w, y1 + h]

                # Get new label index based on our selected classes
                label = self.cat2label[category_name]

                gt_bboxes.append(bbox)
                gt_labels.append(label)
                gt_ignore_flags.append(ann.get('ignore', False))

                # Add image to category mapping
                self.cat_img_map[label].append(img_info['id'])

            # Build instances list (may be empty for images with no annotations)
            data_info['instances'] = [{
                'bbox': bbox,
                'bbox_label': label,
                'ignore_flag': ignore_flag,
            } for bbox, label, ignore_flag in zip(gt_bboxes, gt_labels, gt_ignore_flags)]

            # Include image based on filter_empty_gt setting.
            # In test_mode, mmengine skips filter_data(), so we handle it here.
            # By default, only include images with annotations (original behavior).
            # Set include_empty_images=True in dataset config to include all images.
            if gt_bboxes or self._include_empty_images:
                data_list.append(data_info)

        print(f"\nLoaded {len(data_list)} images")
        return data_list

    def parse_data_info(self, raw_data_info):
        """Parse raw annotation to target format."""
        return raw_data_info

    def _join_prefix(self):
        """Join paths in data_root."""
        # Override to prevent the parent class from modifying our paths
        pass

    def prepare_data(self, idx):
        """Get data processed by ``self.pipeline``.
        
        This is modified to prepare data for cross-modal distillation.
        - The IR image path ('img2_path') is moved to 'img_path' so that the
          standard `LoadImageFromFile` transform processes it for the student.
        - The original RGB image path is preserved in 'img2_path' for the
          teacher model to use later.
        """
        data_info = self.get_data_info(idx)

        # The student model trains on IR images. The standard pipeline uses
        # the 'img_path' key. So, we swap the paths here.
        # 'img_path' will now point to the IR image.
        # 'img2_path' will now point to the RGB image.
        results = {
            'img_path': data_info['img2_path'],  # IR image for student
            'img2_path': data_info['img_path'],  # RGB image for teacher
            'img_id': data_info['img_id'],
            'height': data_info['height'],
            'width': data_info['width'],
            'instances': data_info.get('instances', [])
        }
        return self.pipeline(results)