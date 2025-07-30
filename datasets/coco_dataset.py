import os
import cv2
import numpy as np
from pycocotools.coco import COCO


class CocoDataset:

    def __init__(self, data_dir, data_type, seed=32, max_instances=-1, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.data_type = data_type
        ann_fp = '{}/annotations/instances_{}.json'.format(data_dir, data_type)
        self.coco = COCO(annotation_file=ann_fp)
        self.random = np.random.RandomState(seed=seed)
        self.max_instances = max_instances if max_instances > 0 else 100 # Default 

    def configure_targets(self, target_categories):
        try:
            target_cat_ids = self.coco.getCatIds(catNms=target_categories)
        except:
            cats = self.coco.loadCats(self.coco.getCatIds())
            cat_names =[cat['name'] for cat in cats]
            raise Exception("Incorrect categories passed"
                                ". Choose from the following: {}".format(cat_names))
        
        return target_cat_ids, self.coco.getImgIds(catIds=target_cat_ids)

    def load_image(self, img_id):
        img = self.coco.loadImgs([img_id])[0]

        img_filename = img['file_name']
        img_fp = '{}/images/{}/{}'.format(self.data_dir, self.data_type, img_filename)

        if not os.path.exists(img_fp):
            raise FileNotFoundError()
        
        img = cv2.imread(img_fp, -1)
        if img is None:
            raise ValueError(f"Image {img_fp} could not be loaded.")
        
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB format

    def get_sample(self, target_categories):
        target_cat_ids, img_ids = self.configure_targets(target_categories)

        while True:
            sample_img_id = self.random.choice(img_ids)
            sample_img = self.load_image(sample_img_id)
            h,w,_ = sample_img.shape

            ann_ids = self.coco.getAnnIds(imgIds=sample_img_id, 
                                        catIds=target_cat_ids, iscrowd=None)
            anns = self.coco.loadAnns(ann_ids)

            anns = sorted(anns, key=lambda x: x['area'], reverse=True)

            masks = []
            # Select the top_k largest annotations
            for ann in anns[:self.max_instances]:
                # Ensure the largest annotation is significant enough
                # to be considered a valid sample
                # This is a heuristic to avoid very small annotations
                # that might not be useful for training or evaluation
                if ann['area'] < 0.01 * h * w:
                    break
                mask = self.coco.annToMask(ann)

                # Convert the mask to a categorical mask
                mask[mask > 0] = target_cat_ids.index(ann['category_id']) + 1
                masks.append(mask)

            if len(masks) == 0:
                continue
            
            sample_mask = np.stack(masks, axis=-1)
            break

        # Ensure the sample image is in BGR format
        return sample_img, sample_mask

    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    data_dir = '/Users/shantanu/Datasets/coco-dataset'
    data_type = 'val2017'
    target_categories = ['person', 'cat']

    coco_dataset = CocoDataset(data_dir, data_type, seed=5122)

    fig = plt.figure()
    ax = plt.gca()

    while True:
        img, categorical_instance_masks = coco_dataset.get_sample(target_categories, top_k=3)
        mask_scale = int(255 / len(target_categories))

        # Plot top 3 masks
        mask_instances_color = np.zeros((*categorical_instance_masks.shape[:2], 3), dtype=np.uint8)
        mask_title = []
        for idx in range(categorical_instance_masks.shape[-1]):
            cat_idx = categorical_instance_masks[..., idx].max() - 1
            cat_color = (cat_idx + 1) * mask_scale
            mask_instances_color[categorical_instance_masks[..., idx] > 0] = [
                cat_color, cat_color, cat_color
            ]
            mask_title.append(target_categories[cat_idx])

        output_img = np.hstack([img, mask_instances_color])
        
        ax.imshow(output_img)
        ax.set_title(f"Masks: {', '.join(mask_title)}")
        ax.axis('off')
        fig.canvas.draw()
        
        plt.waitforbuttonpress()
