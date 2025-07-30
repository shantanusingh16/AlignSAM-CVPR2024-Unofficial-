import numpy as np
import glob
import re
import cv2
import gymnasium as gym
from gymnasium import spaces

import sys
import os
sys.path.append(os.path.abspath(os.getcwd()))

from custom_gym_implns.envs.utils.repvit_sam_wrapper import RepVITSamWrapper
from datasets import get_dataset


class SamSegEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 1}

    def __init__(self, img_shape, embedding_shape, mask_shape, render_frame_shape,
                 max_steps, target_categories, dataset_config, sam_ckpt_fp, 
                 penalize_for_wrong_input=True, use_dice_score=False,
                 img_patch_size=None, num_patches=None, render_mode='rgb_array'):
        
        assert len(img_shape) == 3, "Image shape should be (H, W, C)"
        assert len(embedding_shape) == 3, "Embedding shape should be (C, H, W)"
        assert len(mask_shape) == 2, "Mask shape should be (H, W)"
        assert len(render_frame_shape) == 2, "Render frame shape should be (H, W)"
        assert max_steps is None or max_steps > 0, "Max steps should be None or > 0"
        
        assert (img_patch_size is not None and num_patches is None) or \
            (img_patch_size is None and num_patches is not None), \
            "Either img_patch_size or num_patches should be provided, not both"

        assert render_mode in self.metadata["render_modes"], "Invalid render mode"


        self.img_shape = img_shape  # The size of the image (HxWxC)
        self.embedding_shape = embedding_shape  # The size of the SAM image encoder output
        self.mask_shape = mask_shape  # The size of the mask (HxW)
        self.render_frame_shape = render_frame_shape  # The size of the frame to render

        self.max_steps = max_steps  # The maximum number of steps the agent can take
        self.penalize_for_wrong_input = penalize_for_wrong_input  # Whether to penalize for wrong input
        self.use_dice_score = use_dice_score

        self.target_categories = target_categories  # The categories to segment
        self.dataset_config = dataset_config
        self.dataset = get_dataset(self.dataset_config)

        self.img_patch_size = img_patch_size

        self._image = None
        self._curr_target_cat = None
        self._sam_image_embeddings = None
        self._sam_pred_mask_prob = None
        self._sam_pred_mask = None
        self._categorical_instance_masks = None # GT binary instance masks with cat-ids as values instead of 1
        self._num_steps = 0
        self._last_actions = {'input_points':[], 'input_labels':[]}
        self._last_reward = 0
        self._last_best_score = 0
        self.render_mode = render_mode

        self.sam_predictor = RepVITSamWrapper(sam_ckpt_fp)

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(0, 255, shape=(*img_shape,), dtype=np.uint8),
                "target_category": spaces.Text(max_length=max(map(len, target_categories))),
                "sam_image_embeddings": spaces.Box(-np.inf, np.inf, shape=(*embedding_shape,), dtype=np.float32),
                "sam_pred_mask_prob": spaces.Box(0, 1, shape=(*mask_shape,), dtype=np.float32),
            }
        )

        img_h, img_w = img_shape[:2]
        if img_patch_size is not None:
            width_patch_size = height_patch_size = img_patch_size
            # number of patches along a single dimension is the ceiling of the division of the image size by the patch size
            num_width_patches = img_w//width_patch_size + int(img_w%width_patch_size != 0)
            num_height_patches = img_h//height_patch_size + int(img_h%height_patch_size != 0)
        elif num_patches is not None:
            num_width_patches = num_height_patches = num_patches
            width_patch_size = img_w / num_width_patches
            height_patch_size = img_h / num_height_patches

            self.img_patch_size = max(width_patch_size, height_patch_size) 
        else:
            raise ValueError("Either img_patch_size or num_patches should be provided")
        # print(num_width_patches, num_height_patches)

        # The following dictionary maps abstract actions from `self.action_space` to
        # the input_points and input_labels submitted to SAM in if that action is taken.
        # I.e. 0 corresponds to "(0, 0), positive", 1 to "(0, 0), negative",
        # 2 to "(0, 1), positive", 3 to "(0, 1), negative", etc
        
        self._action_to_input = {}
        for wpatch_idx in range(0, num_width_patches):
            for hpatch_idx in range(0, num_height_patches):
                # Mark patch center as input point (x, y)
                patch_center_x = int(wpatch_idx * width_patch_size + width_patch_size/2)
                patch_center_y = int(hpatch_idx * height_patch_size + height_patch_size/2)
                input_point = (patch_center_x, patch_center_y)
                
                self._action_to_input[len(self._action_to_input)] = (input_point, 'pos')
                self._action_to_input[len(self._action_to_input)] = (input_point, 'neg')

        # Add 2nd last action to remove previous input
        # self._action_to_input[len(self._action_to_input)] = ((1e6,1e6), 'remove')
        # Add last action to mark the task as done
        # self._action_to_input[len(self._action_to_input)] = ((1e6, 1e6), 'done')
        self.action_space = spaces.Discrete(len(self._action_to_input))

        self._load_sample_from_dataset()


    def _load_sample_from_dataset(self):
        # Choose a random sample from the target categories
        self._curr_target_cat = np.random.choice(self.target_categories)

        # Expects the image to be in RGB format
        img, categorical_instance_masks = self.dataset.get_sample(
            target_categories=[self._curr_target_cat])

        self._image = cv2.resize(img, self.img_shape[:2][::-1])
        self.sam_predictor.set_image(self._image)
        self._sam_image_embeddings = self.sam_predictor.get_image_embeddings()
        self._sam_pred_mask_prob = np.zeros(self.mask_shape, dtype=np.float32)
        self._sam_pred_mask = np.zeros(self.img_shape[:2], dtype=np.float32)

        resized_mask = cv2.resize(categorical_instance_masks, 
                                  self.img_shape[:2][::-1], 
                                  interpolation=cv2.INTER_NEAREST)
        
        # Ensure the mask has 3 dimensions (H, W, K)
        if len(resized_mask.shape) == 2:
            resized_mask = resized_mask[..., np.newaxis]
        self._categorical_instance_masks = resized_mask


    def _get_obs(self):
        return {
            "image": self._image, 
            "target_category": self._curr_target_cat,
            "sam_image_embeddings": self._sam_image_embeddings,
            "sam_pred_mask_prob": self._sam_pred_mask_prob,
        }
    

    def _get_info(self):
        return {
            "last_input_points": self._last_actions["input_points"].copy(),
            "last_input_labels": self._last_actions["input_labels"].copy(),
            "sam_pred_mask": self._sam_pred_mask.copy(),
        }
    

    def run_sam(self):
        masks, ious, low_res_mask_logits = self.sam_predictor.predict(
            np.array(self._last_actions["input_points"]), 
            np.array(self._last_actions["input_labels"])
        )

        best_mask_idx = np.argmax(ious)
        best_pred_mask_prob = 1 / (1 + np.exp(-low_res_mask_logits[best_mask_idx]))
        self._sam_pred_mask_prob = best_pred_mask_prob
        self._sam_pred_mask = (1 / (1 + np.exp(-masks[best_mask_idx]))).astype(np.float32)


    def convert_raw_input_to_action(self, input_point, input_label):
        point_dist = lambda x: np.linalg.norm([input_point[0]-x[0], input_point[1]-x[1]])
        input_dist = np.array([point_dist(point) + (1e6 * (label != input_label)) 
                               for (point, label) in self._action_to_input.values() 
                               if type(point) == tuple])
        sample_action = np.argmin(input_dist)
        return sample_action


    def compute_reward(self, pred_mask, act):
        # Compute dice score
        resized_gt_mask = cv2.resize(self._categorical_instance_masks,
                                     pred_mask.shape[::-1],
                                     interpolation=cv2.INTER_NEAREST)
        # Ensure the resized_gt_mask has 3 dimensions (H, W, K)
        if len(resized_gt_mask.shape) == 2:
            resized_gt_mask = resized_gt_mask[..., np.newaxis]

        if self.use_dice_score:
            eps = 1e-6
            # Flatten the binary masks per instance to compute the dice score
            gt_mask = np.any(resized_gt_mask > 0, axis=-1).astype(np.float32)
            intersection = np.sum(gt_mask * pred_mask)
            union = np.sum(gt_mask) + np.sum(pred_mask) #- intersection
            dice_score = (2 * intersection + eps)/ (union + eps)

            dice_reward = dice_score - self._last_best_score
            self._last_best_score = max(dice_score, self._last_best_score)
        else:
            dice_reward = 0.0

        correct_input_reward = 0
        if act == 'add':
            input_point = self._last_actions["input_points"][-1]
            input_label = self._last_actions["input_labels"][-1]

            # Check if the input is repeated
            if len(self._last_actions["input_points"]) > 1:
                point_dist = lambda x: np.linalg.norm([input_point[0]-x[0], input_point[1]-x[1]])
                input_dist = np.array([point_dist(point) + (1e6 * (label != input_label)) \
                                       for (point, label) in zip(self._last_actions["input_points"][:-1], 
                                                                 self._last_actions["input_labels"][:-1])])
                # Use 2 * img_patch_size as a threshold
                dist_coeff = 1 - np.exp(-np.min(input_dist) / (2 * self.img_patch_size))
            else:
                dist_coeff = 1.0

            point_image_indices = tuple(map(int, (input_point[1], input_point[0])))
            # print(point_image_indices, input_label)

            # Check across instance masks to see if its background or foreground
            gt_label = np.max(self._categorical_instance_masks[point_image_indices] > 0)

            if gt_label == 1:
                # Reward for correct input for positive class
                correct_input_reward = int(input_label == gt_label)
            elif self.penalize_for_wrong_input:
                # Penalize for wrong input for negative class
                correct_input_reward = -1 * int(input_label != gt_label)

            # # Check if too many negative inputs are given
            # num_input_label = np.sum(np.array(self._last_actions["input_labels"]) == input_label)
            # if num_input_label > int(self.max_steps * 0.5):
            #     correct_input_reward = 0  # Penalize for too many same input types (even if the last input was correct)

            correct_input_reward *= dist_coeff

            # Only add dice reward if the input is negative
            # Boosts using negative for refining masks
            correct_input_reward += int(dice_reward > 0.01) * (input_label == 0) * 0.5

        # reward = dice_reward + correct_input_reward
        reward = correct_input_reward

        # # Add bonus reward if dice score is above a threshold and not the first action
        # dice_reward_coefficient = 2.0 if len(self._last_actions)> 0 else 0.0
        # reward += int((dice_reward > 0.05)) * dice_reward_coefficient

        self._last_reward = reward

        return reward


    def step(self, action):
        if action < 0 or action >= self.action_space.n:
            raise ValueError(f"Invalid action: {action}")
        
        input_point, input_type = self._action_to_input[action]
        
        terminated = input_type == 'done'
        trunc = (self.max_steps is not None and self._num_steps >= self.max_steps)
        if terminated or trunc:
            return self._get_obs(), 0, terminated, trunc, self._get_info()
        
    
        act = None
        if input_type == 'remove':
            # Remove the last input
            if len(self._last_actions["input_points"]) > 0:
                self._last_actions["input_points"].pop()
                self._last_actions["input_labels"].pop()
                act = 'remove'
        elif input_type in ['pos', 'neg']:
            # Add a new input point
            input_label = 1 if input_type == 'pos' else 0 # label 0-neg, 1-pos

            self._last_actions["input_points"].append(input_point)
            self._last_actions["input_labels"].append(input_label)
            act = 'add'
        elif input_type == 'next':
            #TODO : Not implemented yet, 
            # Is meant to be used for predicting the next instance mask
            act = 'next'
            self._last_actions = {'input_points':[], 'input_labels':[]}
        else:
            raise ValueError(f"Invalid input type: {input_type}")

        if len(self._last_actions["input_points"]) > 0:
            self.run_sam()
        else:
            # Set the mask and mask_prob to initial state
            self._sam_pred_mask_prob = np.zeros(self.mask_shape, dtype=np.float32)
            self._sam_pred_mask = np.zeros(self.img_shape[:2], dtype=np.float32)
        
        self._num_steps += 1

        reward = self.compute_reward(self._sam_pred_mask, act)

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, False, trunc, info
    

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self._load_sample_from_dataset()

        self._last_actions = {'input_points':[], 'input_labels':[]}
        self._num_steps = 0
        self._last_reward = 0
        self._last_best_score = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    

    def render(self):
        img = cv2.resize(cv2.cvtColor(self._image, cv2.COLOR_RGB2BGR), self.render_frame_shape[::-1])

        mask_instances_color = np.zeros((*self._categorical_instance_masks.shape[:2], 3), dtype=np.uint8)
        num_instances = self._categorical_instance_masks.shape[-1]
        for idx in range(num_instances):
            instance_color = (idx + 1) * int(255 / num_instances)
            mask_instances_color[self._categorical_instance_masks[..., idx] > 0] = [
                instance_color, instance_color, instance_color
            ]
        gt_mask = cv2.resize(mask_instances_color, self.render_frame_shape[::-1])


        mask = cv2.cvtColor((self._sam_pred_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for point,label in zip(self._last_actions["input_points"], self._last_actions["input_labels"]):
            cv2.circle(mask, point, 10, (0, 255, 0) if label == 1 else (0, 0, 255), -1)
        mask = cv2.resize(mask, self.render_frame_shape[::-1])
        
        concat_img = np.concatenate([img, gt_mask, mask], axis=1)

        if self.render_mode == 'rgb_array':
            concat_img = cv2.cvtColor(concat_img, cv2.COLOR_BGR2RGB)

        # Add the target category text to the image
        cv2.putText(concat_img,
                    self._curr_target_cat,
                    (10, concat_img.shape[0]//2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2)
        
        # Add the last reward text to the image
        cv2.putText(concat_img,
                    f"Reward: {self._last_reward:.2f}",
                    (10, concat_img.shape[0]//2 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2)
        return concat_img


if __name__ == "__main__":
    import os

    img_shape = (375, 500, 3) # HxWxC
    embedding_shape = (256, 64, 64) # CxHxW
    mask_shape = (256, 256) # HxW
    render_frame_shape = (320, 426) # HxW
    max_steps = 5
    penalize_for_wrong_input = False 
    use_dice_score = True
    img_patch_size = 32
    render_mode = 'human'

    target_categories = ['person', 'cat', 'dog', 'car', 'bicycle', 'bus']
    dataset_config = {
        'type': 'coco',
        'data_dir': os.path.join(os.getcwd(), 'data', 'coco-dataset'),
        'data_type': 'val2017',
        'seed': 42,
        'max_instances': 5,  # Maximum number of instances per image
    }

    sam_ckpt_fp = os.path.join(os.getcwd(), 'RepViT', 'sam', 'weights', 'repvit_sam.pt')

    env = SamSegEnv(img_shape=img_shape, 
                    embedding_shape=embedding_shape,
                    mask_shape=mask_shape,
                    render_frame_shape=render_frame_shape,
                    max_steps=max_steps,
                    target_categories=target_categories,
                    dataset_config=dataset_config,
                    penalize_for_wrong_input=penalize_for_wrong_input,
                    use_dice_score=use_dice_score,
                    sam_ckpt_fp=sam_ckpt_fp,
                    img_patch_size=img_patch_size,
                    render_mode=render_mode)

    print(env.action_space.n)
    obs, info = env.reset()
    
    global sample_action
    sample_action = env.action_space.n - 2 #np.random.randint(0, env.action_space.n)

    def get_action(event,x,y,flags,param):
        global sample_action
        if event == cv2.EVENT_LBUTTONUP:
            tgt_label = 'pos'
        elif event == cv2.EVENT_RBUTTONUP:
            tgt_label = 'neg'
        else:
            return
        # Ensure the coordinates are within the image bounds
        # Since we concat the images horizontally, we need to adjust the coordinates
        x = x % render_frame_shape[1]
        y = y % render_frame_shape[0]

        # Scale the coordinates to match the original image size
        scaled_x = int(x * img_shape[1] / render_frame_shape[1])
        scaled_y = int(y * img_shape[0] / render_frame_shape[0])

        # Convert the raw input to an action
        sample_action = env.convert_raw_input_to_action((scaled_x, scaled_y), tgt_label)


    cv2.namedWindow('image', cv2.WINDOW_GUI_NORMAL)
    cv2.setMouseCallback('image', get_action)
    total_reward = 0
    while True:
        obs, reward, done, trunc, info = env.step(sample_action)
        print('reward:', reward)
        total_reward += reward

        # print(obs['image'].shape, info['sam_pred_mask'].shape)
        print(info['last_input_labels'], info['last_input_points'])

        if done or trunc:
            print("Task done, Total reward:", total_reward)
            print("-"*50)
            total_reward = 0
            obs, info = env.reset()

        frame = env.render()
        cv2.imshow('image', frame)

        sample_action = env.action_space.n - 1  # Mark task as done by default

        if cv2.waitKey(0) == 27:  # Esc key to stop the loop
            break


    cv2.destroyAllWindows()



    