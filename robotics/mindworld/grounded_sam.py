import numpy as np
from typing import Tuple, List
import argparse
import os
import copy

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert

# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.inference import annotate, load_image, predict

import supervision as sv

# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt


import torch
device = 'cuda:0'


from huggingface_hub import hf_hub_download

def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model   

# Use this command for evaluate the Grounding DINO model
# Or you can download the model by yourself
ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"


class GroundSAMInterface:
    def __init__(self) -> None:
        self.groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)
        if not os.path.exists('sam_vit_h_4b8939.pth'):
            os.system('wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth')
        device = 'cuda:0'
        sam_checkpoint = 'sam_vit_h_4b8939.pth'
        sam = build_sam(checkpoint=sam_checkpoint)
        sam.to(device=device)
        sam_predictor = SamPredictor(sam)
        self.sam_predictor = sam_predictor

    def dino(self, image: np.ndarray, text_prompt: str):
        TEXT_PROMPT = text_prompt
        BOX_TRESHOLD = 0.25
        TEXT_TRESHOLD = 0.25

        

        def load_image(image: np.ndarray):
            transform = T.Compose(
                [
                    T.RandomResize([800], max_size=1333),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            #image_source = Image.open(image_path).convert("RGB")
            image_source = Image.fromarray(np.ascontiguousarray(image)).convert("RGB")
            image = np.asarray(image_source)
            image_transformed, _ = transform(image_source, None)
            return image, image_transformed

        image_source, image = load_image(image)

        boxes, logits, phrases = predict(
            model=self.groundingdino_model, 
            image=image, 
            caption=TEXT_PROMPT, 
            box_threshold=BOX_TRESHOLD, 
            text_threshold=TEXT_TRESHOLD
        )

        #annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        #annotated_frame = annotated_frame[...,::-1] # BGR to RGB
        return image_source, boxes, logits, phrases

    def segment(self, image_source: np.ndarray, boxes):
        self.sam_predictor.set_image(image_source)

        H, W, _ = image_source.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])


        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(device)
        masks, _, _ = self.sam_predictor.predict_torch(
                    point_coords = None,
                    point_labels = None,
                    boxes = transformed_boxes,
                    multimask_output = False,
                )
        return masks, boxes_xyxy

    def annotate(self, image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, phrases: List[str]) -> np.ndarray:
        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        return annotated_frame

    def run(self, image: np.ndarray, text_prompt: str):
        image_source, boxes, logits, phrases = self.dino(image=image, text_prompt=text_prompt)
        masks, boxes_xyxy = self.segment(image_source, boxes)
        # print(np.array(masks).shape)
        return image_source, boxes_xyxy, logits, phrases, masks

    def show_mask(self, mask, image, random_color=True):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.detach().cpu().numpy().reshape(h, w, 1) * color.reshape(1, 1, -1)
        
        annotated_frame_pil = Image.fromarray(image).convert("RGBA")
        mask_image_pil = Image.fromarray((mask_image * 255).astype(np.uint8)).convert("RGBA")

        return Image.alpha_composite(annotated_frame_pil, mask_image_pil)


if __name__ == '__main__':
    interface = GroundSAMInterface()
    image_path = os.path.join(os.path.dirname(__file__), '../../third_party/Grounded-Segment-Anything/assets/inpaint_demo.jpg')
    image = cv2.imread(image_path)[...,::-1]
    image_source, boxes, logits, phrases, masks = interface.run(image=image, text_prompt="dog, bench")