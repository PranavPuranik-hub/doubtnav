import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

class Perceiver:
    def __init__(self, model_name="nvidia/segformer-b0-finetuned-ade-512-512", device="cpu"):
        self.device = device
        
        # Load model and processor
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
        
        # 5 groups: 0: ignore, 1: traversable, 2: risky, 3: obstacle, 4: non_traversable
        self.group_mapping = self._create_group_mapping()
        
    def _create_group_mapping(self):
        traversable_kw = ['road', 'path', 'sidewalk', 'grass', 'field', 'sand', 'dirt track', 'earth']
        risky_kw = ['plant', 'rug', 'mud'] # mud-like
        obstacle_kw = ['tree', 'rock', 'person', 'car', 'wall', 'fence', 'building', 'boulder']
        non_traversable_kw = ['water', 'river', 'sea']
        ignore_kw = ['sky']
        
        mapping = np.zeros(len(self.id2label), dtype=np.uint8)
        
        for idx_str, label in self.id2label.items():
            idx = int(idx_str)
            label_lower = label.lower()
            
            if any(kw in label_lower for kw in traversable_kw):
                mapping[idx] = 1
            elif any(kw in label_lower for kw in risky_kw):
                mapping[idx] = 2
            elif any(kw in label_lower for kw in obstacle_kw):
                mapping[idx] = 3
            elif any(kw in label_lower for kw in non_traversable_kw):
                mapping[idx] = 4
            else:
                mapping[idx] = 0 # default ignore
                
            # Override for explicit ignore keywords
            if any(kw in label_lower for kw in ignore_kw):
                mapping[idx] = 0
                
        return mapping

    def predict(self, frame_bgr):
        # Resize to 512x384 (width x height)
        resized_frame = cv2.resize(frame_bgr, (512, 384))
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        
        # Process image
        # SegformerImageProcessor automatically rescales, normalizes etc.
        # We specify do_resize=False as we have resized already, or we let it handle the tensor formatting.
        inputs = self.processor(images=rgb_frame, return_tensors="pt", do_resize=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # shape (batch_size, num_classes, height, width)
            
            # Upsample logits to match input image size (384x512)
            logits = F.interpolate(
                logits,
                size=(384, 512), # (height, width)
                mode="bilinear",
                align_corners=False
            )
            
            probs = F.softmax(logits, dim=1) # (1, num_classes, 384, 512)
            
            # Calculate uncertainty (normalized entropy)
            # max_entropy = log(num_classes)
            log_probs = F.log_softmax(logits, dim=1)
            entropy = -torch.sum(probs * log_probs, dim=1) # (1, 384, 512)
            max_entropy = np.log(probs.shape[1])
            uncertainty = (entropy / max_entropy).squeeze(0).cpu().numpy()
            
            # Class predictions
            preds = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy() # (384, 512)
            
        # Map original classes to groups
        group_mask = self.group_mapping[preds]
        
        # Create overlay
        overlay = self._create_overlay(resized_frame, group_mask)
        
        return group_mask, uncertainty, overlay
        
    def _create_overlay(self, image_bgr, group_mask):
        # Colors in BGR
        color_map = np.zeros((5, 3), dtype=np.uint8)
        color_map[0] = [0, 0, 0]           # ignore
        color_map[1] = [128, 128, 0]       # traversable (Teal: #008080 -> BGR 128,128,0)
        color_map[2] = [0, 191, 255]       # risky (Amber: #FFBF00 -> BGR 0,191,255)
        color_map[3] = [80, 127, 255]      # obstacle (Coral: #FF7F50 -> BGR 80,127,255)
        color_map[4] = [144, 128, 112]     # blocked/non_traversable (Slate: #708090 -> BGR 144,128,112)
        
        colored_mask = color_map[group_mask]
        
        # Soft blend (50% image, 50% mask for non-ignore regions)
        alpha = 0.5
        ignore_mask = (group_mask == 0)
        
        overlay = cv2.addWeighted(image_bgr, 1 - alpha, colored_mask, alpha, 0)
        overlay[ignore_mask] = image_bgr[ignore_mask]
        
        return overlay

    def export_onnx(self, output_path="segformer.onnx"):
        dummy_input = torch.randn(1, 3, 512, 512, device=self.device) # Using processor default size or 384, 512?
        # Segformer standard model expects 512x512 input. Wait! SegFormer expects the same size it was trained on or 
        # any size, but it's typically fine with generic sizes. Let's use 384x512.
        dummy_input = torch.randn(1, 3, 384, 512, device=self.device)
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            input_names=["pixel_values"],
            output_names=["logits"],
            opset_version=14,
            dynamic_axes={'pixel_values': {0: 'batch_size', 2: 'height', 3: 'width'}, 'logits': {0: 'batch_size', 2: 'height', 3: 'width'}}
        )
        print(f"Exported ONNX model to {output_path}")
