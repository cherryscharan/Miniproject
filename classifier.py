import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
import io
import logging
import os
import json
import torch.nn as nn

logger = logging.getLogger(__name__)

class ZooLensClassifier:
    """
    Classifier class using PyTorch MobileNetV2 for image classification.
    """
    def __init__(self, model_filename='zoolens_64_species_best.pth', classes_filename='classes.json'):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Base dir relative to this file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.model_path = os.path.join(base_dir, model_filename)
        self.classes_path = os.path.join(base_dir, classes_filename)
        
        # Check if custom model exists
        if os.path.exists(self.model_path) and os.path.exists(self.classes_path):
            logger.info(f"Loading custom model from {self.model_path}...")
            self.model = models.mobilenet_v2(weights=None)
            # Recreate the head configuration used in training
            self.model.classifier[1] = nn.Linear(self.model.last_channel, 64)
            
            # Load weights
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            # Load custom labels
            with open(self.classes_path, 'r') as f:
                self.labels = json.load(f)
            logger.info("Custom model and labels loaded successfully.")
            
        else:
            logger.warning(f"Custom model not found at {self.model_path}. Loading generic MobileNetV2 (ImageNet)...")
            self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            self.model.to(self.device)
            self.model.eval()
            self.labels = models.MobileNet_V2_Weights.IMAGENET1K_V1.meta["categories"]
            logger.info("Generic MobileNetV2 loaded.")

    def _load_labels(self):
        # Deprecated/Unused helper in this new logic, but keeping for reference if needed
        pass

    def preprocess_image(self, image_bytes):
        """
        Preprocesses the image bytes to a tensor compatible with MobileNetV2.
        Args:
            image_bytes (bytes): Raw image data.
        Returns:
            torch.Tensor: Preprocessed image batch.
        """
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
        return input_batch

    def predict(self, image_bytes, top_k=3):
        """
        Returns top K predictions for the given image bytes.
        
        Args:
            image_bytes (bytes): Raw image data.
            top_k (int): Number of top predictions to return.
            
        Returns:
            list: List of dictionaries with 'species' and 'confidence'.
        """
        try:
            input_batch = self.preprocess_image(image_bytes).to(self.device)

            with torch.no_grad():
                output = self.model(input_batch)

            # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
            # Get top k
            top_prob, top_catid = torch.topk(probabilities, top_k)
            
            results = []
            for i in range(top_prob.size(0)):
                score = top_prob[i].item()
                label_idx = top_catid[i].item()
                label = self.labels[label_idx]
                
                results.append({
                    "species": label.replace('_', ' ').title(),
                    "confidence": score
                })
            
            logger.debug(f"Top prediction: {results[0]['species']} ({results[0]['confidence']:.2f})")
            return results
        except Exception as e:
            logger.error(f"Prediction Error: {e}")
            return []
