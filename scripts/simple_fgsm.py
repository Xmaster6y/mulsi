"""Simple FGSM attack.
"""

from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from mulsi import Attack, DiffClipProcessor

####################
# HYPERPARAMETERS
####################
image_path = "assets/orange.jpg"
model_name = "openai/clip-vit-base-patch32"
epsilon = 0.1
####################

attack = Attack.from_name("fgsm", epsilon=epsilon)
image = Image.open(image_path)
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)
diff_processor = DiffClipProcessor(processor.image_processor)

# TODO: Have an objective function like embeddings distance
