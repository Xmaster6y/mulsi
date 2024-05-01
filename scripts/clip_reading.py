"""Simple FGSM attack."""

from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from transformers import CLIPModel, CLIPProcessor

from mulsi import AdversarialImage, DiffCLIPImageProcessor

####################
# HYPERPARAMETERS
####################
image_path = "assets/orange.jpg"
model_name = "openai/clip-vit-base-patch32"
epsilon = 0.1
####################

image = Image.open(image_path)
model = CLIPModel.from_pretrained(model_name)
model.eval()
for param in model.parameters():
    param.requires_grad = False
processor = CLIPProcessor.from_pretrained(model_name)
diff_processor = DiffCLIPImageProcessor(processor.image_processor)

image_tensor = pil_to_tensor(image).float().unsqueeze(0)
adv_image = AdversarialImage(image_tensor, model, None)
