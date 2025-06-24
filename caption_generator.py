#from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import torch

class CaptionGenerator:
#     def __init__(self):
#         self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
#         self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

#     def generate_caption(self, image):
#         inputs = self.processor(images=image, return_tensors="pt")
#         output = self.model.generate(**inputs)
#         caption = self.processor.decode(output[0], skip_special_tokens=True)
#         return caption
    def __init__(self):
        self.model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    
    def generate_caption(self, image):
        image = image.resize((224, 224))
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values

        with torch.no_grad():
            output_ids = self.model.generate(pixel_values, max_length=16, min_length=5, num_beams=1)
        caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return caption
