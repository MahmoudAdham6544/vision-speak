# 🎯 VisionSpeak – Image Captioning with GPT-2

VisionSpeak is a deep learning project that bridges computer vision and natural language processing to automatically generate descriptive captions for images. Built using a Vision-Encoder-Decoder architecture with a CNN-based visual encoder and a GPT-2 text decoder, this project explores the intersection of visual perception and language modeling.

![VisionSpeak Demo](demo.gif) <!-- You can add a gif or image here -->

---

## 📌 Why VisionSpeak?

With the exponential growth of visual content, there is a rising demand for intelligent systems that can understand and describe images in natural language. VisionSpeak simulates a human-like interpretation of images and transforms visual scenes into coherent, concise textual descriptions.

Whether you're building assistive tech for the visually impaired or creating metadata for image datasets, this project demonstrates the power and elegance of multi-modal AI.

---

## 🧠 What It Does

- 🖼️ Accepts image inputs (or crops of larger images)
- 📚 Encodes the visual features using a pretrained vision transformer (ViT)
- 📝 Decodes the embeddings into fluent captions using GPT-2
- 🗣️ Outputs a human-readable sentence describing the image content

---

## 🚀 Quick Demo

```bash
# Clone the repo
git clone https://github.com/yourusername/vision_speak.git
cd vision_speak

# Activate environment (assuming conda)
conda activate vision_speak

# Run the main script
python main.py --image_path path/to/image.jpg

