import gradio as gr
from fastai.vision.all import *
import traceback

learn = load_learner('hairstyle_classifier.pkl', cpu=True)
labels = learn.dls.vocab

def predict(img):
    try:
        img = img.convert('RGB')
        pred, pred_idx, probs = learn.predict(img)
        return {labels[i]: float(probs[i]) for i in range(len(labels))}
    except Exception as e:
        return {"ERROR: " + str(e): 1.0}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=8),
    title="Hairstyle Classifier",
    description="Upload a photo to classify the hairstyle"
)

demo.launch()
