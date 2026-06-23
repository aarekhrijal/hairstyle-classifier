# Hairstyle Classifier

An image classifier that identifies 8 different hairstyles using deep learning.

## Categories
- Buzz cut
- Wolf cut
- Mullet
- Pompadour
- Mohawk
- Quiff
- Fohawk
- Man bun

## Approach
- Collected 85-144 images per category using image search
- Trained using fastai with ResNet34 (transfer learning)
- Fine-tuned for 5 epochs

## Results
- Final error rate: 4.4%
- Validation accuracy: ~95.6%

## Tech Stack
- Python, fastai, PyTorch
- Trained on Kaggle (free GPU)

## Files
- `notebook.py` - training code
- `app.py` - Gradio deployment interface
- `hairstyle_classifier.pkl` - trained model
