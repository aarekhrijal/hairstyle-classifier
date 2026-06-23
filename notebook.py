# Install libraries
!pip install -Uqq ddgs fastai

# Imports
from ddgs import DDGS
from fastcore.all import *
from fastai.vision.all import *
import time

def search_images(term, max_images=100):
    return L(DDGS().images(term, max_results=max_images)).itemgot('image')

# Download images
hairstyles = ['buzz cut hairstyle', 'wolf cut hairstyle', 'mullet hairstyle', 
              'pompadour hairstyle', 'mohawk hairstyle', 'quiff hairstyle', 
              'fohawk hairstyle', 'man bun hairstyle']

path = Path('hairstyles')

for style in hairstyles:
    dest = (path/style)
    dest.mkdir(exist_ok=True, parents=True)
    urls = search_images(style, max_images=100)
    download_images(dest, urls=urls)
    time.sleep(3)
    print(f'Done: {style}')

# Clean corrupted images
failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
print(f'Failed: {len(failed)}')

# Create dataloaders
dls = ImageDataLoaders.from_folder(path, valid_pct=0.2, seed=42,
                                    item_tfms=Resize(224))

# Train model
learn = vision_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(5)

# Check results
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(10,10))

# Export model
learn.export('/kaggle/working/hairstyle_classifier.pkl')
