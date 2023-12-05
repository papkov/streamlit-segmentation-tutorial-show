import streamlit as st
from streamlit.hello.utils import show_code
import numpy as np
import torch
from torchvision.transforms import v2
from torchvision import tv_tensors
from PIL import Image

import segmentation_models_pytorch as smp


def segmentation_demo():
    image = st.file_uploader("Choose an image")
    model_weights = st.file_uploader("Choose a model file")

    if image is not None:
        image = Image.open(image).convert("RGB")
        st.image(image, "Uploaded image")

    if model_weights is not None:
        # Define model as above since you saved only the weights
        model = ...
        ### BEGIN SOLUTION
        model = smp.Unet(
            encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        )
        ### END SOLUTION

        # Load model (be careful with `map_location`, read documentation above)
        ### BEGIN SOLUTION
        model.load_state_dict(torch.load(model_weights,
                                         map_location="cpu"))
        ### END SOLUTION

        # Switch to eval mode (as in test loop)
        ### BEGIN SOLUTION
        model.eval()
        ### END SOLUTION

        # Use the same transforms as above
        transforms = ...
        ### BEGIN SOLUTION
        transforms = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
        ])
        ### END SOLUTION

        if image is not None:
            # Apply transforms (don't forget to wrap image as tv_tensors.Image)
            tensor = ...

            ### BEGIN SOLUTION
            tensor = transforms(tv_tensors.Image(image))
            ### END SOLUTION

            # Add batch dimension
            tensor = tensor[None, ...]

            # Do the forward pass (don't forget `torch.no_grad()`!)
            ### BEGIN SOLUTION
            with torch.no_grad():
                mask = model(tensor).numpy().clip(0, 1)
            ### END SOLUTION

            # Show image
            st.image(mask[0, 0], "Predicted mask")

# Setup page
st.set_page_config(page_title="Segmentation Demo", page_icon="🔬")
st.markdown("# Segmentation Demo")
st.sidebar.header("Segmentation Demo")