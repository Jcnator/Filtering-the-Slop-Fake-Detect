from pytorch_grad_cam import GradCAM as CAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import numpy as np
from PIL import Image

CAM_MAP_THRESHOLD = 0.2

def reshape_transform(tensor):
    # tensor = tensor[0]
    num_tokens, batch_size, embedding_dim = tensor.shape
    # Calculate grid size (height, width), excluding the class token
    height = width = int((num_tokens - 1) ** 0.5)

    if height * width != (num_tokens - 1):
        raise ValueError(
            f"Invalid number of tokens: {num_tokens - 1}. Cannot reshape to a grid."
        )

    # Exclude the class token (first token)
    tensor = tensor[1:, :, :]

    # Reshape to [batch_size, embedding_dim, height, width]
    reshaped = tensor.permute(1, 2, 0).reshape(batch_size, embedding_dim, height, width)

    return reshaped

def print_grads_hook(module, grad_input, grad_output):
    print(f"Grad Input: {grad_input}")
    print(f"Grad Output: {grad_output}")

    print(f"----------\nshapes - {grad_input[0].shape} - {grad_output[0].shape}")

def get_cam_visualization_and_output(model, target_layers, input_tensor, image, target_class):
    model = model.to("cuda")
    input_tensor = input_tensor.to("cuda")

    targets = [ClassifierOutputTarget(target_class)]
    
    # target_layers[0].register_backward_hook(print_grads_hook)

    with CAM(
        model=model, target_layers=target_layers
    ) as cam:
        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets,
            aug_smooth=True,
            eigen_smooth=True,
        )[0]

        # In this example grayscale_cam has only one image in the batch:
        visualization = show_cam_on_image(image, grayscale_cam, use_rgb=True)

        # You can also get the model outputs without having to redo inference
        model_outputs = cam.outputs
        visualization = (visualization*255).astype(np.uint8)
        
        cv2.imshow(
            "Grad-CAM Visualization", cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
        )
        cv2.waitKey(0)  # Wait for a key press to close the window
        cv2.destroyAllWindows()

        return visualization, model_outputs
