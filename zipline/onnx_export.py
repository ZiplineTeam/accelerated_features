#!/usr/bin/env python3

import argparse
import os
import sys
import torch
import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image
import matplotlib.pyplot as plt

# Add parent directory to Python path to import modules
sys.path.append(os.path.abspath('.'))
from modules.xfeat_match_star import XFeat

def prepare_detect_input(im):
    """Convert image to grayscale tensor for detection."""
    im_np = np.array(im, dtype=np.float32)
    
    # Convert RGB to Grayscale using luminosity formula
    im_gray = 0.299 * im_np[:,:,0] + 0.587 * im_np[:,:,1] + 0.114 * im_np[:,:,2]
    
    # Convert to PyTorch tensor with batch and channel dimensions
    im_tensor = torch.from_numpy(im_gray).unsqueeze(0).unsqueeze(0)
    
    return im_tensor

def visualize_features(image_path, keypoints, descriptors, scales, output_path):
    """Visualize detected features and save to file."""
    print(f"Visualizing features and saving to: {output_path}")
    
    # Load original image
    image = cv2.imread(image_path)
    if image is None:
        # Try loading with PIL and convert
        pil_image = Image.open(image_path)
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    # Extract keypoints from the first batch (assuming batch size 1)
    if len(keypoints.shape) == 3:
        kpts = keypoints[0]  # Remove batch dimension
    else:
        kpts = keypoints
    
    if len(scales.shape) == 3:
        scale_vals = scales[0]  # Remove batch dimension
    else:
        scale_vals = scales
    
    print(f"Number of keypoints: {len(kpts)}")
    print(f"Keypoints shape: {kpts.shape}")
    print(f"Scales shape: {scale_vals.shape}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('XFeat Feature Detection Results', fontsize=16)
    
    # Plot 1: Original image
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Plot 2: Keypoints visualization
    axes[0, 1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 1].scatter(kpts[:, 0], kpts[:, 1], c='red', s=10, alpha=0.7)
    axes[0, 1].set_title(f'Detected Keypoints ({len(kpts)} points)')
    axes[0, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save a simple keypoints-only image
    simple_output = output_path.replace('.png', '_keypoints_only.png')
    cv2_keypoints = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=5) for pt in kpts]
    keypoint_image = cv2.drawKeypoints(image, cv2_keypoints, None, 
                                     color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(simple_output, keypoint_image)
    
    print(f"Visualization saved to: {output_path}")
    print(f"Simple keypoints image saved to: {simple_output}")

def validate_onnx(onnx_path, test_image_path, save_visualization=True):
    """Validate ONNX model by running inference on a test image."""
    print(f"Validating ONNX model with test image: {test_image_path}")
    
    # Load and prepare test image
    test_image = Image.open(test_image_path)
    test_input = prepare_detect_input(test_image)
    test_input_np = test_input.cpu().detach().numpy()
    
    # Create ONNX runtime session
    session = ort.InferenceSession(onnx_path)
    
    # Run inference
    inputs = {'img': test_input_np}
    outputs = session.run(None, inputs)
    
    # Print output shapes to verify model is working
    print("\nOutput shapes:")
    for i, output in enumerate(outputs):
        print(f"Output {i}: {output.shape}")
    
    # Extract outputs
    keypoints = outputs[0]  # Shape: [batch, num_keypoints, 2]
    descriptors = outputs[1]  # Shape: [batch, num_keypoints, descriptor_dim]
    scales = outputs[2]  # Shape: [batch, num_keypoints, 1]
    
    print(f"\nFeature extraction results:")
    print(f"Keypoints: {keypoints.shape}")
    print(f"Descriptors: {descriptors.shape}")
    print(f"Scales: {scales.shape}")
    
    if save_visualization:
        # Create output filename based on input image and model
        base_name = os.path.splitext(os.path.basename(test_image_path))[0]
        model_name = os.path.splitext(os.path.basename(onnx_path))[0]
        output_path = f"features_visualization_{model_name}_{base_name}.png"
        
        visualize_features(test_image_path, keypoints, descriptors, scales, output_path)
    
    print("\nValidation successful!")

def main():
    parser = argparse.ArgumentParser(description='Export XFeat model to ONNX format')
    parser.add_argument('--output', type=str, default='xfeat_star_detect.onnx',
                      help='Output path for ONNX model')
    parser.add_argument('--validate', action='store_true',
                      help='Validate ONNX model with test image')
    parser.add_argument('--test-image', type=str, default='../assets/test_image.jpg',
                      help='Path to test image for validation')
    parser.add_argument('--save-visualization', action='store_true', default=True,
                      help='Save feature visualization to file during validation')
    parser.add_argument('--no-visualization', action='store_true',
                      help='Disable feature visualization saving')
    args = parser.parse_args()

    # Handle visualization flag
    save_viz = args.save_visualization and not args.no_visualization

    # Initialize XFeat model
    print("Initializing XFeat model...")
    xfeat_model = XFeat()
    xfeat_model.eval()

    # Move model to CPU for export
    device = torch.device('cpu')
    xfeat_model = xfeat_model.to(device)
    
    # Print model device information
    print("\nModel device information:")
    print(f"Model device: {next(xfeat_model.parameters()).device}")
    print("Parameter devices:")
    for name, param in xfeat_model.named_parameters():
        print(f"{name}: {param.device}")

    # Create wrapper for detection
    class DetectXFeatWrapper(torch.nn.Module):
        def __init__(self, model):
            super(DetectXFeatWrapper, self).__init__()
            self.model = model

        @torch.inference_mode()
        def forward(self, img):
            # Ensure input is on the correct device
            model_device = next(self.model.parameters()).device
            if img.device != model_device:
                img = img.to(model_device)
            return self.model.xfeat_detect(img)

    detect_xfeat_wrapper = DetectXFeatWrapper(xfeat_model)
    detect_xfeat_wrapper = detect_xfeat_wrapper.to(device)
    
    # Print wrapper device information
    print("\nWrapper device information:")
    print(f"Wrapper device: {next(detect_xfeat_wrapper.parameters()).device}")
    print("Wrapper parameter devices:")
    for name, param in detect_xfeat_wrapper.named_parameters():
        print(f"{name}: {param.device}")

    # Create dummy input for export
    dummy_input = torch.randn(1, 1, 480, 640, device=device)  # Example size, will be dynamic
    print(f"\nDummy input device: {dummy_input.device}")

    # Export to ONNX on CPU
    print(f"\nExporting model to {args.output} on CPU...")
    torch.onnx.export(
        detect_xfeat_wrapper,
        dummy_input,
        args.output,
        export_params=True,
        opset_version=11,
        training=torch.onnx.TrainingMode.EVAL,
        do_constant_folding=True,
        input_names=['img'],
        output_names=['keypoints', 'descriptors', 'scales'],
        dynamic_axes={
            'img': {2: 'height', 3: 'width'},
            'keypoints': {1: 'num_keypoints'},
            'descriptors': {1: 'num_keypoints'},
            'scales': {1: 'num_keypoints'}
        }
    )
    print("Export completed successfully!")

    # Validate if requested
    if args.validate:
        validate_onnx(args.output, args.test_image, save_visualization=save_viz)

if __name__ == '__main__':
    main() 