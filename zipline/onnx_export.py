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
import time

# Add parent directory to Python path to import modules
sys.path.append(os.path.abspath('.'))
from modules.xfeat_match_star import XFeat

def setup_onnx_session(onnx_path, session_name="ONNX"):
    """Set up ONNX runtime session with CUDA fallback."""
    # Try to create ONNX runtime session with CUDA first
    providers = []
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        providers.append('CUDAExecutionProvider')
        print(f"Using CUDA for {session_name} inference")
    else:
        print(f"Warning: CUDA not available for {session_name} inference, falling back to CPU")
    
    providers.append('CPUExecutionProvider')
    
    try:
        session = ort.InferenceSession(onnx_path, providers=providers)
        print(f"{session_name} using provider: {session.get_providers()[0]}")
        return session
    except Exception as e:
        print(f"Warning: Failed to create {session_name} session with preferred providers, using default: {e}")
        return ort.InferenceSession(onnx_path)

def prepare_detect_input(im):
    """Convert image to grayscale tensor for detection."""
    im_np = np.array(im, dtype=np.float32)
    
    # Convert RGB to Grayscale using luminosity formula
    im_gray = 0.299 * im_np[:,:,0] + 0.587 * im_np[:,:,1] + 0.114 * im_np[:,:,2]
    
    # Convert to PyTorch tensor with batch and channel dimensions
    im_tensor = torch.from_numpy(im_gray).unsqueeze(0).unsqueeze(0)
    
    return im_tensor

def prepare_match_inputs(output1, output2):
    """Prepare inputs for matcher from detector outputs."""
    # Create IDs for keypoints
    dec1_ids_np = np.arange(1, output1[0].shape[1] + 1).astype(np.float32).reshape(1, -1)
    dec1_ids_np = np.expand_dims(dec1_ids_np, axis=-1)
    
    dec1_kps_np = output1[0].astype(np.float32)
    dec1_desc_np = output1[1].astype(np.float32)
    dec1_sc_np = output1[2].astype(np.float32)
    dec1_sc_np = np.expand_dims(dec1_sc_np, axis=-1)
    
    dec2_ids_np = np.arange(1, output2[0].shape[1] + 1).astype(np.float32).reshape(1, -1)
    dec2_ids_np = np.expand_dims(dec2_ids_np, axis=-1)
    
    dec2_kps_np = output2[0].astype(np.float32)
    dec2_desc_np = output2[1].astype(np.float32)
    
    return dec1_ids_np, dec1_kps_np, dec1_desc_np, dec1_sc_np, dec2_ids_np, dec2_kps_np, dec2_desc_np

def warp_corners_and_draw_matches(ref_points, dst_points, img1, img2):
    """Visualize matches between two images with homography warping."""
    # Calculate the Homography matrix
    H, mask = cv2.findHomography(ref_points, dst_points, cv2.USAC_MAGSAC, 3.5, maxIters=1000, confidence=0.999)
    mask = mask.flatten() > 0

    # Get corners of the first image (image1)
    h, w = img1.shape[:2]
    corners_img1 = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32).reshape(-1, 1, 2)

    # Warp corners to the second image (image2) space using the Homography matrix
    warped_corners = cv2.perspectiveTransform(corners_img1, H)

    # Draw the warped corners in image2
    img2_with_corners = img2.copy()
    for i in range(len(warped_corners)):
        start_point = tuple(warped_corners[i - 1][0].astype(int))
        end_point = tuple(warped_corners[i][0].astype(int))
        cv2.line(img2_with_corners, start_point, end_point, (0, 255, 0), 4)  # Green color for corners

    # Prepare keypoints from the reference and destination points
    keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in ref_points]
    keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in dst_points]

    # Prepare matches using the mask to filter inliers
    matches = [cv2.DMatch(i, i, 0) for i in range(len(mask)) if mask[i]]

    # Use OpenCV's drawMatches function to visualize matches
    img_matches = cv2.drawMatches(
        img1, keypoints1, 
        img2_with_corners, keypoints2, 
        matches, None,
        matchColor=(0, 255, 0),  # Green color for matches
        singlePointColor=(255, 0, 0),  # Red color for unmatched keypoints
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return img_matches

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
    fig, axes = plt.subplots(1, 2, figsize=(15, 12))
    fig.suptitle('XFeat Feature Detection Results', fontsize=16)
    
    # Plot 1: Original image
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot 2: Keypoints visualization
    axes[1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[1].scatter(kpts[:, 0], kpts[:, 1], c='red', s=10, alpha=0.7)
    axes[1].set_title(f'Detected Keypoints ({len(kpts)} points)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {output_path}")

def validate_onnx_detector(onnx_path, test_image_path, save_visualization=True):
    """Validate ONNX detector model by running inference on a test image."""
    print(f"Validating ONNX detector model with test image: {test_image_path}")
    
    # Load and prepare test image
    test_image = Image.open(test_image_path)
    test_input = prepare_detect_input(test_image)
    test_input_np = test_input.cpu().detach().numpy()
    
    session = setup_onnx_session(onnx_path, "detector")
    inputs = {'img': test_input_np}
    
    # Measure inference time
    start_time = time.time()
    outputs = session.run(None, inputs)
    end_time = time.time()
    
    inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
    print(f"Inference time: {inference_time:.2f} ms")
    
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
    
    print("\nDetector validation successful!")
    return outputs

def validate_onnx_matcher(detector_onnx_path, matcher_onnx_path, test_image1_path, test_image2_path, save_visualization=True):
    """Validate ONNX matcher model by running inference on two test images."""
    print(f"Validating ONNX matcher model with test images: {test_image1_path}, {test_image2_path}")
    
    # Load detector session
    detector_session = setup_onnx_session(detector_onnx_path, "detector")
    
    # Load and prepare test images
    test_image1 = Image.open(test_image1_path)
    test_image2 = Image.open(test_image2_path)
    
    test_input1 = prepare_detect_input(test_image1)
    test_input2 = prepare_detect_input(test_image2)
    
    test_input1_np = test_input1.cpu().detach().numpy()
    test_input2_np = test_input2.cpu().detach().numpy()
    
    # Run detection on both images with timing
    print("Running detection on image 1...")
    inputs1 = {'img': test_input1_np}
    # Warm up
    _ = detector_session.run(None, inputs1)
    # Measure
    start_time = time.time()
    output1 = detector_session.run(None, inputs1)
    end_time = time.time()
    detection_time1 = (end_time - start_time) * 1000
    print(f"Detection time for image 1: {detection_time1:.2f} ms")
    
    print("Running detection on image 2...")
    inputs2 = {'img': test_input2_np}
    # Warm up
    _ = detector_session.run(None, inputs2)
    # Measure
    start_time = time.time()
    output2 = detector_session.run(None, inputs2)
    end_time = time.time()
    detection_time2 = (end_time - start_time) * 1000
    print(f"Detection time for image 2: {detection_time2:.2f} ms")
    
    print(f"Image 1 - Keypoints: {output1[0].shape}, Descriptors: {output1[1].shape}, Scales: {output1[2].shape}")
    print(f"Image 2 - Keypoints: {output2[0].shape}, Descriptors: {output2[1].shape}, Scales: {output2[2].shape}")
    
    # Prepare matcher inputs
    dec1_ids_np, dec1_kps_np, dec1_desc_np, dec1_sc_np, dec2_ids_np, dec2_kps_np, dec2_desc_np = prepare_match_inputs(output1, output2)
    
    # Load matcher session
    matcher_session = setup_onnx_session(matcher_onnx_path, "matcher")
    
    # Run matching with timing
    print("Running matching...")
    match_inputs = {
        'dec1_ids': dec1_ids_np,
        'dec1_kps': dec1_kps_np,
        'dec1_desc': dec1_desc_np,
        'dec1_sc': dec1_sc_np,
        'dec2_ids': dec2_ids_np,
        'dec2_kps': dec2_kps_np,
        'dec2_desc': dec2_desc_np,
    }
    # Warm up
    _ = matcher_session.run(None, match_inputs)
    # Measure
    start_time = time.time()
    match_outputs = matcher_session.run(None, match_inputs)
    end_time = time.time()
    matching_time = (end_time - start_time) * 1000
    print(f"Matching time: {matching_time:.2f} ms")
    
    total_time = detection_time1 + detection_time2 + matching_time
    print(f"Total pipeline time: {total_time:.2f} ms")
    
    print("\nMatcher output shapes:")
    for i, output in enumerate(match_outputs):
        print(f"Output {i}: {output.shape}")
    
    # Extract match results
    ref_idx = match_outputs[0]
    ref_points = match_outputs[1]
    target_idx = match_outputs[2]
    target_points = match_outputs[3]
    
    print(f"\nMatching results:")
    print(f"Number of matches: {len(ref_points)}")
    print(f"Reference points: {ref_points.shape}")
    print(f"Target points: {target_points.shape}")
    
    if save_visualization and len(ref_points) > 0:
        # Create output filename
        base_name1 = os.path.splitext(os.path.basename(test_image1_path))[0]
        base_name2 = os.path.splitext(os.path.basename(test_image2_path))[0]
        model_name = os.path.splitext(os.path.basename(matcher_onnx_path))[0]
        output_path = f"matches_visualization_{model_name}_{base_name1}_{base_name2}.png"
        
        # Convert PIL images to OpenCV format
        img1_cv = cv2.cvtColor(np.array(test_image1), cv2.COLOR_RGB2BGR)
        img2_cv = cv2.cvtColor(np.array(test_image2), cv2.COLOR_RGB2BGR)
        
        # Create visualization
        canvas = warp_corners_and_draw_matches(ref_points, target_points, img1_cv, img2_cv)
        
        # Save visualization
        cv2.imwrite(output_path, canvas)
        print(f"Match visualization saved to: {output_path}")
    
    print("\nMatcher validation successful!")

def export_detector(xfeat_model, output_path, device):
    """Export the detector model to ONNX."""
    print(f"\nExporting detector model to {output_path}...")
    
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

    # Create dummy input for export
    dummy_input = torch.randn(1, 1, 480, 640, device=device)  # Example size, will be dynamic

    # Export to ONNX
    torch.onnx.export(
        detect_xfeat_wrapper,
        dummy_input,
        output_path,
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
    print("Detector export completed successfully!")

def export_matcher(xfeat_model, output_path, device):
    """Export the matcher model to ONNX."""
    print(f"\nExporting matcher model to {output_path}...")
    
    # Create wrapper for matching
    class MatchXFeatWrapper(torch.nn.Module):
        def __init__(self, model):
            super(MatchXFeatWrapper, self).__init__()
            self.model = model

        @torch.inference_mode()
        def forward(self, dec1_ids, dec1_kps, dec1_desc, dec1_scales, dec2_ids, dec2_kps, dec2_desc):
            return self.model.match_xfeat_star_onnx(
                dec1_ids, dec1_kps, dec1_desc, dec1_scales, 
                dec2_ids, dec2_kps, dec2_desc)

    match_xfeat_wrapper = MatchXFeatWrapper(xfeat_model)
    match_xfeat_wrapper = match_xfeat_wrapper.to(device)

    # Create dummy inputs for export (example sizes)
    batch_size = 1
    num_kpts1 = 100
    num_kpts2 = 120
    desc_dim = 64
    
    dec1_ids = torch.randn(batch_size, num_kpts1, 1, device=device)
    dec1_kps = torch.randn(batch_size, num_kpts1, 2, device=device)
    dec1_desc = torch.randn(batch_size, num_kpts1, desc_dim, device=device)
    dec1_sc = torch.randn(batch_size, num_kpts1, 1, device=device)
    
    dec2_ids = torch.randn(batch_size, num_kpts2, 1, device=device)
    dec2_kps = torch.randn(batch_size, num_kpts2, 2, device=device)
    dec2_desc = torch.randn(batch_size, num_kpts2, desc_dim, device=device)

    # Export to ONNX
    torch.onnx.export(
        match_xfeat_wrapper,
        (dec1_ids, dec1_kps, dec1_desc, dec1_sc, dec2_ids, dec2_kps, dec2_desc),
        output_path,
        export_params=True,
        opset_version=18,
        training=torch.onnx.TrainingMode.EVAL,
        do_constant_folding=True,
        input_names=['dec1_ids', 'dec1_kps', 'dec1_desc', 'dec1_sc',
                     'dec2_ids', 'dec2_kps', 'dec2_desc'],
        output_names=['ref_idx', 'ref_pnts', 'target_idx', 'target_pnts'],
        dynamic_axes={
            'dec1_ids': {1: 'kps1'},
            'dec1_kps': {1: 'kps1'},
            'dec1_desc': {1: 'kps1'},
            'dec1_sc': {1: 'kps1'},
            'dec2_ids': {1: 'kps2'},
            'dec2_kps': {1: 'kps2'},
            'dec2_desc': {1: 'kps2'},
            'ref_idx': {0: 'matches'},
            'ref_pnts': {0: 'matches'},
            'target_idx': {0: 'matches'},
            'target_pnts': {0: 'matches'},
        }
    )
    print("Matcher export completed successfully!")

def main():
    parser = argparse.ArgumentParser(description='Export XFeat model to ONNX format')
    parser.add_argument('--model', type=str, choices=['detect', 'match', 'both'], default='detect',
                      help='Which model to export: detect, match, or both')
    parser.add_argument('--output-detect', type=str, default='xfeat_star_detect.onnx',
                      help='Output path for detector ONNX model')
    parser.add_argument('--output-match', type=str, default='xfeat_star_match.onnx',
                      help='Output path for matcher ONNX model')
    parser.add_argument('--validate', action='store_true',
                      help='Validate ONNX model with test images')
    parser.add_argument('--test-image', type=str, default='../assets/test_image.jpg',
                      help='Path to test image for detector validation')
    parser.add_argument('--test-image2', type=str, default='../assets/test_image2.jpg',
                      help='Path to second test image for matcher validation')
    parser.add_argument('--no-visualization', action='store_true',
                      help='Disable feature visualization saving')
    args = parser.parse_args()

    # Handle visualization flag
    save_viz = not args.no_visualization

    # Initialize XFeat model
    print("Initializing XFeat model...")
    device = torch.device('cpu')
    xfeat_model = XFeat().to(device)
    xfeat_model.eval()

    # Export models based on selection
    if args.model in ['detect', 'both']:
        export_detector(xfeat_model, args.output_detect, device)
    
    if args.model in ['match', 'both']:
        export_matcher(xfeat_model, args.output_match, device)

    # Validate if requested
    if args.validate:
        if args.model in ['detect', 'both']:
            validate_onnx_detector(args.output_detect, args.test_image, save_visualization=save_viz)
        
        if args.model in ['match', 'both']:
            # For matcher validation, we need both detector and matcher models
            detector_path = args.output_detect
            if args.model == 'match' and not os.path.exists(detector_path):
                print(f"Warning: Detector model {detector_path} not found. Exporting detector first for matcher validation...")
                export_detector(xfeat_model, detector_path, device)
            
            validate_onnx_matcher(detector_path, args.output_match, args.test_image, args.test_image2, save_visualization=save_viz)

if __name__ == '__main__':
    main() 