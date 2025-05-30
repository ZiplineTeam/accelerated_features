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
import json

# fmt: off
sys.path.append(os.path.abspath('.'))
from modules.xfeat_onnx_compatible import XFeat
# fmt: on


def setup_onnx_session(onnx_path, session_name="ONNX", enable_profiling=False):
    """Set up ONNX runtime session with CUDA fallback and optional profiling."""
    # Try to create ONNX runtime session with CUDA first
    providers = []
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        providers.append('CUDAExecutionProvider')
        print(f"Using CUDA for {session_name} inference")
    else:
        print(
            f"Warning: CUDA not available for {session_name} inference, falling back to CPU")

    providers.append('CPUExecutionProvider')

    # Setup session options for profiling if enabled
    session_options = ort.SessionOptions()
    if enable_profiling:
        session_options.enable_profiling = True
        session_options.profile_file_prefix = f"{session_name.lower().replace(' ', '_')}_profile"
        print(f"Profiling enabled for {session_name}")

    try:
        session = ort.InferenceSession(
            onnx_path, providers=providers, sess_options=session_options)
        print(f"{session_name} using provider: {session.get_providers()[0]}")
        return session
    except Exception as e:
        print(
            f"Warning: Failed to create {session_name} session with preferred providers, using default: {e}")
        return ort.InferenceSession(onnx_path, sess_options=session_options)


def analyze_profile_json(profile_path):
    """Analyze the ONNX Runtime profiling JSON output."""
    if not os.path.exists(profile_path):
        print(f"Profile file not found: {profile_path}")
        return

    print(f"\n=== PROFILING ANALYSIS: {profile_path} ===")

    try:
        with open(profile_path, 'r') as f:
            profile_data = json.load(f)

        print(f"Profile data type: {type(profile_data)}")

        # Handle different profile formats
        if isinstance(profile_data, list):
            # Profile data is directly a list of events
            events = profile_data
            print(f"Profile data is a list with {len(events)} events")
        elif isinstance(profile_data, dict):
            # Profile data is a dictionary containing traceEvents
            events = profile_data.get('traceEvents', [])
            print(f"Profile data keys: {list(profile_data.keys())}")
            print(f"Found {len(events)} events in traceEvents")
        else:
            print(f"Unexpected profile data format: {type(profile_data)}")
            return

        # Sample first few events to understand structure
        print(f"\nFirst 3 events structure:")
        for i, event in enumerate(events[:3]):
            if isinstance(event, dict):
                print(
                    f"Event {i}: {event.get('name', 'No name')}, cat={event.get('cat', 'No cat')}, dur={event.get('dur', 'No dur')}")
            else:
                print(f"Event {i}: type={type(event)}, content={event}")

        # Filter for kernel execution events with proper type checking
        kernel_events = []
        op_events = []
        memory_events = []

        for event in events:
            if not isinstance(event, dict):
                continue  # Skip non-dictionary events

            # Check for kernel events
            if event.get('cat') == 'Kernel' and 'dur' in event:
                kernel_events.append(event)
            # Check for operation events
            elif event.get('cat') == 'Node' and 'dur' in event:
                op_events.append(event)
            # Check for memory operations
            elif ('memory' in event.get('name', '').lower() or
                  'copy' in event.get('name', '').lower()) and 'dur' in event:
                memory_events.append(event)

        print(
            f"Found {len(kernel_events)} kernel events, {len(op_events)} operation events, {len(memory_events)} memory events")

        if kernel_events:
            print(f"\n--- KERNEL EXECUTION TIMES ---")
            kernel_times = {}
            for event in kernel_events:
                name = event.get('name', 'Unknown')
                duration = event.get('dur', 0) / 1000.0  # Convert to ms
                if name in kernel_times:
                    kernel_times[name] += duration
                else:
                    kernel_times[name] = duration

            # Sort by duration (descending)
            sorted_kernels = sorted(
                kernel_times.items(), key=lambda x: x[1], reverse=True)
            total_kernel_time = sum(kernel_times.values())

            print(f"Total kernel execution time: {total_kernel_time:.2f} ms")
            print("\nTop 10 slowest kernels:")
            for i, (name, duration) in enumerate(sorted_kernels[:10]):
                percentage = (duration / total_kernel_time) * \
                    100 if total_kernel_time > 0 else 0
                print(
                    f"{i+1:2d}. {name:<50} {duration:8.2f} ms ({percentage:5.1f}%)")

        if op_events:
            print(f"\n--- OPERATION EXECUTION TIMES ---")
            op_times = {}
            for event in op_events:
                name = event.get('name', 'Unknown')
                duration = event.get('dur', 0) / 1000.0  # Convert to ms
                if name in op_times:
                    op_times[name] += duration
                else:
                    op_times[name] = duration

            # Sort by duration (descending)
            sorted_ops = sorted(
                op_times.items(), key=lambda x: x[1], reverse=True)
            total_op_time = sum(op_times.values())

            print(f"Total operation execution time: {total_op_time:.2f} ms")
            print("\nTop 10 slowest operations:")
            for i, (name, duration) in enumerate(sorted_ops[:10]):
                percentage = (duration / total_op_time) * \
                    100 if total_op_time > 0 else 0
                print(
                    f"{i+1:2d}. {name:<50} {duration:8.2f} ms ({percentage:5.1f}%)")

        if memory_events:
            print(f"\n--- MEMORY OPERATIONS ---")
            memory_time = 0
            for event in memory_events:
                duration = event.get('dur', 0) / 1000.0
                memory_time += duration
                print(f"{event.get('name', 'Unknown'):<50} {duration:8.2f} ms")
            print(f"Total memory operation time: {memory_time:.2f} ms")

        # Additional analysis - look for all categories
        categories = {}
        for event in events:
            if isinstance(event, dict) and 'cat' in event:
                cat = event.get('cat', 'Unknown')
                categories[cat] = categories.get(cat, 0) + 1

        if categories:
            print(f"\n--- EVENT CATEGORIES ---")
            for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                print(f"{cat:<20} {count:>6} events")

    except Exception as e:
        print(f"Error analyzing profile: {e}")
        import traceback
        traceback.print_exc()


def prepare_detect_input(im):
    """Convert image to grayscale tensor for detection with proper preprocessing."""
    im_np = np.array(im, dtype=np.float32)

    # Convert RGB to Grayscale using luminosity formula
    im_gray = 0.299 * im_np[:, :, 0] + 0.587 * \
        im_np[:, :, 1] + 0.114 * im_np[:, :, 2]

    # Convert to PyTorch tensor with batch and channel dimensions
    im_tensor = torch.from_numpy(im_gray).unsqueeze(0).unsqueeze(0)

    return im_tensor


def prepare_match_inputs(output1, output2):
    """Prepare inputs for matcher from detector outputs."""
    # Create IDs for keypoints
    dec1_ids_np = np.arange(
        1, output1[0].shape[1] + 1).astype(np.float32).reshape(1, -1)
    dec1_ids_np = np.expand_dims(dec1_ids_np, axis=-1)

    dec1_kps_np = output1[0].astype(np.float32)
    dec1_desc_np = output1[1].astype(np.float32)
    dec1_sc_np = output1[2].astype(np.float32)
    dec1_sc_np = np.expand_dims(dec1_sc_np, axis=-1)

    dec2_ids_np = np.arange(
        1, output2[0].shape[1] + 1).astype(np.float32).reshape(1, -1)
    dec2_ids_np = np.expand_dims(dec2_ids_np, axis=-1)

    dec2_kps_np = output2[0].astype(np.float32)
    dec2_desc_np = output2[1].astype(np.float32)

    return dec1_ids_np, dec1_kps_np, dec1_desc_np, dec1_sc_np, dec2_ids_np, dec2_kps_np, dec2_desc_np


def warp_corners_and_draw_matches(ref_points, dst_points, img1, img2):
    """Visualize matches between two images"""
    # # Calculate the Homography matrix
    # H, mask = cv2.findHomography(
    #     ref_points, dst_points, cv2.USAC_MAGSAC, 3.5, maxIters=1000, confidence=0.999)
    # mask = mask.flatten() > 0

    # # Get corners of the first image (image1)
    # h, w = img1.shape[:2]
    # corners_img1 = np.array(
    #     [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32).reshape(-1, 1, 2)

    # # Warp corners to the second image (image2) space using the Homography matrix
    # warped_corners = cv2.perspectiveTransform(corners_img1, H)

    # # Draw the warped corners in image2
    # img2_with_corners = img2.copy()
    # for i in range(len(warped_corners)):
    #     start_point = tuple(warped_corners[i - 1][0].astype(int))
    #     end_point = tuple(warped_corners[i][0].astype(int))
    #     cv2.line(img2_with_corners, start_point, end_point,
    #              (0, 255, 0), 4)  # Green color for corners

    # Prepare keypoints from the reference and destination points
    keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in ref_points]
    keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in dst_points]

    # Prepare matches using the mask to filter inliers
    matches = [cv2.DMatch(i, i, 0) for i in range(len(ref_points))]

    # Use OpenCV's drawMatches function to visualize matches
    img_matches = cv2.drawMatches(
        img1, keypoints1,
        img2, keypoints2,
        matches, None,
        matchColor=(0, 255, 0),  # Green color for matches
        singlePointColor=(0, 0, 255),  # Red color for unmatched keypoints
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

    # ensure all kps are unique
    kpts = np.unique(kpts, axis=0)

    # Create figure with subplots
    fig, axes = plt.subplots(1, figsize=(15, 12))
    fig.suptitle('XFeat Feature Detection Results', fontsize=16)

    axes.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes.scatter(kpts[:, 0], kpts[:, 1], c='red', s=10, alpha=0.7)
    axes.set_title(f'Detected Keypoints ({len(kpts)} points)')
    axes.axis('off')

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

        visualize_features(test_image_path, keypoints,
                           descriptors, scales, output_path)

    print("\nDetector validation successful!")
    return outputs


def validate_onnx_detector_sparse(onnx_path, test_image_path, save_visualization=True):
    """Validate ONNX sparse detector model with detailed profiling."""
    print(
        f"Validating ONNX sparse detector model with profiling: {test_image_path}")

    # Load and prepare test image
    test_image = Image.open(test_image_path)
    test_input = prepare_detect_input(test_image)
    test_input_np = test_input.cpu().detach().numpy()

    print(f"Input image size: {test_image.size}")
    print(f"Input tensor shape: {test_input_np.shape}")

    # Create session with profiling enabled
    session = setup_onnx_session(
        onnx_path, "sparse detector", enable_profiling=True)
    inputs = {'img': test_input_np}

    # Warm up run (3 times to stabilize performance)
    print("Warming up...")
    for i in range(3):
        _ = session.run(None, inputs)

    # Profiled run
    print("Running profiled inference...")
    start_time = time.time()
    outputs = session.run(None, inputs)
    end_time = time.time()

    inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
    print(f"Profiled inference time: {inference_time:.2f} ms")

    # End profiling and get profile file
    profile_file = session.end_profiling()
    print(f"Profile saved to: {profile_file}")

    # Analyze the profile
    analyze_profile_json(profile_file)

    # Print output shapes to verify model is working
    print("\nOutput shapes:")
    for i, output in enumerate(outputs):
        print(f"Output {i}: {output.shape}")

    # Extract outputs
    keypoints = outputs[0]  # Shape: [batch, num_keypoints, 2]
    descriptors = outputs[1]  # Shape: [batch, num_keypoints, descriptor_dim]
    scores = outputs[2]  # Shape: [batch, num_keypoints]

    print(f"\nSparse feature extraction results:")
    print(f"Keypoints: {keypoints.shape}")
    print(f"Descriptors: {descriptors.shape}")
    print(f"Scores: {scores.shape}")
    print(f"Scores min: {scores.min()}, max: {scores.max()}")

    # Print stats about keypoint range
    print(
        f"Keypoint range: [{keypoints[:,:,0].min()}, {keypoints[:,:,0].max()}] x [{keypoints[:,:,1].min()}, {keypoints[:,:,1].max()}]")

    # Count valid keypoints (scores > 0)
    valid_mask = scores > 0
    num_valid = valid_mask.sum().item()
    print(f"Number of valid keypoints: {num_valid}")

    # Run multiple timing tests for statistical analysis
    print("\n=== PERFORMANCE ANALYSIS ===")
    timing_runs = 10
    times = []

    print(f"Running {timing_runs} timing iterations...")
    for i in range(timing_runs):
        start_time = time.time()
        _ = session.run(None, inputs)
        end_time = time.time()
        iteration_time = (end_time - start_time) * 1000
        times.append(iteration_time)
        print(f"Run {i+1:2d}: {iteration_time:6.2f} ms")

    # Calculate statistics
    min_time = min(times)
    max_time = max(times)
    avg_time = sum(times) / len(times)
    std_dev = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5

    print(f"\nTiming Statistics ({timing_runs} runs):")
    print(f"  Average: {avg_time:.2f} ms")
    print(f"  Min:     {min_time:.2f} ms")
    print(f"  Max:     {max_time:.2f} ms")
    print(f"  Std Dev: {std_dev:.2f} ms")

    if save_visualization:
        # Create output filename based on input image and model
        base_name = os.path.splitext(os.path.basename(test_image_path))[0]
        model_name = os.path.splitext(os.path.basename(onnx_path))[0]
        output_path = f"features_visualization_{model_name}_{base_name}.png"

        # For visualization, use scores as scales (they serve similar purpose for visualization)
        visualize_features(test_image_path, keypoints,
                           descriptors, scores, output_path)

    print("\nSparse detector profiled validation successful!")
    return outputs


def validate_onnx_matcher(detector_onnx_path, matcher_onnx_path, test_image1_path, test_image2_path, save_visualization=True):
    """Validate ONNX matcher model by running inference on two test images."""
    print(
        f"Validating ONNX matcher model with test images: {test_image1_path}, {test_image2_path}")

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

    print(
        f"Image 1 - Keypoints: {output1[0].shape}, Descriptors: {output1[1].shape}, Scales: {output1[2].shape}")
    print(
        f"Image 2 - Keypoints: {output2[0].shape}, Descriptors: {output2[1].shape}, Scales: {output2[2].shape}")

    # Prepare matcher inputs
    dec1_ids_np, dec1_kps_np, dec1_desc_np, dec1_sc_np, dec2_ids_np, dec2_kps_np, dec2_desc_np = prepare_match_inputs(
        output1, output2)

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

    # We want some stats about how keypoints moved due to refinement
    # We can do this by comparing the original keypoints with the refined ones
    for i in range(len(ref_points)):
        print(f"Refined reference keypoint {i}: {ref_points[i]}")
        idx = int(ref_idx[i][0])
        print(idx)
        original = dec1_kps_np[0, idx]
        print(f"Original target keypoint {idx}: {original}")

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
        canvas = warp_corners_and_draw_matches(
            ref_points, target_points, img1_cv, img2_cv)

        # Save visualization
        cv2.imwrite(output_path, canvas)
        print(f"Match visualization saved to: {output_path}")
        print("Legend: Green lines = refined matches, Blue circles = original positions, Cyan lines = refinement displacement")

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
            return self.model.xfeat_detect(img, top_k=6096)

    detect_xfeat_wrapper = DetectXFeatWrapper(xfeat_model)
    detect_xfeat_wrapper = detect_xfeat_wrapper.to(device)

    # Create dummy input for export
    # Example size, will be dynamic
    dummy_input = torch.randn(1, 1, 480, 640, device=device)

    # Export to ONNX
    torch.onnx.export(
        detect_xfeat_wrapper,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,
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


def export_detector_sparse(xfeat_model, output_path, device):
    """Export the sparse detector model to ONNX."""
    print(f"\nExporting sparse detector model to {output_path}...")

    # Create ONNX-compatible model for sparse detection
    from modules.xfeat_onnx_compatible import XFeat as XFeatONNX
    xfeat_onnx_model = XFeatONNX().to(device)

    # Load the same weights
    if hasattr(xfeat_model, 'net'):
        xfeat_onnx_model.net.load_state_dict(xfeat_model.net.state_dict())

    # Create wrapper for sparse detection
    class DetectXFeatSparseWrapper(torch.nn.Module):
        def __init__(self, model):
            super(DetectXFeatSparseWrapper, self).__init__()
            self.model = model

        @torch.inference_mode()
        def forward(self, img):
            # Ensure input is on the correct device
            model_device = next(self.model.parameters()).device
            if img.device != model_device:
                img = img.to(model_device)
            # xfeat_detect_sparse returns a tuple (keypoints, descriptors, scores)
            return self.model.xfeat_detect_sparse(img, top_k=500, detection_threshold=0.25, kernel_size=17)

    detect_xfeat_sparse_wrapper = DetectXFeatSparseWrapper(xfeat_onnx_model)
    detect_xfeat_sparse_wrapper = detect_xfeat_sparse_wrapper.to(device)

    # Create dummy input for export
    # Example size, will be dynamic
    dummy_input = torch.randn(1, 1, 480, 640, device=device)

    # Export to ONNX
    torch.onnx.export(
        detect_xfeat_sparse_wrapper,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,
        training=torch.onnx.TrainingMode.EVAL,
        do_constant_folding=True,
        input_names=['img'],
        output_names=['keypoints', 'descriptors', 'scores'],
        dynamic_axes={
            'img': {2: 'height', 3: 'width'},
            'keypoints': {1: 'num_keypoints'},
            'descriptors': {1: 'num_keypoints'},
            'scores': {1: 'num_keypoints'}
        }
    )
    print("Sparse detector export completed successfully!")


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
            return self.model.match_xfeat_star(
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


def export_detect_and_match_dense(xfeat_model, output_path, device):
    """Export the combined detect_and_match_dense model to ONNX."""
    print(f"\nExporting combined detect-and-match model to {output_path}...")

    # Create wrapper for combined detection and matching
    class DetectAndMatchWrapper(torch.nn.Module):
        def __init__(self, model):
            super(DetectAndMatchWrapper, self).__init__()
            self.model = model

        @torch.inference_mode()
        def forward(self, img1, img2):
            return self.model.detect_and_match_dense(img1, img2, top_k=2000, min_cossim=0.82)

    detect_match_wrapper = DetectAndMatchWrapper(xfeat_model)
    detect_match_wrapper = detect_match_wrapper.to(device)

    # Create dummy inputs for export (grayscale images)
    dummy_img1 = torch.randn(1, 1, 480, 640, device=device)
    dummy_img2 = torch.randn(1, 1, 480, 640, device=device)

    # Export to ONNX
    torch.onnx.export(
        detect_match_wrapper,
        (dummy_img1, dummy_img2),
        output_path,
        export_params=True,
        opset_version=18,
        training=torch.onnx.TrainingMode.EVAL,
        do_constant_folding=True,
        input_names=['img1', 'img2'],
        output_names=['keypoints1', 'keypoints2',
                      'descriptors1', 'descriptors2', 'target_indices'],
        dynamic_axes={
            'img1': {2: 'height1', 3: 'width1'},
            'img2': {2: 'height2', 3: 'width2'},
            'keypoints1': {0: 'num_matches'},
            'keypoints2': {0: 'num_matches'},
            'descriptors1': {0: 'num_matches'},
            'descriptors2': {0: 'num_matches'},
            'target_indices': {0: 'num_matches'}
        }
    )
    print("Combined detect-and-match export completed successfully!")


def validate_onnx_detect_and_match(onnx_path, test_image1_path, test_image2_path, save_visualization=True):
    """Validate ONNX detect_and_match model by running inference on two test images."""
    print(
        f"Validating ONNX detect-and-match model with test images: {test_image1_path}, {test_image2_path}")

    # Load and prepare test images
    test_image1 = Image.open(test_image1_path)
    test_image2 = Image.open(test_image2_path)

    test_input1 = prepare_detect_input(test_image1)
    test_input2 = prepare_detect_input(test_image2)

    test_input1_np = test_input1.cpu().detach().numpy()
    test_input2_np = test_input2.cpu().detach().numpy()

    # Load ONNX session
    session = setup_onnx_session(onnx_path, "detect-and-match")

    # Run inference with timing
    print("Running combined detect-and-match inference...")
    inputs = {'img1': test_input1_np, 'img2': test_input2_np}

    # Warm up
    _ = session.run(None, inputs)

    # Measure
    start_time = time.time()
    outputs = session.run(None, inputs)
    end_time = time.time()
    inference_time = (end_time - start_time) * 1000
    print(f"Combined inference time: {inference_time:.2f} ms")

    print("\nOutput shapes:")
    for i, output in enumerate(outputs):
        print(f"Output {i}: {output.shape}")

    # Extract outputs
    keypoints1 = outputs[0]
    keypoints2 = outputs[1]
    descriptors1 = outputs[2]
    descriptors2 = outputs[3]
    target_indices = outputs[4]

    print(f"\nDetect-and-match results:")
    print(f"Matched keypoints 1: {keypoints1.shape}")
    print(f"Matched keypoints 2: {keypoints2.shape}")
    print(f"Descriptors 1: {descriptors1.shape}")
    print(f"Descriptors 2: {descriptors2.shape}")
    print(f"Target indices: {target_indices.shape}")
    print(f"Number of matches: {len(keypoints1)}")
    print(f"Number of valid correspondences: {(target_indices >= 0).sum()}")

    if save_visualization and len(keypoints1) > 0:
        # Create output filename
        base_name1 = os.path.splitext(os.path.basename(test_image1_path))[0]
        base_name2 = os.path.splitext(os.path.basename(test_image2_path))[0]
        model_name = os.path.splitext(os.path.basename(onnx_path))[0]
        output_path = f"combined_matches_visualization_{model_name}_{base_name1}_{base_name2}.png"

        # Convert PIL images to OpenCV format
        img1_cv = cv2.cvtColor(np.array(test_image1), cv2.COLOR_RGB2BGR)
        img2_cv = cv2.cvtColor(np.array(test_image2), cv2.COLOR_RGB2BGR)

        # Create visualization
        canvas = warp_corners_and_draw_matches(
            keypoints1, keypoints2, img1_cv, img2_cv)

        # Save visualization
        cv2.imwrite(output_path, canvas)
        print(f"Combined match visualization saved to: {output_path}")

    print("\nCombined detect-and-match validation successful!")
    return outputs


def main():
    parser = argparse.ArgumentParser(
        description='Export XFeat model to ONNX format')
    parser.add_argument('--model', type=str, choices=['detect', 'detect-sparse', 'match', 'detect-and-match', 'all'], default='detect',
                        help='Which model to export: detect (dense), detect-sparse, match, detect-and-match (combined), or all')
    parser.add_argument('--output-detect', type=str, default='xfeat_star_detect.onnx',
                        help='Output path for dense detector ONNX model')
    parser.add_argument('--output-detect-sparse', type=str, default='xfeat_sparse_detect.onnx',
                        help='Output path for sparse detector ONNX model')
    parser.add_argument('--output-match', type=str, default='xfeat_star_match.onnx',
                        help='Output path for matcher ONNX model')
    parser.add_argument('--output-detect-and-match', type=str, default='xfeat_detect_and_match.onnx',
                        help='Output path for combined detect-and-match ONNX model')
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
    if args.model in ['detect', 'all']:
        export_detector(xfeat_model, args.output_detect, device)

    if args.model in ['detect-sparse', 'all']:
        export_detector_sparse(xfeat_model, args.output_detect_sparse, device)

    if args.model in ['match', 'all']:
        export_matcher(xfeat_model, args.output_match, device)

    if args.model in ['detect-and-match', 'all']:
        export_detect_and_match_dense(
            xfeat_model, args.output_detect_and_match, device)

    # Validate if requested
    if args.validate:
        if args.model in ['detect', 'all']:
            validate_onnx_detector(
                args.output_detect, args.test_image, save_visualization=save_viz)

        if args.model in ['detect-sparse', 'all']:
            validate_onnx_detector_sparse(
                args.output_detect_sparse, args.test_image, save_visualization=save_viz)

        if args.model in ['match', 'all']:
            # For matcher validation, we need both detector and matcher models
            detector_path = args.output_detect
            if args.model == 'match' and not os.path.exists(detector_path):
                print(
                    f"Warning: Detector model {detector_path} not found. Skipping matcher validation.")
                return
            validate_onnx_matcher(detector_path, args.output_match,
                                  args.test_image, args.test_image2, save_visualization=save_viz)

        if args.model in ['detect-and-match', 'all']:
            validate_onnx_detect_and_match(
                args.output_detect_and_match, args.test_image, args.test_image2, save_visualization=save_viz)


if __name__ == '__main__':
    main()
