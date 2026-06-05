import os

import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry
from skimage.filters import frangi

from image_analysis.config import GROUND_TRUTH_DIR

drawing = False
mode = True
ix, iy = -1, -1
brush_size = 3


def mouse_callback_paint(event, x, y, flags, param):
    global ix, iy, drawing, mode, brush_size
    mask_canvas = param

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        mode = True
    elif event == cv2.EVENT_RBUTTONDOWN:
        drawing = True
        mode = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            color = 255 if mode else 0
            cv2.circle(mask_canvas, (x, y), brush_size, color, -1)
    elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
        drawing = False


def automated_generation_with_manual_edit(
    image_path, sam_checkpoint_path, output_dir=GROUND_TRUTH_DIR
):

    os.makedirs(output_dir, exist_ok=True)

    print("Processing raw image with automated AI pipeline...")
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ridge_map = frangi(gray, sigmas=np.arange(1, 3, 1))
    root_pixels = np.argwhere(ridge_map > 0.1)

    if len(root_pixels) > 0:
        sampled_coords = root_pixels[::20]
        input_points = np.flip(sampled_coords, axis=1)
        input_labels = np.ones(len(input_points))

        sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint_path)
        sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
        predictor = SamPredictor(sam)
        predictor.set_image(image_rgb)

        masks, _, _ = predictor.predict(
            input_points, input_labels, multimask_output=False
        )
        auto_mask = (masks[0] * 255).astype(np.uint8)
    else:
        auto_mask = np.zeros(gray.shape, dtype=np.uint8)

    print("\n🎨 Opening Editor Window...")
    print("👉 INSTRUCTIONS:")
    print(" - HOLD LEFT MOUSE BUTTON to paint/add a missing thin root.")
    print(" - HOLD RIGHT MOUSE BUTTON to erase background noise or artifacts.")
    print(" - Press 'S' on your keyboard to SAVE and finish.")
    print(" - Press 'ESC' to discard changes and exit.")

    cv2.namedWindow("Root Mask Editor", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Root Mask Editor", mouse_callback_paint, param=auto_mask)

    while True:
        display_img = image.copy()
        green_overlay = np.zeros_like(image)
        green_overlay[:, :, 1] = auto_mask  # Put mask in green channel

        blended = cv2.addWeighted(display_img, 0.6, green_overlay, 0.4, 0)

        cv2.imshow("Root Mask Editor", blended)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            final_binary_mask = (auto_mask > 127).astype(np.uint8)
            filename = os.path.basename(image_path)
            output_path = os.path.join(output_dir, filename)

            # Save final target as strict 0 and 1
            cv2.imwrite(output_path, final_binary_mask * 255)
            print(f"Ground truth saved to: {output_path}")
            break
        elif key == 27:
            print("Editing cancelled. No files saved.")
            break

    cv2.destroyAllWindows()
