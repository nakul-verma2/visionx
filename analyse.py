import os
from ultralytics import YOLO
from glob import glob
import cv2

# Load the model
model = YOLO("runs/train/hardcase_exp/weights/best.pt")  # Adjust path if needed

# Path to your test images folder
test_images_folder = "hard_cases/images"  # Adjust this path if different

# Get list of image file paths (jpg, png, jpeg)
image_extensions = ('*.jpg', '*.png', '*.jpeg')
image_paths = []

for ext in image_extensions:
    image_paths.extend(glob(os.path.join(test_images_folder, ext)))

# Sort and take the first 5 images
image_paths = sorted(image_paths)[:5]

# Run predictions and show outputs
results = model.predict(source=image_paths, save=True, stream=True)

for result in results:
    # Convert to numpy array and draw boxes
    img = result.plot()
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Resize image to fit screen (e.g., 800x600 or smaller)
    max_width = 800
    max_height = 600
    h, w = img_bgr.shape[:2]

    if w > max_width or h > max_height:
        scaling_factor = min(max_width / w, max_height / h)
        new_size = (int(w * scaling_factor), int(h * scaling_factor))
        img_bgr = cv2.resize(img_bgr, new_size, interpolation=cv2.INTER_AREA)

    # Show image
    cv2.imshow("Detection", img_bgr)
    key = cv2.waitKey(0)  # Wait for key press
    if key == 27:  # ESC to exit early
        break

cv2.destroyAllWindows()
