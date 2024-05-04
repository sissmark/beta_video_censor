#todo
# resolution selection
# cancel current censoring process
# select output file
# copy audio
# implement audio filters (muffling)

import os
import cv2
from nudenet import NudeDetector
from tkinter import filedialog, Tk, Button, Label, Radiobutton, IntVar, ttk
import tempfile
import numpy as np
from PIL import Image, ImageTk

continue_censoring = True

# Function to display the logo
def display_logo():
    logo_path = "logo.png"  # Path to your logo image
    logo_image = Image.open(logo_path)
    # Resize the logo to 640x480 pixels
    logo_image = logo_image.resize((640, 480))
    logo_photo = ImageTk.PhotoImage(logo_image)
    logo_label = Label(root, image=logo_photo, borderwidth=0, padx=0, pady=0)
    logo_label.image = logo_photo
    logo_label.pack(pady=(0,20))

# Function to quit the application
def quit_application():
    root.quit()


# Function to pixelate the specified region of an image and overlay text
def pixelate_region_and_overlay_text(image, x, y, w, h, text, pixel_size=50):
    # Calculate the intersection of the region to pixelate with the image bounds
    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + w, image.shape[1])
    y2 = min(y + h, image.shape[0])

    # Check if there is an intersection
    if x1 < x2 and y1 < y2:
        # Calculate the size of the pixelated region
        pixelated_w = (x2 - x1) // pixel_size
        pixelated_h = (y2 - y1) // pixel_size

        # Ensure that the pixel size is not larger than the region size
        if pixelated_w > 0 and pixelated_h > 0:
            # Extract the region to pixelate
            region = image[y1:y2, x1:x2]

            # Resize the region to a smaller size
            small_region = cv2.resize(region, (pixelated_w, pixelated_h))

            # Resize the small region back to the original size
            pixelated_region = cv2.resize(small_region, ((x2 - x1), (y2 - y1)), interpolation=cv2.INTER_NEAREST)

            # Convert pixelated region to grayscale
            gray_pixelated_region = cv2.cvtColor(pixelated_region, cv2.COLOR_BGR2GRAY)

            # Desaturate the grayscale region
            desaturated_region = cv2.cvtColor(gray_pixelated_region, cv2.COLOR_GRAY2BGR)

            # Replace the region with the desaturated version
            image[y1:y2, x1:x2] = desaturated_region

            # Calculate the position to center the text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2
            thickness = 4
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = x1 + ((x2 - x1) - text_size[0]) // 2
            text_y = y1 + ((y2 - y1) + text_size[1]) // 2

            # Add text overlay
            cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

    return image

# Function to blur the specified region of an image and overlay text
def blur_region_and_overlay_text(image, x, y, w, h, text, blur_amount=100):
    # Ensure that blur_amount is an odd number
    blur_amount = max(1, blur_amount)  # Ensure it's at least 1
    blur_amount = blur_amount + 1 if blur_amount % 2 == 0 else blur_amount

    # Calculate the intersection of the region to blur with the image bounds
    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + w, image.shape[1])
    y2 = min(y + h, image.shape[0])

    # Check if there is an intersection
    if x1 < x2 and y1 < y2:
        # Extract the region to blur
        region = image[y1:y2, x1:x2]

        # Apply Gaussian blur to the region
        blurred_region = cv2.GaussianBlur(region, (blur_amount, blur_amount), 0)

        # Replace the region with the blurred version
        image[y1:y2, x1:x2] = blurred_region

        # Calculate the position to center the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        thickness = 4
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = x1 + ((x2 - x1) - text_size[0]) // 2
        text_y = y1 + ((y2 - y1) + text_size[1]) // 2

        # Add text overlay
        cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

    return image

# Function to invert colors of the specified region of an image and overlay text
def invert_colors_and_overlay_text(image, x, y, w, h, text):
    # Calculate the intersection of the region to invert colors with the image bounds
    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + w, image.shape[1])
    y2 = min(y + h, image.shape[0])

    # Check if there is an intersection
    if x1 < x2 and y1 < y2:
        # Extract the region to invert colors
        region = image[y1:y2, x1:x2]

        # Invert colors in the region
        inverted_region = cv2.bitwise_not(region)

        # Replace the region with the inverted version
        image[y1:y2, x1:x2] = inverted_region

        # Calculate the position to center the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        thickness = 4
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = x1 + ((x2 - x1) - text_size[0]) // 2
        text_y = y1 + ((y2 - y1) + text_size[1]) // 2

        # Add text overlay
        cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

    return image

# Function to apply motion blur to the specified region of an image and overlay text
def motion_blur_and_overlay_text(image, x, y, w, h, text, blur_amount=50):
    # Calculate the intersection of the region to apply motion blur with the image bounds
    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + w, image.shape[1])
    y2 = min(y + h, image.shape[0])

    # Check if there is an intersection
    if x1 < x2 and y1 < y2:
        # Extract the region to apply motion blur
        region = image[y1:y2, x1:x2]

        # Apply motion blur to the region
        kernel_size = max(1, blur_amount)  # Ensure it's at least 1
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel /= kernel_size
        motion_blurred_region = cv2.filter2D(region, -1, kernel)

        # Replace the region with the motion blurred version
        image[y1:y2, x1:x2] = motion_blurred_region

        # Calculate the position to center the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        thickness = 4
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = x1 + ((x2 - x1) - text_size[0]) // 2
        text_y = y1 + ((y2 - y1) + text_size[1]) // 2

        # Add text overlay
        cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

    return image

# Function to apply dithering to the specified region of an image and overlay text
def dither_and_overlay_text(image, x, y, w, h, text):
    # Calculate the intersection of the region to apply dithering with the image bounds
    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + w, image.shape[1])
    y2 = min(y + h, image.shape[0])

    # Check if there is an intersection
    if x1 < x2 and y1 < y2:
        # Extract the region to apply dithering
        region = image[y1:y2, x1:x2]

        # Apply dithering to the region (Floyd-Steinberg dithering algorithm)
        dithered_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        dithered_region = cv2.cvtColor(dithered_region, cv2.COLOR_GRAY2BGR)

        for y in range(h):
            for x in range(w):
                old_pixel = dithered_region[y, x]
                new_pixel = 255 if old_pixel[0] > 127 else 0
                dithered_region[y, x] = [new_pixel, new_pixel, new_pixel]
                quant_error = old_pixel - [new_pixel, new_pixel, new_pixel]

                if x < w - 1:
                    dithered_region[y, x + 1] = (dithered_region[y, x + 1] + quant_error * 7 // 16).clip(0, 255)
                if y < h - 1 and x > 0:
                    dithered_region[y + 1, x - 1] = (dithered_region[y + 1, x - 1] + quant_error * 3 // 16).clip(0, 255)
                if y < h - 1:
                    dithered_region[y + 1, x] = (dithered_region[y + 1, x] + quant_error * 5 // 16).clip(0, 255)
                if y < h - 1 and x < w - 1:
                    dithered_region[y + 1, x + 1] = (dithered_region[y + 1, x + 1] + quant_error * 1 // 16).clip(0, 255)

        # Replace the region with the dithered version
        image[y1:y2, x1:x2] = dithered_region

        # Calculate the position to center the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        thickness = 4
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = x1 + ((x2 - x1) - text_size[0]) // 2
        text_y = y1 + ((y2 - y1) + text_size[1]) // 2

        # Add text overlay
        cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

    return image

# Function to translate class names to more descriptive text
def translate_classname(text):
    if text == "FEMALE_GENITALIA_COVERED":
        retval = "No Pussy"
    elif text == "FACE_FEMALE":
        retval = "No Face"
    elif text == "BUTTOCKS_EXPOSED":
        retval = "No Ass"
    elif text == "FEMALE_BREAST_EXPOSED":
        retval = "No Tits"
    elif text == "FEMALE_GENITALIA_EXPOSED":
        retval = "No Pussy"
    elif text == "MALE_BREAST_EXPOSED":
        retval = "No Tits"
    elif text == "ANUS_EXPOSED":
        retval = "No Anal"
    elif text == "FEET_EXPOSED":
        retval = "No Feet"
    elif text == "BELLY_COVERED":
        retval = "No Belly"
    elif text == "FEET_COVERED":
        retval = "No Feet"
    elif text == "ARMPITS_COVERED":
        retval = "No Pits"
    elif text == "ARMPITS_EXPOSED":
        retval = "No Pits"
    elif text == "FACE_MALE":
        retval = "No Face"
    elif text == "BELLY_EXPOSED":
        retval = "No Belly"
    elif text == "MALE_GENITALIA_EXPOSED":
        retval = "No Dick"
    elif text == "ANUS_COVERED":
        retval = "No Anal"
    elif text == "FEMALE_BREAST_COVERED":
        retval = "No Tits"
    elif text == "BUTTOCKS_COVERED":
        retval = "No Ass"
    else:
        retval = "Censored"
    return retval

# Function to detect and censor explicit frames in a video and save the censored video
def detect_and_censor_video(video_path, output_path, censor_method, resolution=1):
    global continue_censoring
    # Initialize the NudeDetector
    detector = NudeDetector()

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file is opened successfully
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    # Get the frame rate of the original video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Get the video frame properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Get the codec of the original video
    codec = int(cap.get(cv2.CAP_PROP_FOURCC))

    # Calculate the new dimensions for resizing
    new_width = int(width * resolution)
    new_height = int(height * resolution)

    # Create a VideoWriter object to write the censored frames to a new video file
    out = cv2.VideoWriter(output_path, codec, fps, (new_width, new_height))

    # Create a temporary directory to save the video frames
    temp_dir = tempfile.TemporaryDirectory()

    # Create a window to display processed frames
    cv2.namedWindow("Processed Frame", cv2.WINDOW_NORMAL)

    # Initialize progress variables
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress = 0

    # Iterate through each frame
    while continue_censoring:
        # Read the frame
        success, frame = cap.read()

        if not success:
            break

        # Resize the frame to the specified resolution
        frame = cv2.resize(frame, (new_width, new_height))

        # Save the frame as a temporary image file
        temp_image_path = os.path.join(temp_dir.name, "temp_frame.jpg")
        cv2.imwrite(temp_image_path, frame)

        # Detect nudity in the frame
        results = detector.detect(temp_image_path)

        # Check if explicit content is detected
        if results:
            # Apply censorship and text overlay
            for result in results:
                if result['score'] >= 0.0:
                    x, y, w, h = result['box']
                    # Only censor female parts
                    if "MALE_BREAST_EXPOSED" != result['class'] and "MALE_GENITALIA_EXPOSED" != result[
                        'class'] and "FACE_MALE" != result['class']:
                        # Select censoring method
                        if censor_method == 1:
                            # Pixelation method
                            frame = pixelate_region_and_overlay_text(frame, x, y, w, h,
                                                                      translate_classname(result['class']))
                        elif censor_method == 2:
                            # Blurring method
                            frame = blur_region_and_overlay_text(frame, x, y, w, h,
                                                                 translate_classname(result['class']))
                        elif censor_method == 3:
                            # Invert colors method
                            frame = invert_colors_and_overlay_text(frame, x, y, w, h,
                                                                    translate_classname(result['class']))
                        elif censor_method == 4:
                            # Motion blur method
                            frame = motion_blur_and_overlay_text(frame, x, y, w, h,
                                                                  translate_classname(result['class']))
                        elif censor_method == 5:
                            # Dithering method
                            frame = dither_and_overlay_text(frame, x, y, w, h,
                                                            translate_classname(result['class']))

        # Display the processed frame with progress bar
        progress += 1
        progress_width = int((progress / total_frames) * new_width)
        cv2.rectangle(frame, (0, new_height - 1), (progress_width, new_height), (0, 255, 0), -1)
        cv2.imshow("Processed Frame", frame)
        cv2.setWindowTitle("Processed Frame", "Frame: " + str(progress) + "/" + str(total_frames))

        cv2.waitKey(1)

        # Write the censored frame to the output video file
        out.write(frame)

        # Remove the temporary image file
        os.remove(temp_image_path)

    # Release the video capture object and close the output video file
    cap.release()
    out.release()

    # Remove the temporary directory
    temp_dir.cleanup()

    # Destroy the OpenCV window
    cv2.destroyAllWindows()




# Function to handle the button click event
def select_video_file():
    root = Tk()
    root.withdraw()  # Hide the main window

    # Get the path of the selected video file
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4")])

    if video_path:
        # Specify the output path for the censored video
        output_path = os.path.splitext(video_path)[0] + "_censored.mp4"

        # Get censoring method selection
        censor_method = censor_method_var.get()

        # Call the function to detect and censor explicit frames in the video and save the censored video
        detect_and_censor_video(video_path, output_path, censor_method)

        # Display a message when processing is complete
        Label(root, text="Video censorship completed.").pack()

    root.mainloop()

# Create the main window
root = Tk()
root.title("Video Censorship")
root.geometry('640x700')
root.configure()

# Call the function to display the logo
display_logo()

# Create a button to select the video file
Button(root, text="Select Video File", command=select_video_file).pack()

# Create radio buttons for censoring methods
censor_method_var = IntVar()
censor_method_var.set(1)  # Default selection is Pixelation method

Label(root, text="Censoring Method:").pack()

# Create radio buttons for censoring methods with background color #962A49
Radiobutton(root, text="Pixelation", variable=censor_method_var, value=1).pack()
Radiobutton(root, text="Blurring", variable=censor_method_var, value=2).pack()
Radiobutton(root, text="Invert Colors", variable=censor_method_var, value=3).pack()
Radiobutton(root, text="Motion Blur", variable=censor_method_var, value=4).pack()
#removed, too slow: Radiobutton(root, text="Dithering", variable=censor_method_var, value=5).pack()

# Create a button to quit the application
Button(root, text="Quit", command=quit_application).pack()

# Run the main event loop
root.mainloop()
