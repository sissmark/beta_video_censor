import os
import cv2
from nudenet import NudeDetector
from tkinter import filedialog, Tk, Button, Label
import tempfile

# Function to detect and censor explicit frames in a video and save the censored video
def detect_and_censor_video(video_path, output_path):
    # Initialize the NudeDetector
    detector = NudeDetector()

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file is opened successfully
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    # Get the video frame properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create a VideoWriter object to write the censored frames to a new video file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for the output video
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Get the audio codec from the source video
    audio_codec = int(cap.get(cv2.CAP_PROP_FOURCC))

    # Create a VideoWriter object to write the audio to the censored video file
    audio_out = cv2.VideoWriter(output_path, audio_codec, fps, (width, height))

    # Create a temporary directory to save the video frames
    temp_dir = tempfile.TemporaryDirectory()

    # Create a window to display processed frames
    cv2.namedWindow("Processed Frame", cv2.WINDOW_NORMAL)

    # Initialize progress variables
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress = 0
    last_result = None
    last_result_frames_remaining = 0

    # Iterate through each frame
    while True:
        # Read the frame
        success, frame = cap.read()

        if not success:
            break

        # Save the frame as a temporary image file
        temp_image_path = os.path.join(temp_dir.name, "temp_frame.jpg")
        cv2.imwrite(temp_image_path, frame)

        # Detect nudity in the frame
        results = detector.detect(temp_image_path)

        # Check if explicit content is detected
        if results:
            last_result = results
            last_result_frames_remaining = 240

        # If there's a last known result, blur the frame
        if last_result_frames_remaining > 0:
            for result in last_result:
                if result['score'] >= 0.0:
                    x, y, w, h = result['box']
                    # Only censor female parts
                    if "MALE_GENITALIA_EXPOSED" not in result['class'] and "FACE_MALE" not in result['class']:
                        frame = blur_and_desaturate_region_and_overlay_text(frame, x, y, w, h, translate_classname(result['class']))

            last_result_frames_remaining -= 1

        # Display the processed frame with progress bar
        progress += 1
        progress_width = int((progress / total_frames) * width)
        cv2.rectangle(frame, (0, height - 1), (progress_width, height), (0, 0, 0), -1)
        cv2.imshow("Processed Frame", frame)
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

def blur_and_desaturate_region_and_overlay_text(image, x, y, w, h, text, blur_amount=90):
    # Calculate the intersection of the region to blur with the image bounds
    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + w, image.shape[1])
    y2 = min(y + h, image.shape[0])

    # Check if there is an intersection and the region to blur has non-zero dimensions
    if x1 < x2 and y1 < y2 and x2 - x1 > 0 and y2 - y1 > 0:
        # Extract the region to blur
        region = image[y1:y2, x1:x2]

        # Apply Gaussian blur to the region
        if blur_amount % 2 == 0:  # Ensure blur_amount is odd
            blur_amount += 1
        blurred_region = cv2.GaussianBlur(region, (blur_amount, blur_amount), 0)

        # Replace the region with the blurred version
        image[y1:y2, x1:x2] = blurred_region

        # Calculate the position to center the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = x1 + ((x2 - x1) - text_size[0]) // 2
        text_y = y1 + ((y2 - y1) + text_size[1]) // 2

        # Add text overlay
        cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

    return image


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

# Function to handle the button click event
def select_video_file():
    root = Tk()
    root.withdraw()  # Hide the main window

    # Get the path of the selected video file
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4")])

    if video_path:
        # Specify the output path for the censored video
        output_path = os.path.splitext(video_path)[0] + "_censored.mp4"

        # Call the function to detect and censor explicit frames in the video and save the censored video
        detect_and_censor_video(video_path, output_path)

        # Display a message when processing is complete
        Label(root, text="Video censorship completed.").pack()

    root.mainloop()


# Create the main window
root = Tk()
root.title("Video Censorship")
root.geometry('1024x768')

# Create a button to select the video file
Button(root, text="Select Video File", command=select_video_file).pack()

# Run the main event loop
root.mainloop()