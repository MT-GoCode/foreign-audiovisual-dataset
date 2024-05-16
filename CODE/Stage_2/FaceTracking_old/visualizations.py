import cv2
import imutils

def show_output(video_path, bounding_boxes, output_video_path, predictions=None):
    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video.")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define colors for different IDs
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255)]

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Process each frame
    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=1024)

        # Draw bounding boxes and predictions for each ID
        for id, boxes in bounding_boxes.items():
            if frame_index in boxes:
                box = boxes[frame_index]
                color = colors[id % len(colors)]  # Choose color based on ID
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

                # Add predictions if available
                if predictions and id in predictions:
                    pred_lines = predictions[id]
                    y_offset = y - 10
                    for line in pred_lines:
                        cv2.putText(frame, line, (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        y_offset -= 20  # Adjust the offset for the next line


        frame = cv2.resize(frame, (width, height))
        # Write the frame into the output video
        out.write(frame)

        frame_index += 1

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()