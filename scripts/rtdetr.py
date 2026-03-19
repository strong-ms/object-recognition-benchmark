import cv2
import datetime
import os
import torch
import csv
from ultralytics import RTDETR

def test_rtdetr_video(input_video_path, output_dir, model_name="rtdetr-l.pt", conf_threshold=0.5):
    
    # Hardware check
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Hardware acceleration enabled. GPU detected: {gpu_name}")
        compute_device = '0'  # NVIDIA GPU identifier
    else:
        print("WARNING: CUDA not detected. Inference will be performed on CPU.")
        compute_device = 'cpu'
    
    # Initialization of the model.
    print(f"Loading model {model_name}...")
    model = RTDETR(model_name)

    # Opening the input video stream
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video {input_video_path}")
        return

    # Extraction of video properties for writer configuration
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_str = model_name.split('.')[0]
    conf_str = str(conf_threshold).replace('.', '')
    
    # Configuration of output paths (Video and CSV)
    output_filename = f"{model_str}_conf{conf_str}_{timestamp}.mp4"
    output_video_path = os.path.join(output_dir, "video", output_filename)
    
    csv_filename = f"metrics_{model_str}_conf{conf_str}_{timestamp}.csv"
    csv_path = os.path.join(output_dir, "metrics", csv_filename)

    # Configuration of the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Initialization of the CSV file
    csv_file = open(csv_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Frame", "Tempo_Inferenza_ms", "Confidenza_Media"])

    print(f"Starting video processing: {input_video_path}")
    print(f"Resolution: {width}x{height} at {fps} FPS. Total frames: {total_frames}")

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Performing inference on the current frame
        results = model(frame, stream=True, conf=conf_threshold, verbose=False, device=compute_device)

        for result in results:
            # Calculation and saving of metrics
            inference_time = result.speed['inference'] if hasattr(result, 'speed') and 'inference' in result.speed else 0.0
            
            # Average confidence calculation
            if result.boxes is not None and len(result.boxes.conf) > 0:
                avg_conf = result.boxes.conf.mean().item()
            else:
                avg_conf = 0.0
            
            # Writing the row to the CSV file
            csv_writer.writerow([frame_count, f"{inference_time:.2f}", f"{avg_conf:.4f}"])

            # result.plot() generates a new frame with the bounding boxes and labels superimposed
            annotated_frame = result.plot()
            out.write(annotated_frame)

        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames...")

    # Closing and releasing resources
    cap.release()
    out.release()
    csv_file.close()
    cv2.destroyAllWindows()
    
    print(f"Processing completed. Video saved in: {output_video_path}")
    print(f"Metrics saved in: {csv_path}")

if __name__ == "__main__":
    FILE_INPUT = r"input\industrial.mp4"
    OUTPUT_DIRECTORY = r"output"
    
    test_rtdetr_video(FILE_INPUT, OUTPUT_DIRECTORY, model_name="rtdetr-l.pt", conf_threshold=0.5)