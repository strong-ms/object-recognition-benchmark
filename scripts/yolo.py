import cv2
import os
import torch
import csv
from datetime import datetime
from ultralytics import YOLO

def test_yolo_video(input_video, output_dir, model_name="yolov10n.pt", conf_threshold=0.5):
    
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
    model = YOLO(model_name)

    if "world" in model_name.lower():
        classes_to_detect = ["person", "industrial robot arm with gripper", "box"]
        print(f"Setting custom vocabulary: {classes_to_detect}")
        model.set_classes(classes_to_detect)

    # Opening the input video stream
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error: Unable to open video {input_video}")
        return

    # Extraction of video properties for writer configuration
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    model_str = model_name.split('.')[0]
    
    # Creating output file names
    video_basename = os.path.basename(input_video)
    
    output_filename = f"{model_str}_{video_basename}"
    # Assicuriamoci che la cartella "video" esista
    os.makedirs(os.path.join(output_dir, "video"), exist_ok=True)
    output_video_path = os.path.join(output_dir, "video", output_filename)
    
    # Il percorso del file CSV globale
    os.makedirs(output_dir, exist_ok=True)
    global_csv_path = os.path.join(output_dir,"metrics","metrics.csv")

    # Configuration of the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"Inizio elaborazione video: {input_video}")
    print(f"Risoluzione: {width}x{height} a {fps} FPS. Totale frame: {total_frames}")

    frame_count = 0
    total_inference_time = 0.0
    total_conf_sum = 0.0
    total_detections = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        results = model(frame, stream=True, conf=conf_threshold, verbose=False, device=compute_device)

        for result in results:
            # Accumulo dei tempi di inferenza
            inference_time = result.speed['inference'] if hasattr(result, 'speed') and 'inference' in result.speed else 0.0
            total_inference_time += inference_time
            
            # Accumulo delle confidenze
            if result.boxes is not None and len(result.boxes.conf) > 0:
                total_conf_sum += result.boxes.conf.sum().item()
                total_detections += len(result.boxes.conf)

            # result.plot() generates a new frame with the bounding boxes and labels superimposed
            annotated_frame = result.plot()
            out.write(annotated_frame)

        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames...")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # --- Calcolo delle metriche globali e salvataggio nel CSV ---
    
    avg_frame_process_time = total_inference_time / frame_count if frame_count > 0 else 0.0
    # Media della confidenza su tutti gli oggetti rilevati nell'intero video
    average_confidence = total_conf_sum / total_detections if total_detections > 0 else 0.0
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    file_exists = os.path.isfile(global_csv_path)
    
    with open(global_csv_path, mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        # Se il file non esisteva prima, scrivo l'intestazione
        if not file_exists:
            csv_writer.writerow(["model_name", "input_video_name", "avg_frame_process_time", "average_confidence", "date"])
        
        # Scrivo la riga con le metriche calcolate
        csv_writer.writerow([
            model_name, 
            video_basename, 
            f"{avg_frame_process_time:.2f}", 
            f"{average_confidence:.4f}", 
            current_date
        ])

    print(f"Processing completed. Video saved in: {output_video_path}")
    print(f"Metrics appended to: {global_csv_path}")

if __name__ == "__main__":
    FILE_INPUT = r"input\video\industrial_robot_arm.mp4"
    OUTPUT_DIRECTORY = r"output"
    
    test_yolo_video(FILE_INPUT, OUTPUT_DIRECTORY, model_name="yolov8m-world.pt", conf_threshold=0.4)