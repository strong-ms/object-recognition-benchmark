import cv2
import json
import torch
import time
import csv
import os
from datetime import datetime
from PIL import Image
import re
from transformers import AutoProcessor, AutoModelForCausalLM

def run_inference(input_video_path, output_json_path, output_dir, target_list, model_id="microsoft/Florence-2-large"):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Caricamento del modello {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, trust_remote_code=True).to(device)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Errore: Impossibile aprire il video {input_video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    task_prompt = "<OPEN_VOCABULARY_DETECTION>"
    detections_data = {}
    
    # Setup cartelle e file globali
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    global_csv_path = os.path.join(output_dir, "metrics", "metrics.csv")

    frame_count = 0
    total_inference_time = 0.0

    print(f"Inizio elaborazione video per inferenza: {input_video_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        frame_detections = []
        frame_inference_time = 0.0

        # Esecuzione di un'inferenza separata e indipendente per ogni oggetto nella lista
        for target in target_list:
            full_prompt = task_prompt + target
            inputs = processor(text=full_prompt, images=image_pil, return_tensors="pt").to(device, torch_dtype)

            start_time = time.time()
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    early_stopping=False,
                    do_sample=False,
                    num_beams=3,
                )
            
            # Somma il tempo di inferenza di ogni singolo ciclo per avere la latenza totale del frame
            frame_inference_time += (time.time() - start_time) * 1000

            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            
            clean_text = generated_text.replace('</s>', '').replace('<s>', '').replace('<pad>', '')
            clean_text = clean_text.replace('<OPEN_VOCABULARY_DETECTION>', '').replace('OPEN_VOCABULARY_DETECTION', '')

            pattern = r'([^<]+)((?:<loc_\d+>){4,})'
            matches = re.findall(pattern, clean_text)
            
            for label, loc_string in matches:
                coords = [int(c) for c in re.findall(r'<loc_(\d+)>', loc_string)]
                
                for i in range(0, len(coords), 4):
                    if i + 3 < len(coords):
                        x1 = coords[i] * image_pil.width / 1000.0
                        y1 = coords[i+1] * image_pil.height / 1000.0
                        x2 = coords[i+2] * image_pil.width / 1000.0
                        y2 = coords[i+3] * image_pil.height / 1000.0
                        
                        frame_detections.append({
                            "bbox": [x1, y1, x2, y2],
                            "label": target  # Forziamo l'uso del nome esatto richiesto per pulizia visiva
                        })
        
        # Accumulo del tempo
        total_inference_time += frame_inference_time
        detections_data[frame_count] = frame_detections
        
        if frame_count % 10 == 0:
            print(f"Elaborated {frame_count}/{total_frames} frames...")

    cap.release()

    # Salvataggio del file JSON con le coordinate
    with open(output_json_path, 'w') as f:
        json.dump(detections_data, f, indent=4)
    print(f"Bbox salvate in: {output_json_path}")

    # --- Calcolo metriche globali e scrittura nel CSV in append ---
    avg_frame_process_time = total_inference_time / frame_count if frame_count > 0 else 0.0
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    video_basename = os.path.basename(input_video_path)
    model_name_clean = model_id.split('/')[-1]  # Es. Estrae "Florence-2-large"

    file_exists = os.path.isfile(global_csv_path)
    
    with open(global_csv_path, mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        if not file_exists:
            csv_writer.writerow(["model_name", "input_video_name", "avg_frame_process_time", "average_confidence", "date"])
        
        csv_writer.writerow([
            model_name_clean, 
            video_basename, 
            f"{avg_frame_process_time:.2f}", 
            "N/A", 
            current_date
        ])
    
    print(f"Metriche accodate in: {global_csv_path}")

if __name__ == "__main__":
    FILE_INPUT = r"input\video\warehouse robot.mp4"
    JSON_OUTPUT = r"output\bbox\detections.json"
    OUTPUT_DIRECTORY = r"output"
    
    TARGET_OBJECTS = ["wharehouse automated guided vehicle","box"]    
    run_inference(FILE_INPUT, JSON_OUTPUT, OUTPUT_DIRECTORY, TARGET_OBJECTS)