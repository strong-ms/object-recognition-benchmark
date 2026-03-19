import cv2
import json
import torch
import time
import csv
from PIL import Image
import re
from transformers import AutoProcessor, AutoModelForCausalLM

def run_inference(input_video_path, output_json_path, output_csv_path, target_list, model_id="microsoft/Florence-2-large"):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, trust_remote_code=True).to(device)

    cap = cv2.VideoCapture(input_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    task_prompt = "<OPEN_VOCABULARY_DETECTION>"

    detections_data = {}
    
    csv_file = open(output_csv_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Frame", "Tempo_Inferenza_ms"])

    frame_count = 0

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
        
        detections_data[frame_count] = frame_detections
        csv_writer.writerow([frame_count, f"{frame_inference_time:.2f}"])
        print(f"Elaborated {frame_count}/{total_frames} frames. Found {len(frame_detections)} objects.")

    cap.release()
    csv_file.close()

    with open(output_json_path, 'w') as f:
        json.dump(detections_data, f, indent=4)

if __name__ == "__main__":
    FILE_INPUT = r"input\trimmed_video.mp4"
    JSON_OUTPUT = r"output\bbox\detections.json"
    CSV_OUTPUT = r"output\metrics\metrics_florence.csv"
    
    # Per l'Open Vocabulary nativo, gli oggetti vengono gestiti come una lista Python standard
    TARGET_OBJECTS = ["Formula 1 car", "mechanic", "racing tire"]    
    run_inference(FILE_INPUT, JSON_OUTPUT, CSV_OUTPUT, TARGET_OBJECTS)