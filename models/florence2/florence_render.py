import cv2
import json
import os

def render_video(input_video_path, input_json_path, output_video_path):
    with open(input_json_path, 'r') as f:
        detections_data = json.load(f)

    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_key = str(frame_count)

        if frame_key in detections_data:
            for det in detections_data[frame_key]:
                x1, y1, x2, y2 = map(int, det["bbox"])
                label = det["label"]
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Video saved in: {output_video_path}")

if __name__ == "__main__":
    FILE_INPUT = r"input\video\warehouse robot.mp4"
    JSON_INPUT = r"output\bbox\detections.json"
    
    video_basename = os.path.basename(FILE_INPUT)
    VIDEO_OUTPUT = os.path.join("output", "video", f"florence_{video_basename}")
    
    render_video(FILE_INPUT, JSON_INPUT, VIDEO_OUTPUT)