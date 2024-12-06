from argparse import ArgumentParser
import cv2
import numpy as np

from mmdet.apis import init_detector
from libs.api.inference import inference_one_image
from libs.utils.visualizer import visualize_lanes


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('video', help='Video file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default='result.mp4', help='Path to output file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold'
    )
    args = parser.parse_args()
    return args


def process_frame(model, frame):
    original_size = frame.shape[:2]  # (height, width)
    # 프레임에 대한 추론 수행
    src, preds = inference_one_image(model, frame)
    # 결과 시각화
    frame_show = visualize_lanes(src, preds)
    # 원본 크기로 다시 리사이즈
    frame_show = cv2.resize(frame_show, (original_size[1], original_size[0]))
    # uint8 형식으로 변환 확인
    if frame_show.dtype != np.uint8:
        frame_show = frame_show.astype(np.uint8)
    return frame_show


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        return
    
    # 비디오 writer 설정
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video: {args.video}")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps}")
    print(f"Resolution: {width}x{height}")
    
    # VideoWriter 설정 시 파일 확장자를 .avi로 통일
    output_path = args.out_file.replace('.mp4', '.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(
        output_path,
        fourcc, 
        fps, 
        (width, height)
    )
    
    if not video_writer.isOpened():
        print("Error: Could not create video writer")
        return
    
    # 각 프레임에 대해 처리
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 10 == 0:  # 진행 상황 표시
            print(f"Processing frame: {frame_count}/{total_frames}")
        
        # 프레임 처리
        frame_show = process_frame(model, frame)
        
        # 프레임 형식 확인 및 저장
        if frame_show is not None and frame_show.shape == (height, width, 3):
            video_writer.write(frame_show)
        else:
            print(f"Warning: Invalid frame at frame {frame_count}")
            print(f"Frame shape: {frame_show.shape if frame_show is not None else 'None'}")
            print(f"Expected shape: {(height, width, 3)}")
    
    # 정리
    cap.release()
    video_writer.release()
    print(f"\nDone! Output saved to: {output_path}")


if __name__ == '__main__':
    args = parse_args()
    main(args)