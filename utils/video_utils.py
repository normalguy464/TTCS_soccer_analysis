import cv2

def read_video(video_path, process_func=None, resize_factor=0.5, batch_size=None):
    """Đọc video với tùy chọn giảm kích thước để tiết kiệm bộ nhớ"""
    import gc  # Thêm garbage collector
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    
    # Nếu xử lý theo batch
    if batch_size:
        all_frames = []
        for i in range(0, total_frames, batch_size):
            batch_frames = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            
            # Đọc batch_size frames
            for j in range(batch_size):
                if i + j >= total_frames:
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Giảm kích thước nếu cần
                if resize_factor != 1.0:
                    width = int(frame.shape[1] * resize_factor)
                    height = int(frame.shape[0] * resize_factor)
                    frame = cv2.resize(frame, (width, height))
                
                batch_frames.append(frame)
            
            # Xử lý và giải phóng bộ nhớ
            if process_func and batch_frames:
                batch_results = process_func(batch_frames)
                all_frames.extend(batch_results)
            else:
                all_frames.extend(batch_frames)
                
            del batch_frames
            gc.collect()  # Giải phóng bộ nhớ
            
        frames = all_frames
    else:
        # Đọc toàn bộ video với resize
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Giảm kích thước frame để giảm bộ nhớ
            if resize_factor != 1.0:
                width = int(frame.shape[1] * resize_factor)
                height = int(frame.shape[0] * resize_factor)
                frame = cv2.resize(frame, (width, height))
                
            frames.append(frame)
            
            # Giải phóng bộ nhớ định kỳ
            if len(frames) % 100 == 0:
                gc.collect()
    
    cap.release()
    return frames

def save_video(output_video_frames, output_video_path):
    # height, width, _ = output_video_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 25, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))

    for frame in output_video_frames:
        out.write(frame)

    out.release()
    print(f"Video saved to {output_video_path}")