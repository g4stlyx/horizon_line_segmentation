# TODO

* sonuç odaklı git, bir videoda (tercihen çok temiz olmayan-sisli, güneş parlamalı vs.-, youtube'dan rastgele alınmış bir video) sonucu gösterebil.

## Preprocessing update for a way better ship-aware hld
* ship-aware hld ve segmentasyon için: 
    * eğitim öncesi data hazırlığı yapılırken YOLO veya RT-DETR vs. tabanlı, bizim oluşturduğumuz modellerle ship-aware maskeler içeren bir dataset çıkar. 
    * preprocess kodunda color-based vs. gemi detect yerine bu modelle gemileri tespit etsin, ona göre mask oluştursun.

## Efficient Video Processing: Avoiding Redundant Frame-by-Frame Analysis
`` modelin tüm frame'leri tek tek işlemesi yerine performans ve fps kazanmak için bir frame'de segmentasyon yapıp sonucu diğer frame'lere aktarıp kullanmayı dene. ``

Running a heavy segmentation model on every single frame of a video is often unnecessary and computationally expensive. The horizon line in a maritime setting typically moves slowly and predictably. You can achieve a significant performance boost by running the U-Net model intermittently and using a simpler method to update the horizon for the frames in between.

A straightforward and effective strategy is to run the U-Net prediction at a set interval (e.g., every 10 frames) and simply reuse the last known mask for the intermediate frames.

code:

    elif args.video or args.camera:
        # --- Process a video file or camera feed ---
        if args.video:
            cap = cv2.VideoCapture(args.video)
        else: # args.camera
            cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open video source.")
            exit()
            
        video_writer = None
        if args.save:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            save_path = "output_segmented.mp4"
            video_writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
            print(f"Saving output video to {save_path}")

        print("Processing video with interval-based prediction. Press 'q' to quit.")

        # --- NEW: Frame processing optimization ---
        frame_count = 0
        PROCESS_INTERVAL = 10  # Run the U-Net model every 10 frames
        last_mask = None
        # ----------------------------------------
        
        while True:
            ret, frame = cap.read()
            if not ret: break

            start_time = time.time()
            
            # Only run the full prediction pipeline at the specified interval
            if frame_count % PROCESS_INTERVAL == 0:
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                mask = predict(model, frame_pil, device, transform)
                last_mask = mask  # Cache the new mask
            else:
                # For intermediate frames, reuse the last calculated mask
                mask = last_mask

            # Ensure we have a mask to work with
            if last_mask is not None:
                result_frame = create_overlay(frame, mask)
            else:
                # If no mask yet, just show the original frame
                result_frame = frame

            frame_count += 1
            
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            cv2.putText(result_frame, f"FPS: {fps:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if video_writer:
                video_writer.write(result_frame)

            cv2.imshow('Horizon Segmentation', result_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()


## How can I calculate the distance to detected objects?
`` gemi, şamandıra gibi objelerin merkezlerinin ufuk çizgisine olan uzaklığını bul ``

* add a new function to your inference script that:
    * Derives a simplified horizon line from the U-Net's mask.
    * Finds the contours of ships (using the existing detect_ships_for_correction logic or a separate detector).
    * Calculates the vertical pixel distance between the horizon and the center of each ship.
    * Draws this information onto the output frame.