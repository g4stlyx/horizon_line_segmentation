# TODO

* sonuç odaklı git, bir videoda (tercihen çok temiz olmayan-sisli, güneş parlamalı vs.-, youtube'dan rastgele alınmış bir video) sonucu gösterebil.

## Preprocessing update for a way better ship-aware hld
`` DONE, using the same RT-DETR obj. det. model taken from barissglc``
* ship-aware hld ve segmentasyon için: 
    * eğitim öncesi data hazırlığı yapılırken YOLO veya RT-DETR vs. tabanlı, bizim oluşturduğumuz modellerle ship-aware maskeler içeren bir dataset çıkar. 
    * preprocess kodunda color-based vs. gemi detect yerine bu modelle gemileri tespit etsin, ona göre mask oluştursun.

## Efficient Video Processing: Avoiding Redundant Frame-by-Frame Analysis
`` DONE ``

`` modelin tüm frame'leri tek tek işlemesi yerine performans ve fps kazanmak için bir frame'de segmentasyon yapıp sonucu diğer frame'lere aktarıp kullanmayı dene. ``

Running a heavy segmentation model on every single frame of a video is often unnecessary and computationally expensive. The horizon line in a maritime setting typically moves slowly and predictably. You can achieve a significant performance boost by running the U-Net model intermittently and using a simpler method to update the horizon for the frames in between.

A straightforward and effective strategy is to run the U-Net prediction at a set interval (e.g., every 10 frames) and simply reuse the last known mask for the intermediate frames.

## Calculate the distance between detected objects and the horizon line.
`` gemi, şamandıra gibi objelerin merkezlerinin ufuk çizgisine olan uzaklığını bul `` <br>

`` DONE using a HF RT-DETR object detection model taken from barissglc. Since a second, heavy model is used beside my unet segmentation model, we got an FPS drop from ~40 to ~15 tho.``

## Performance Improvements

``Using U-Net + RT-DETR in the runner code for distance calculation made the app a lot heavier then it is used to be (40 FPS -> 10 FPS), find a way to increase the FPS. ``

this is ```KIND OF DONE``` (ofc it can always be better):
* Horizon computation: kept the new, accurate sky/water separator with object-occlusion bridging. Also vectorized it to reduce per-column Python loops.
* Enabled cudnn.benchmark when using CUDA to accelerate convolutions with fixed-size inputs.
* Added optional FP16 inference for the U-Net in the runner; enabled by default in calls.

---

* ~15-20 FPS on GTX 1650 Super 4GB with these settings:
    * py z_unet_runner_dist_calc_rtdetr_obj_det_3_class.py --video .\0example_data\VIS_Onshore\Videos\MVI_1614_VIS.avi --rtdetr-model rtdetr_obj_det_model/final_best_model --prefer-rtdetr --rtdetr-conf 0.25 --rtdetr-classes 0,1,2,4,6,7,8 --rtdetr-interval 5 --band-up 160 --band-down 140 --min-area 350 --show-horizon --save

## New Classes on Segmentation Masks

0: Sky, 1: Sea, 2: Ship/Obstacle, 3: Land/coastal

## Distance to the Horizon Line Calculation Improvements

Calculate the distance between horizon line and the center of the object in sea mile/kilometer instead of a pixel-based calculation.