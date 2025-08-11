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