* sonuç odaklı git, bir videoda (tercihen çok temiz olmayan-sisli, güneş parlamalı vs.-, youtube'dan rastgele alınmış bir video) sonucu gösterebil.
* modelin tüm frame'leri tek tek işlemesi yerine performans ve fps kazanmak için bir frame'de segmentasyon yapıp sonucu diğer frame'lere aktarıp kullanmayı dene.
* gemi gibi objelerin merkezlerinin ufuk çizgisine olan uzaklığını bul (bunun için yolov8 gibi bir obj. detection modeli gerekebilir, 'şirket'in modeli de burada kullanılabilir.)

<hr>

Gemi Gibi Yapıların Ufuk Çizgisine Uzaklığını Bulma
Sunumda anlatılanlara ek olarak, tespit edilen ufuk çizgisini kullanarak diğer gemilerin veya nesnelerin uzaklığını nasıl tahmin edebileceğimizi şu şekilde açıklayabilirsiniz:

Bu işlem için iki temel adıma ihtiyacımız var: Nesne Tespiti ve Geometrik Analiz.

Nesne Tespiti: Öncelikle, YOLOv8-seg veya benzeri bir nesne tanıma modeli kullanarak görüntüdeki diğer gemileri tespit etmemiz gerekir. Bu model bize geminin konumunu bir sınırlayıcı kutu (bounding box) içinde verir.

Referans Olarak Ufuk Çizgisi: Ufuk çizgisi, sonsuzdaki bir referans noktasıdır. Kamera sabit bir yükseklikte olduğu sürece, görüntüdeki bir nesnenin ufuk çizgisine olan dikey mesafesi, o nesnenin bize olan gerçek uzaklığıyla ters orantılıdır. Yani, bir geminin alt kısmı ufuk çizgisine ne kadar yakınsa, o gemi o kadar uzaktadır. Geminin alt kısmı görüntüde ne kadar aşağıdaysa (ufuk çizgisinden ne kadar uzaksa), o gemi o kadar yakındır.

Mesafe Tahmini:

Kalibrasyonsuz (Göreceli) Tahmin: En basit yöntem, "Gemi A, Gemi B'den daha yakında" gibi göreceli tahminler yapmaktır. Bu, sadece nesnelerin ufuk çizgisine olan piksel mesafelerini karşılaştırarak yapılabilir.

Kalibrasyonlu (Metrik) Tahmin: Eğer kameranın yerden (deniz seviyesinden) yüksekliği (h) ve odak uzaklığı gibi içsel parametreleri biliniyorsa, basit trigonometri kullanarak piksel mesafesini metre cinsinden yaklaşık bir uzaklığa çevirmek mümkündür. Bu, "pinhole kamera modeli" ve üçgen benzerliği kullanılarak formüle edilebilir. Bu yöntem, otonom sistemlerin çarpışma önleme (collision avoidance) manevraları için kritik öneme sahiptir.

Kısacası, ufuk çizgisi segmentasyonu ve nesne tanımayı birleştirdiğimizde, sadece "orada bir gemi var" demekle kalmaz, aynı zamanda o geminin yaklaşık olarak ne kadar uzakta olduğunu da anlayabiliriz.