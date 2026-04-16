# Konvolúciós neurális hálózatok
:label:`chap_cnn`

A képadatokat kétdimenziós pixelrácsként ábrázoljuk, legyen a kép egyszínű vagy színes. Ennek megfelelően minden pixel egy vagy több numerikus értéknek felel meg. Eddig figyelmen kívül hagytuk ezt a gazdag struktúrát, és a képeket számvektorként kezeltük azzal, hogy *kiterítettük* őket, függetlenül a pixelek közötti térbeli kapcsolatoktól. Ez a mélységesen nem kielégítő megközelítés szükséges volt ahhoz, hogy az így kapott egydimenziós vektorokat egy teljesen összekötött MLP-n keresztül vezessük.

Mivel ezek a hálózatok invariánsak a jellemzők sorrendjére nézve, hasonló eredményeket kapnánk, függetlenül attól, hogy megőrizzük-e a pixelek térbeli struktúrájának megfelelő sorrendet, vagy az MLP paramétereinek illesztése előtt permutáljuk a tervmátrixunk oszlopait. Ideális esetben kihasználnánk azt az előzetes tudásunkat, hogy a szomszédos pixelek jellemzően összefüggenek egymással, hogy hatékony modelleket építsünk a képadatokból való tanuláshoz.

Ez a fejezet a *konvolúciós neurális hálózatokat* (CNN) mutatja be
:cite:`LeCun.Jackel.Bottou.ea.1995`, a neurális hálózatok egy hatékony családját, amelyet pontosan erre a célra terveztek. A CNN-alapú architektúrák mára mindenütt jelen vannak a számítógépes látás területén. Például az Imagenet gyűjteményen :cite:`Deng.Dong.Socher.ea.2009` csak a konvolúciós neurális hálózatok, röviden Convnetek alkalmazása hozott jelentős teljesítménybeli javulást :cite:`Krizhevsky.Sutskever.Hinton.2012`.

A modern CNN-ek, ahogyan köznyelvben nevezik őket, tervüket a biológiából, a csoportelméletből és a kísérleti próbálkozások egészséges adagjából merítik. A pontos modellek elérésében mutatott mintahatékonyságuk mellett a CNN-ek jellemzően számítási szempontból is hatékonyak, egyrészt mert kevesebb paramétert igényelnek, mint a teljesen összekötött architektúrák, másrészt mert a konvolúciók könnyen párhuzamosíthatók a GPU-magok között :cite:`Chetlur.Woolley.Vandermersch.ea.2014`. Következésképpen a szakemberek szinte mindig alkalmaznak CNN-eket, ha lehetséges, és egyre inkább komoly versenytársakként jelennek meg még az egydimenziós szekvenciastruktúrájú feladatokban is, mint például az audio :cite:`Abdel-Hamid.Mohamed.Jiang.ea.2014`, a szöveg :cite:`Kalchbrenner.Grefenstette.Blunsom.2014` és az idősor-elemzés :cite:`LeCun.Bengio.ea.1995`, ahol hagyományosan visszatérő neurális hálózatokat használnak. A CNN-ek néhány ügyes adaptációja gráfstruktúrált adatokra :cite:`Kipf.Welling.2016` és ajánlórendszerekben is alkalmazhatóvá tette őket.

Először mélyebben belemerülünk a konvolúciós neurális hálózatok motivációjába. Ezt követi az összes konvolúciós hálózat gerincét alkotó alapvető műveletek bemutatása. Ezek közé tartoznak maguk a konvolúciós rétegek, az aprólékos részletek, beleértve a párnázást és a lépésközt, a pooling rétegek, amelyek az információt összesítik a szomszédos térbeli régiók között, a több csatorna használata minden rétegben, valamint a modern architektúrák struktúrájának alapos tárgyalása. A fejezetet a LeNet teljes, működő példájával zárjuk, amely az első sikeresen alkalmazott konvolúciós hálózat volt, jóval a modern deep learning felemelkedése előtt. A következő fejezetben néhány népszerű és viszonylag újabb CNN-architektúra teljes megvalósításába merülünk bele, amelyek tervei a modern szakemberek által általánosan használt technikák nagy részét képviselik.

```toc
:maxdepth: 2

why-conv
conv-layer
padding-and-strides
channels
pooling
lenet
```

