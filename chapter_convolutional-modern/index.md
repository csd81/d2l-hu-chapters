# Modern konvolúciós neurális hálózatok
:label:`chap_modern_cnn`

Most, hogy megértettük a konvolúciós neurális hálózatok összekapcsolásának alapjait, tegyünk egy körútat a modern CNN-architektúrák között. Ez a körút szükségképpen nem teljes, hiszen rengeteg izgalmas új tervezési megoldás jelenik meg folyamatosan. Fontosságukat az adja, hogy nemcsak közvetlenül alkalmazhatók látási feladatokra, hanem alapvető jellemzőgenerátorként is szolgálnak összetettebb feladatokhoz, mint például a követés
:cite:`Zhang.Sun.Jiang.ea.2021`, a szegmentálás :cite:`Long.Shelhamer.Darrell.2015`, az objektumdetektálás :cite:`Redmon.Farhadi.2018` vagy a stílustranszformáció
:cite:`Gatys.Ecker.Bethge.2016`. Ebben a fejezetben a legtöbb szakasz egy-egy jelentős CNN-architektúrának felel meg, amelyek valamilyen időpontban (vagy jelenleg is) alapmodellként szolgáltak számos kutatási projekt és élesben telepített rendszer számára. Ezek a hálózatok mindegyike rövid ideig domináns architektúra volt, és sokan nyertesek vagy dobogós helyezések voltak az
[ImageNet-versenyen](https://www.image-net.org/challenges/LSVRC/),
amely 2010 óta a felügyelt tanulás előrehaladásának mércéjeként szolgál a számítógépes látásban. Csak a közelmúltban kezdték a Transformerek kiszorítani a CNN-eket, :citet:`Dosovitskiy.Beyer.Kolesnikov.ea.2021` munkájával kezdődően, amelyet a Swin Transformer :cite:`liu2021swin` követett. Ezt a fejlődést később tárgyaljuk a :numref:`chap_attention-and-transformers` fejezetben.

Bár a *mély* neurális hálózatok gondolata meglehetősen egyszerű (csak halmozzunk össze egy csomó réteget), a teljesítmény nagyon eltérő lehet az architektúrák és a hiperparaméterek megválasztásától függően. Az ebben a fejezetben bemutatott neurális hálózatok az intuíció, néhány matematikai meglátás és rengeteg próbálkozás eredményei. Ezeket a modelleket időrendi sorrendben mutatjuk be, részben azért, hogy érzékeltessük a terület történetét, hogy saját intuíciót alakíthass ki arról, merre tart a terület, és esetleg saját architektúrákat fejleszthess. Például az ebben a fejezetben ismertetett batchnormalizáció és reziduális kapcsolatok két népszerű ötletet kínáltak a mély modellek tanításához és tervezéséhez, amelyeket azóta a számítógépes látáson túlmutató architektúrákra is alkalmaztak.

Modern CNN-körútunkat az AlexNet-tel :cite:`Krizhevsky.Sutskever.Hinton.2012` kezdjük, amely az első nagy méretű hálózat volt, amelyet nagyszabású látási kihívásban a hagyományos számítógépes látási módszerek legyőzésére alkalmaztak; a VGG-hálózattal :cite:`Simonyan.Zisserman.2014`, amely ismétlődő elemblokkok sorozatát alkalmazza; a Network in Network (NiN) architektúrával, amely egész neurális hálózatokat konvolválja patchenként a bemenetek felett :cite:`Lin.Chen.Yan.2013`; a GoogLeNet-tel, amely többágú konvolúciókat alkalmazó hálózatokat használ :cite:`Szegedy.Liu.Jia.ea.2015`; a reziduális hálózattal (ResNet) :cite:`He.Zhang.Ren.ea.2016`, amely napjainkban is az egyik legnépszerűbb, azonnal alkalmazható architektúra a számítógépes látásban; a ResNeXt blokkokkal :cite:`Xie.Girshick.Dollar.ea.2017` ritkább kapcsolatokhoz; valamint a DenseNet-tel :cite:`Huang.Liu.Van-Der-Maaten.ea.2017`, amely a reziduális architektúra általánosítása. Az idők során számos speciális optimalizálást fejlesztettek ki a hatékony hálózatokhoz, például a koordináta-eltolásokat (ShiftNet) :cite:`wu2018shift`. Ez végül elvezetett a hatékony architektúrák automatikus kereséséhez, mint például a MobileNet v3 :cite:`Howard.Sandler.Chu.ea.2019`. Ide tartozik :citet:`Radosavovic.Kosaraju.Girshick.ea.2020` félig automatikus tervezési feltárása is, amely a RegNetX/Y modellekhez vezetett, amelyeket később tárgyalunk ebben a fejezetben. A munka annyiban tanulságos, amennyiben utat kínál a nyers számítási kapacitás és a kísérletező leleményességének ötvözéséhez a hatékony tervezési terek keresésében. Figyelemre méltó :citet:`liu2022convnet` munkája is, mivel megmutatja, hogy a tanítási technikák (például az optimalizálók, az adataugmentáció és a regularizáció) döntő szerepet játszanak a pontosság javításában. Azt is megmutatja, hogy régóta fennálló feltételezéseket — például a konvolúciós ablak méretét — újra kell gondolni a számítási kapacitás és az adatok növekedése miatt. Ezeket és még sok más kérdést idővel tárgyalunk ebben a fejezetben.

```toc
:maxdepth: 2

alexnet
vgg
nin
googlenet
batch-norm
resnet
densenet
cnn-design
```
