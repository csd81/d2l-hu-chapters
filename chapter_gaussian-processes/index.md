# Gauss-folyamatok
:label:`chap_gp`

**Andrew Gordon Wilson** (*New York University and Amazon*)


A Gauss-folyamatok mindenütt jelen vannak. Már számos példával találkoztál Gauss-folyamatokra anélkül, hogy észrevetted volna. Bármely modell, amely lineáris a paramétereiben és Gauss-eloszlással rendelkezik a paraméterek felett, egy Gauss-folyamat. Ez az osztály diszkrét modelleket is felölel, beleértve a véletlen bolyongást és autoregresszív folyamatokat, valamint folytonos modelleket, beleértve a Bayes-i lineáris regressziós modelleket, polinomokat, Fourier-sorokat, radiális bázisfüggvényeket, sőt neurális hálózatokat végtelen számú rejtett egységgel. Elterjedt vicc, hogy „minden a Gauss-folyamatok speciális esete".

A Gauss-folyamatok megismerése három okból fontos: (1) _függvénytér_ perspektívát nyújtanak a modellezéshez, amely sokkal könnyebben megközelíthetővé teszi számos modellosztály, köztük a deep neurális hálózatok megértését; (2) rendkívül széles alkalmazási körrel rendelkeznek, ahol a legkorszerűbbek, beleértve az aktív tanulást, a hiperparaméter-tanulást, az auto-ML-t és a téridőbeli regressziót; (3) az elmúlt néhány évben az algoritmikus fejlődés egyre skálázhatóbbá és relevánsabbá tette a Gauss-folyamatokat, harmonizálva a deep learninggel olyan keretrendszereken keresztül, mint a [GPyTorch](https://gpytorch.ai) :cite:`Gardner.Pleiss.Weinberger.Bindel.Wilson.2018`. Valójában a Gauss-folyamatok és a deep neurális hálózatok nem versengő megközelítések, hanem rendkívül kiegészítők, és nagy sikerrel kombinálhatók. Ezek az algoritmikus fejlesztések nemcsak a Gauss-folyamatokra vonatkoznak, hanem a numerikus módszerek alapját képezik, amely széles körben hasznos a deep learningben.

Ebben a fejezetben bemutatjuk a Gauss-folyamatokat. A bevezető notebookban intuitívan kezdjük megérteni, hogy mik is a Gauss-folyamatok, és hogyan modelleznek közvetlenül függvényeket. A priorokról szóló notebookban arra összpontosítunk, hogyan specifikáljunk Gauss-folyamat priorokat. Közvetlenül összekapcsoljuk a hagyományos súlytér megközelítést a modellezéshez a függvénytérrel, ami segít a gépi tanulási modellek, köztük a deep neurális hálózatok felépítésének és megértésének átgondolásában. Ezután bemutatjuk a népszerű kovariancia függvényeket, más néven _kerneleket_, amelyek egy Gauss-folyamat általánosítási tulajdonságait vezérlik. Egy adott kernellel rendelkező Gauss-folyamat priort definiál a függvények felett. Az inferenciáról szóló notebookban megmutatjuk, hogyan használjunk adatokat a _poszterior_ inferálásához a predikciók elkészítéséhez. Ez a notebook nulláról írt kódot tartalmaz a Gauss-folyamatokkal történő predikciókhoz, valamint egy bevezetést a GPyTorch-ba. A következő notebookokban bemutatjuk a Gauss-folyamatok mögött álló numerikát, amely hasznos a Gauss-folyamatok skálázásához, de egyben erős általános alapot is nyújt a deep learninghez, valamint fejlett felhasználási eseteket, mint például a hiperparaméter-hangolás a deep learningben. Példáink a GPyTorch-ot használják, amely skálázhatóvá teszi a Gauss-folyamatokat, és szorosan integrált a deep learning funkcionalitással és a PyTorch-csal.

```toc
:maxdepth: 2

gp-intro
gp-priors
gp-inference
```

