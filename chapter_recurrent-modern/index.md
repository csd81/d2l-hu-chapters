# Modern Rekurrens Neurális Hálózatok
:label:`chap_modern_rnn`

Az előző fejezet bemutatta a rekurrens neurális hálózatok (RNN-ek) mögötti kulcsfontosságú ötleteket.
Azonban, akárcsak a konvolúciós neurális hálózatoknál,
az RNN-architektúrákban is rengeteg innováció ment végbe,
amelyek végül számos összetett tervhez vezettek,
amelyek a gyakorlatban sikeresnek bizonyultak.
Különösen a legismertebb tervek tartalmaznak
mechanizmusokat az RNN-eket sújtó hírhedt
numerikus instabilitás (eltűnő és robbanó gradiensek) enyhítésére.
Felidézve, hogy a :numref:`chap_rnn` fejezetben
a robbanó gradienseket egy egyszerű gradiensvágási
heurisztika alkalmazásával kezeltük.
Bár ez a megközelítés hatékony,
az eltűnő gradiensek problémáját nyitva hagyja.

Ebben a fejezetben bemutatjuk a legsikeresebbnek bizonyult
RNN-architektúrák mögötti kulcsgondolatokat a sorozatokhoz,
amelyek két cikkből erednek.
Az első, a *Long Short-Term Memory* :cite:`Hochreiter.Schmidhuber.1997`,
bevezeti a *memóriacellát*, egy számítási egységet, amely felváltja
a hálózat rejtett rétegének hagyományos csomópontjait.
Ezen memóriacellák segítségével a hálózatok képesek leküzdeni
a korábbi rekurrens hálózatok tanítása során
felmerülő nehézségeket.
Intuitívan, a memóriacella elkerüli
az eltűnő gradiens problémáját azáltal, hogy
minden memóriacella belső állapotában az értékeket
1-es súlyú rekurrens élen továbbítja
sok egymást követő időlépésen keresztül.
A szorzó kapuk egy halmaza segít a hálózatnak meghatározni
nemcsak azt, hogy milyen bemeneteket engedjen
a memóriaállapotba,
hanem azt is, hogy a memóriaállapot tartalma
mikor befolyásolja a modell kimenetét.

A második cikk, a *Bidirectional Recurrent Neural Networks* :cite:`Schuster.Paliwal.1997`,
bevezet egy olyan architektúrát, amelyben
mind a jövőből (következő időlépésekből)
mind a múltból (megelőző időlépésekből) érkező információt
felhasználják a sorozat bármely pontján lévő kimenet meghatározásához.
Ez szemben áll a korábbi hálózatokkal,
amelyekben csak a múltbeli bemenet befolyásolhatja a kimenetet.
A kétirányú RNN-ek a természetes nyelvi feldolgozás
sorozatcímkézési feladatainak alapvető eszközévé váltak,
egyéb feladatok mellett is.
Szerencsére a két innováció nem zárja ki egymást,
és sikeresen kombinálták őket foném-osztályozáshoz
:cite:`Graves.Schmidhuber.2005` és kézírás-felismeréshez :cite:`graves2008novel`.


A fejezet első részei elmagyarázzák az LSTM-architektúrát,
annak egy könnyebb változatát, a kapuzott rekurrens egységet (GRU),
a kétirányú RNN-ek mögötti kulcsgondolatokat,
valamint röviden bemutatják, hogyan kapcsolhatók össze
az RNN-rétegek mély RNN-ek alkotásához.
Ezt követően az RNN-ek alkalmazását vizsgáljuk
sorozatból sorozatba irányuló feladatokban,
bemutatva a gépi fordítást
az olyan kulcsgondolatokkal együtt, mint a *kódoló–dekódoló* architektúrák és a *beam search*.

```toc
:maxdepth: 2

lstm
gru
deep-rnn
bi-rnn
machine-translation-and-dataset
encoder-decoder
seq2seq
beam-search
```

