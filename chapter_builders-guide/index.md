# Építők kézikönyve
:label:`chap_computation`

A hatalmas adathalmazok és a nagy teljesítményű hardver mellett
a kiváló szoftvereszközök nélkülözhetetlen szerepet játszottak
a deep learning rohamos fejlődésében.
A 2007-ben kiadott úttörő Theano könyvtárral kezdődően
a rugalmas, nyílt forráskódú eszközök lehetővé tették a kutatók számára,
hogy gyorsan prototípusokat készítsenek, elkerülve az ismétlődő munkát
a szabványos komponensek újrafelhasználásakor,
miközben megőrizték az alacsony szintű módosítások elvégzésének lehetőségét.
Az idők során a deep learning könyvtárai egyre durvább absztrakciókat kínáltak.
Ahogyan a félvezető-tervezők az egyes tranzisztorok megadásától
a logikai áramkörökön át az kódírásig jutottak,
a neurális hálózatok kutatói az egyes mesterséges neuronok viselkedéséről
a hálózatok egész rétegek alapján való felfogásáig léptek előre,
és ma már sokszor jóval durvább *blokkokban* gondolkodva terveznek architektúrákat.


Eddig bemutattunk néhány alapvető gépi tanulási fogalmat,
fokozatosan eljutva a teljes értékű deep learning modellekig.
Az előző fejezetben egy MLP minden komponensét nulláról implementáltuk,
és megmutattuk, hogyan lehet a magas szintű API-k segítségével
könnyedén létrehozni ugyanazokat a modelleket.
Hogy ilyen gyorsan idáig eljussunk, *igénybe vettük* a könyvtárakat,
de kihagytuk a részleteket arról, *hogyan működnek*.
Ebben a fejezetben félrehúzzuk a függönyt,
és mélyebben beleásunk a deep learning számítás kulcsfontosságú összetevőibe,
nevezetesen a modellépítésbe, a paraméterek elérésébe és inicializálásába,
az egyéni rétegek és blokkok tervezésébe, a modellek lemezre írásába és onnan való olvasásába,
valamint a GPU-k kihasználásába a drámai gyorsítás érdekében.
Ezek az ismeretek az *egyszerű felhasználóból* *haladó felhasználóvá* tesznek,
megadva az eszközöket egy érett deep learning könyvtár előnyeinek kiaknázásához,
miközben megőrzi a rugalmasságot összetettebb modellek implementálásához,
beleértve azokat is, amelyeket te magad találsz fel!
Bár ez a fejezet nem mutat be új modelleket vagy adathalmazokat,
a következő haladó modellezési fejezetek nagymértékben támaszkodnak ezekre a technikákra.

```toc
:maxdepth: 2

model-construction
parameters
init-param
lazy-init
custom-layer
read-write
use-gpu
```

