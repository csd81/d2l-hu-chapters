# Előszó

Néhány évvel ezelőtt még nem dolgoztak deep learning kutatók seregei
nagyvállalatoknál és startupoknál azon, hogy intelligens termékeket és
szolgáltatásokat hozzanak létre. Amikor mi beléptünk erre a területre, a
gépi tanulás nem szerepelt a napilapok címlapjain. A szüleink azt sem
tudták, mi az a gépi tanulás, nemhogy azt, miért választanánk inkább ezt
a pályát az orvosi vagy jogi karrier helyett. A gépi tanulás egyfajta
alapkutatási („blue skies") akadémiai diszciplínának számított, amelynek ipari
jelentősége néhány szűk valós alkalmazási területre korlátozódott,
például a beszédfelismerésre és a számítógépes látásra. Ráadásul ezek
közül sok alkalmazás annyi speciális szaktudást igényelt, hogy gyakran
teljesen különálló területként tekintettek rájuk, amelyekben a gépi
tanulás csak egy kisebb alkotóelem volt. Abban az időben a neurális
hálózatokat, vagyis azoknak a deep learning módszereknek az elődeit,
amelyekre ebben a könyvben összpontosítunk, általában elavultnak
tartották.

Néhány év alatt azonban a deep learning teljesen meglepte a világot, és
gyors fejlődést indított el olyan sokféle területen, mint a
számítógépes látás, a természetesnyelv-feldolgozás, az automatikus
beszédfelismerés, a megerősítéses tanulás és a biomedikai informatika.
Sőt, a deep learning sikere számos, a gyakorlatban is fontos feladatban
még az elméleti gépi tanulás és a statisztika fejlődését is ösztönözte.
Ezekre az eredményekre támaszkodva ma már olyan autókat tudunk építeni,
amelyek minden korábbinál nagyobb önállósággal vezetnek maguktól
(bár kevesebbel, mint amit néhány vállalat elhitetne velünk), olyan
párbeszédrendszereket, amelyek tisztázó kérdések feltevésével hibakeresést
végeznek a kódban, valamint olyan szoftverügynököket, amelyek legyőzik a világ
legjobb emberi játékosait olyan táblajátékokban, mint a Go, ami korábban
évtizedekre lévő áttörésnek tűnt. Ezek az eszközök már most is egyre
nagyobb hatást gyakorolnak az iparra és a társadalomra: megváltoztatják,
hogyan készülnek a filmek, hogyan diagnosztizálunk betegségeket, és egyre
fontosabb szerepet játszanak az alaptudományokban is az asztrofizikától
az éghajlatmodellezésen és az időjárás-előrejelzésen át egészen a
biomedicináig.



## A könyvről

Ez a könyv arra tett kísérletünk, hogy a deep learninget közel hozzuk az
olvasóhoz, és egyszerre tanítsuk meg a *fogalmakat*, az
*összefüggéseket* és a *kódot*.

### Egyetlen közeg, amely egyesíti a kódot, a matematikát és a HTML-t

Ahhoz, hogy bármely számítástechnikai technológia valóban kibontakoztassa
a hatását, jól érthetőnek, jól dokumentáltnak és kiforrott, gondosan
karbantartott eszközök által támogatottnak kell lennie. A legfontosabb
ötleteket világosan kell lepárolni, hogy az új belépők minél gyorsabban
felzárkózhassanak. A kiforrott könyvtáraknak automatizálniuk kell a
gyakori feladatokat, a mintakódnak pedig meg kell könnyítenie, hogy a
gyakorlati felhasználók módosítsák, alkalmazzák és továbbfejlesszék a
szokásos megoldásokat a saját igényeikhez.

Vegyük példának a dinamikus webalkalmazásokat. Bár már az 1990-es években
is számos vállalat, például az Amazon, sikeres adatbázis-alapú
webalkalmazásokat fejlesztett, ennek a technológiának a kreatív
vállalkozókat segítő lehetőségei igazán csak az elmúlt tíz évben
teljesedtek ki, részben az erős és jól dokumentált keretrendszerek
megjelenésének köszönhetően.

A deep learning lehetőségeinek kipróbálása különleges kihívásokat jelent,
mert már egyetlen alkalmazás is számos különböző tudásterületet kapcsol
össze. A deep learning alkalmazásához egyszerre kell érteni
(i) egy probléma adott módon történő megfogalmazásának motivációit;
(ii) egy adott modell matematikai alakját;
(iii) azokat az optimalizálási algoritmusokat, amelyekkel a modelleket az
adatokra illesztjük;
(iv) azokat a statisztikai elveket, amelyek megmondják, mikor várhatjuk,
hogy modelljeink ismeretlen adatokra is általánosítani fognak, valamint
azokat a gyakorlati módszereket, amelyekkel ezt ellenőrizni tudjuk; és
(v) azokat a mérnöki technikákat, amelyek a modellek hatékony betanításához
szükségesek, beleértve a numerikus számítás csapdáinak elkerülését és a
rendelkezésre álló hardver minél jobb kihasználását. Hatalmas kihívás
egyetlen helyen megtanítani a problémák megfogalmazásához szükséges
kritikus gondolkodást, a megoldásukhoz kellő matematikát és a megoldások
megvalósításához szükséges szoftveres eszközöket. Ebben a könyvben az a
célunk, hogy ehhez egy egységes forrást adjunk.

Amikor belekezdtünk ebbe a könyvprojektbe, nem létezett olyan anyag,
amely egyszerre
(i) naprakész maradt volna;
(ii) kellő technikai mélységgel fedte volna le a modern gépi tanulási
gyakorlat széles spektrumát; és
(iii) ötvözte volna a tankönyvektől elvárható minőségű magyarázatot a
gyakorlati útmutatóktól elvárt tiszta, futtatható kóddal.
Rengeteg példakódot találtunk, amelyek megmutatták, hogyan kell egy adott
deep learning keretrendszert használni (például mátrixokkal számolni
TensorFlow-ban), vagy hogyan lehet egy-egy konkrét technikát
megvalósítani (például LeNet, AlexNet, ResNet stb.), szétszórva
blogbejegyzésekben és GitHub-repókban. Ezek a példák azonban jellemzően
arra koncentráltak, *hogyan* kell egy adott megközelítést implementálni,
és nem tárgyalták, *miért* születnek bizonyos algoritmikus döntések.
Időről időre felbukkantak interaktív források is, amelyek egy-egy témát
próbáltak feldolgozni, például a [Distill](http://distill.pub) oldal
igényes cikkei vagy személyes blogok, de ezek csak a deep learning néhány
kiválasztott területét fedték le, és gyakran nem tartalmaztak kapcsolódó
kódot. Másfelől több deep learning tankönyv is megjelent, például
:citet:`Goodfellow.Bengio.Courville.2016`, amely átfogó áttekintést ad a
deep learning alapjairól, de ezek a források nem kötötték össze a
leírásokat a fogalmak kódbeli megvalósításával, így az olvasó gyakran
nem kapott kapaszkodót ahhoz, hogyan valósítsa meg mindezt a gyakorlatban.
Ráadásul túl sok hasznos anyag rejtőzött kereskedelmi kurzusszolgáltatók
fizetős falai mögött.

Olyan forrást akartunk létrehozni, amely
(i) mindenki számára szabadon hozzáférhető;
(ii) elegendő technikai mélységet ad ahhoz, hogy kiindulópont legyen az
alkalmazott gépi tanulási szakértővé válás útján;
(iii) futtatható kódot tartalmaz, és megmutatja az olvasóknak, *hogyan*
oldjanak meg problémákat a gyakorlatban;
(iv) gyorsan frissíthető, részünkről és a közösség részéről egyaránt; és
(v) kiegészül egy [fórummal](https://discuss.d2l.ai/c/5), ahol a technikai
részleteket meg lehet vitatni, és ahol kérdéseket lehet feltenni.

Ezek a célok gyakran ütköztek egymással. Az egyenleteket, tételeket és
hivatkozásokat a legjobb LaTeX-ben kezelni és tördelni. A kódot legjobban
Pythonban lehet bemutatni. A weboldalak természetes közege pedig a HTML
és a JavaScript. Emellett azt szerettük volna, hogy a tartalom egyszerre
legyen elérhető futtatható kódként, nyomtatott könyvként, letölthető
PDF-ként és internetes weboldalként is. Egyetlen meglévő munkafolyamat sem
tűnt alkalmasnak ezekre az igényekre, ezért úgy döntöttünk, sajátot
építünk (:numref:`sec_how_to_contribute`). A forrás megosztására és a
közösségi közreműködés támogatására a GitHubot választottuk; a kód, az
egyenletek és a szöveg együtt kezelésére a Jupyter notebookokat; a
renderelő motornak a Sphinxet; a beszélgetések platformjának pedig a
Discourse-t. Bár a rendszerünk nem tökéletes, ezek a választások ésszerű
kompromisszumot jelentenek a versengő szempontok között. Úgy gondoljuk,
hogy a *Dive into Deep Learning* talán az első olyan könyv, amely ilyen
integrált munkafolyamattal jelent meg.


### Tanulás gyakorlással

Sok tankönyv egymás után vezeti be a fogalmakat, és mindegyiket rendkívül
részletesen tárgyalja. Például :citet:`Bishop.2006` kiváló könyve olyan
alaposan magyaráz el minden témát, hogy már az odáig való eljutás is
komoly munkát igényel, ahol a lineáris regresszióról szóló fejezet
kezdődik. A szakértők éppen ezért szeretik ezt a könyvet, valódi
kezdőknek azonban ez a tulajdonsága korlátozza a használhatóságát mint
bevezető szöveget.

Ebben a könyvben a legtöbb fogalmat *éppen időben* tanítjuk meg. Más
szóval a koncepciókat akkor fogod megtanulni, amikor valamilyen gyakorlati
cél eléréséhez ténylegesen szükség van rájuk. Bár a legelején időt
szánunk az olyan alapokra, mint a lineáris algebra és a valószínűségszámítás,
azt szeretnénk, hogy már azelőtt átélhesd az első modell betanításának
örömét, mielőtt az elvontabb fogalmakkal kezdenénk foglalkozni.

Néhány előkészítő notebookot leszámítva, amelyek gyors áttekintést adnak
az alapvető matematikai háttérről, minden további fejezet egyszerre vezet
be új fogalmakat és mutat be több önállóan futtatható, valós adathalmazokon
alapuló példát. Ez szerkezeti kihívást is jelentett. Egyes modellek
logikailag összetartozhatnának egyetlen notebookban. Néhány ötletet pedig
talán több modell egymás utáni lefuttatásával lehetne a legjobban
megtanítani. Ezzel szemben nagy előnye van annak az elvnek, hogy
*egy működő példa, egy notebook*. Így a lehető legegyszerűbb a saját
kutatási projektjeid elindítása a mi kódunkra építve: egyszerűen másolj
le egy notebookot, és kezdd el módosítani.

A könyv egészében úgy váltogatjuk a futtatható kódot és a háttérmagyarázatot,
ahogy arra szükség van. Általában inkább azt a stratégiát követjük, hogy
egy eszközt már azelőtt használatba adunk, mielőtt teljesen kifejtenénk
(és a háttérmagyarázatot gyakran csak később pótoljuk). Előfordulhat
például, hogy *sztochasztikus gradienscsökkenést* használunk azelőtt, hogy
elmagyaráznánk, miért hasznos, vagy intuíciót adnánk arra, miért működik.
Ez segít abban, hogy a gyakorlati felhasználók gyorsan megkapják a
problémák megoldásához szükséges eszköztárat, cserébe viszont az olvasónak
időnként el kell fogadnia néhány szerkesztői döntésünket.

Ez a könyv a deep learning fogalmait az alapoktól tanítja. Néha olyan
aprólékos részletekbe is belemegyünk modellekkel kapcsolatban, amelyeket a
modern deep learning keretrendszerek általában elrejtenének a
felhasználók elől. Ez főleg az alapozó bemutatókban jelenik meg, ahol azt
szeretnénk, hogy pontosan lásd, mi történik egy adott rétegben vagy
optimalizálóban. Ilyenkor gyakran kétféle változatot mutatunk be:
egyet, ahol mindent a nulláról implementálunk, csupán NumPy-szerű
funkcionalitásra és automatikus differenciálásra támaszkodva; valamint
egy gyakorlatiasabb példát, ahol tömör kódot írunk a deep learning
keretrendszerek magas szintű API-jait használva. Miután elmagyaráztuk,
hogyan működik egy adott komponens, a későbbi bemutatókban már a
magas szintű API-kra támaszkodunk.


### Tartalom és felépítés

A könyv nagyjából három részre osztható: az alapozásra, a deep learning
technikákra, valamint a valós rendszerekre és alkalmazásokra fókuszáló
haladó témákra (:numref:`fig_book_org`).

![A könyv felépítése.](../img/book-org.svg)
:label:`fig_book_org`


* **1. rész: Alapok és előkészületek**.
:numref:`chap_introduction` bevezetést ad a deep learning világába.
Ezután a :numref:`chap_preliminaries` gyorsan áttekinti azokat az
előfeltételeket, amelyek a gyakorlati deep learninghez szükségesek:
hogyan tároljuk és kezeljük az adatokat, és hogyan alkalmazzunk különféle
numerikus műveleteket a lineáris algebra, a kalkulus és a
valószínűségszámítás elemi fogalmaira építve.
:numref:`chap_regression` és :numref:`chap_perceptrons` a deep learning
legalapvetőbb fogalmait és technikáit tárgyalja, beleértve a regressziót
és az osztályozást, a lineáris modelleket, a többrétegű perceptronokat,
valamint a túlillesztést és a regularizációt.

* **2. rész: Modern deep learning technikák**.
:numref:`chap_computation` bemutatja a deep learning rendszerek alapvető
számítási komponenseit, és megteremti az alapot a későbbi, összetettebb
modellek implementálásához. Ezután :numref:`chap_cnn` és
:numref:`chap_modern_cnn` a konvolúciós neurális hálózatokat (CNN-eket)
ismerteti, amelyek a legtöbb modern számítógépes látórendszer gerincét
adják. Hasonlóképpen :numref:`chap_rnn` és :numref:`chap_modern_rnn` a
rekurzív neurális hálózatokat (RNN-eket) vezeti be, vagyis azokat a
modelleket, amelyek kihasználják az adatok szekvenciális
(például időbeli) szerkezetét, és amelyeket gyakran használnak
természetesnyelv-feldolgozásban és idősor-előrejelzésben.
:numref:`chap_attention-and-transformers` egy viszonylag új modellosztályt
mutat be, az úgynevezett *figyelmi mechanizmusokra* épülő modelleket,
amelyek a legtöbb természetesnyelv-feldolgozási feladatban kiszorították
az RNN-eket, mint uralkodó architektúrát. Ezek a fejezetek felzárkóztatnak
a deep learning szakemberek által széles körben használt legerősebb és
legáltalánosabb eszközökhöz.

* **3. rész: Skálázhatóság, hatékonyság és alkalmazások** (elérhető
[online](https://d2l.ai)).
A 12. fejezetben több gyakori optimalizálási algoritmust tárgyalunk,
amelyeket deep learning modellek tanítására használnak. Ezután a 13.
fejezetben több kulcsfontosságú tényezőt vizsgálunk meg, amelyek
befolyásolják a deep learning kód futási teljesítményét. A 14.
fejezetben a deep learning legfontosabb számítógépes látási
alkalmazásait mutatjuk be. Végül a 15. és 16. fejezetben megmutatjuk,
hogyan lehet nyelvi reprezentációs modelleket előtanítani, majd ezeket
természetesnyelv-feldolgozási feladatokra alkalmazni.


### Kód
:label:`sec_code`

A könyv legtöbb fejezete futtatható kódot is tartalmaz. Úgy gondoljuk,
hogy bizonyos intuíciókat a legjobban próbálgatással lehet kialakítani:
kis mértékben módosítod a kódot, majd megfigyeled az eredményt.
Ideális esetben egy elegáns matematikai elmélet pontosan megmondaná,
hogyan kell a kódot módosítani egy kívánt eredmény eléréséhez. A mai
deep learning szakembereknek azonban gyakran olyan területen kell
haladniuk, ahol még nincs szilárd elméleti útmutatás. Legjobb
erőfeszítéseink ellenére sok technika hatékonyságára még mindig nincs
teljes formális magyarázat, több okból is: e modellek matematikai
jellemzése rendkívül nehéz lehet; a magyarázat valószínűleg az adatok
olyan tulajdonságaitól függ, amelyeknek jelenleg sincs egyértelmű
definíciójuk; és az ilyen kérdések komoly vizsgálata csak az utóbbi
években gyorsult fel igazán. Reméljük, hogy ahogy a deep learning
elmélete fejlődik, a könyv jövőbeli kiadásai még a mostaninál is mélyebb
felismeréseket tudnak majd nyújtani.

Az indokolatlan ismétlés elkerülése érdekében a leggyakrabban importált
és használt függvényeink, illetve osztályaink egy részét a `d2l`
csomagban gyűjtjük össze. A könyvben a kódblokkokat
(például függvényeket, osztályokat vagy importutasítás-gyűjteményeket)
`#@save` megjegyzéssel jelöljük, jelezve, hogy később a `d2l` csomagon
keresztül fogunk rájuk hivatkozni. Ezekről az osztályokról és
függvényekről részletes áttekintést adunk :numref:`sec_d2l` részben. A
`d2l` csomag könnyűsúlyú, és csak az alábbi függőségeket igényli:

```{.python .input}
#@tab all
#@save
import inspect
import collections
from collections import defaultdict
from IPython import display
import math
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
import os
import pandas as pd
import random
import re
import shutil
import sys
import tarfile
import time
import requests
import zipfile
import hashlib
d2l = sys.modules[__name__]
```

:begin_tab:`mxnet`
A könyv kódjának nagy része Apache MXNetre épül, amely egy nyílt
forráskódú deep learning keretrendszer, és amelyet az AWS (Amazon Web
Services), valamint számos egyetem és vállalat is előnyben részesít. A
könyv összes kódja átment a legfrissebb MXNet-verzión végzett
teszteken. Ugyanakkor a deep learning gyors fejlődése miatt előfordulhat,
hogy a *nyomtatott kiadásban* szereplő egyes kódok a jövőbeli
MXNet-verziókban már nem működnek megfelelően. Az online változatot
igyekszünk naprakészen tartani. Ha problémába ütközöl, nézd meg a :ref:`chap_installation` részt a kód és a futtatási környezet
frissítéséhez. Az alábbiakban az MXNet-implementációnk függőségeit
soroljuk fel.
:end_tab:

:begin_tab:`pytorch`
A könyv kódjának nagy része PyTorchra épül, amely egy népszerű nyílt
forráskódú keretrendszer, és amelyet a deep learning kutatói közösség
nagy lelkesedéssel fogadott. A könyv összes kódja átment a PyTorch
legfrissebb stabil verzióján végzett teszteken. Ugyanakkor a deep
learning gyors fejlődése miatt előfordulhat, hogy a *nyomtatott
kiadásban* szereplő egyes kódok a jövőbeli PyTorch-verziókban már nem
működnek megfelelően. Az online változatot igyekszünk naprakészen
tartani. Ha problémába ütközöl, nézd meg a :ref:`chap_installation` részt a kód és a futtatási környezet
frissítéséhez. Az alábbiakban a PyTorch-implementációnk függőségeit
soroljuk fel.
:end_tab:

:begin_tab:`tensorflow`
A könyv kódjának nagy része TensorFlow-ra épül, amely egy nyílt
forráskódú deep learning keretrendszer, széles körben használják az
iparban, és népszerű a kutatók körében is. A könyv összes kódja átment a
TensorFlow legfrissebb stabil verzióján végzett teszteken. Ugyanakkor a
deep learning gyors fejlődése miatt előfordulhat, hogy a *nyomtatott
kiadásban* szereplő egyes kódok a jövőbeli TensorFlow-verziókban már nem
működnek megfelelően. Az online változatot igyekszünk naprakészen
tartani. Ha problémába ütközöl, nézd meg a :ref:`chap_installation` részt a kód és a futtatási környezet
frissítéséhez. Az alábbiakban a TensorFlow-implementációnk függőségeit
soroljuk fel.
:end_tab:

:begin_tab:`jax`
A könyv kódjának nagy része JAX-re épül, amely egy nyílt forráskódú
keretrendszer, és összefűzhető függvénytranszformációkat tesz lehetővé,
például tetszőleges Python- és NumPy-függvények deriválását, valamint
JIT-fordítást, vektorizálást és még sok minden mást. Egyre népszerűbb a
gépi tanulási kutatásban, és könnyen megtanulható, NumPy-szerű API-val
rendelkezik. Sőt, a JAX célja gyakorlatilag az 1:1 megfelelés a
NumPy-val, így a kódod átállítása akár annyiból is állhat, hogy egyetlen
importutasítást módosítasz. Ugyanakkor a deep learning gyors fejlődése
miatt előfordulhat, hogy a *nyomtatott kiadásban* szereplő egyes kódok a
jövőbeli JAX-verziókban már nem működnek megfelelően. Az online
változatot igyekszünk naprakészen tartani. Ha problémába ütközöl,
nézd meg a :ref:`chap_installation` részt a kód és a futtatási
környezet frissítéséhez. Az alábbiakban a JAX-implementációnk
függőségeit soroljuk fel.
:end_tab:

```{.python .input}
#@tab mxnet
#@save
from mxnet import autograd, context, gluon, image, init, np, npx
from mxnet.gluon import nn, rnn
```

```{.python .input}
#@tab pytorch
#@save
import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from scipy.spatial import distance_matrix
```

```{.python .input}
#@tab tensorflow
#@save
import numpy as np
import tensorflow as tf
```

```{.python .input}
#@tab jax
#@save
from dataclasses import field
from functools import partial
import flax
from flax import linen as nn
from flax.training import train_state
import jax
from jax import numpy as jnp
from jax import grad, vmap
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from types import FunctionType
from typing import Any
```

### Célközönség

Ez a könyv azoknak a hallgatóknak (alap- és mesterszakosoknak),
mérnököknek és kutatóknak szól, akik biztos gyakorlati tudást szeretnének
szerezni a deep learning technikáiból. Mivel minden fogalmat az alapoktól
magyarázunk el, nincs szükség korábbi deep learning vagy gépi tanulási
háttérre. A deep learning módszereinek teljes megértéséhez szükség van
némi matematikára és programozási tudásra, de csak alapvető előismereteket
feltételezünk: némi lineáris algebra, kalkulus, valószínűségszámítás és
Python-programozás elegendő. Ha valamit elfelejtettél volna, az
[online függelék](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/index.html)
felfrissíti a könyvben előforduló matematikai anyag nagy részét.
Általában az intuíciót és az ötleteket részesítjük előnyben a szigorú
matematikai formalizmussal szemben. Ha pedig az itt feltételezett
alapismereteken túl szeretnéd tovább mélyíteni ezeket az alapokat,
örömmel ajánlunk néhány kiváló további forrást: :citet:`Bollobas.1999`
*Linear Analysis* című könyve nagy mélységben tárgyalja a lineáris
algebrát és a funkcionálanalízist. *All of Statistics*
:cite:`Wasserman.2013` csodálatos bevezetést nyújt a statisztikába.
Joe Blitzstein [könyvei](https://www.amazon.com/Introduction-Probability-Chapman-Statistical-Science/dp/1138369918)
és [kurzusai](https://projects.iq.harvard.edu/stat110/home) a
valószínűségszámításról és következtetésről valóságos pedagógiai
gyöngyszemek. És ha még nem használtál korábban Pythont, érdemes lehet
átfutnod ezt a [Python-oktatóanyagot](http://learnpython.org/).


### Notebookok, weboldal, GitHub és fórum

Az összes notebook letölthető a [D2L.ai weboldaláról](https://d2l.ai) és
a [GitHubról](https://github.com/d2l-ai/d2l-en) is. A könyvhöz kapcsolódóan
elindítottunk egy vitafórumot a
[discuss.d2l.ai](https://discuss.d2l.ai/c/5) oldalon. Ha kérdésed van a
könyv bármely részével kapcsolatban, minden notebook végén találsz linket
az adott részhez tartozó beszélgetéshez.



## Köszönetnyilvánítás

Hálával tartozunk az angol és a kínai változat több száz közreműködőjének.
Ők segítettek javítani a tartalmat, és nagyon értékes visszajelzéseket
adtak. Ezt a könyvet eredetileg úgy készítettük el, hogy az MXNet volt
az elsődleges keretrendszer. Köszönjük Anirudh Dagarnak és Yuan Tangnak,
hogy a korábbi
MXNet-kód nagy részét PyTorch-, illetve TensorFlow-implementációkra
alakították át. 2021 júliusa óta a könyvet újraterveztük és újra is
implementáltuk PyTorchban, MXNetben és TensorFlow-ban, a PyTorchot
választva elsődleges keretrendszernek. Köszönjük Anirudh Dagarnak, hogy a
frissebb PyTorch-kód jelentős részét JAX-implementációkra ültette át.
Köszönjük a Baidu munkatársainak, Gaosheng Wunak, Liujun Hunak, Ge
Zhangnak és Jiehang Xienek, hogy a frissebb PyTorch-kód jelentős részét
PaddlePaddle-implementációkká alakították a kínai változatban. Köszönjük
Shuai Zhangnak, hogy a kiadó LaTeX stílusát integrálta a PDF-buildbe.

A GitHubon köszönjük ennek az angol változatnak minden közreműködőjének,
hogy mindenki számára jobbá tették a könyvet. A GitHub-azonosítóik vagy
neveik a következők (különösebb sorrend nélkül):
alxnorden, avinashingit, bowen0701, brettkoonce, Chaitanya Prakash Bapat,
cryptonaut, Davide Fiocco, edgarroman, gkutiel, John Mitro, Liang Pu,
Rahul Agarwal, Mohamed Ali Jamaoui, Michael (Stu) Stewart, Mike Müller,
NRauschmayr, Prakhar Srivastav, sad-, sfermigier, Sheng Zha, sundeepteki,
topecongiro, tpdi, vermicelli, Vishaal Kapoor, Vishwesh Ravi Shrimali, YaYaB, Yuhong Chen,
Evgeniy Smirnov, lgov, Simon Corston-Oliver, Igor Dzreyev, Ha Nguyen, pmuens,
Andrei Lukovenko, senorcinco, vfdev-5, dsweet, Mohammad Mahdi Rahimi, Abhishek Gupta,
uwsd, DomKM, Lisa Oakley, Bowen Li, Aarush Ahuja, Prasanth Buddareddygari, brianhendee,
mani2106, mtn, lkevinzc, caojilin, Lakshya, Fiete Lüer, Surbhi Vijayvargeeya,
Muhyun Kim, dennismalmgren, adursun, Anirudh Dagar, liqingnz, Pedro Larroy,
lgov, ati-ozgur, Jun Wu, Matthias Blume, Lin Yuan, geogunow, Josh Gardner,
Maximilian Böther, Rakib Islam, Leonard Lausen, Abhinav Upadhyay, rongruosong,
Steve Sedlmeyer, Ruslan Baratov, Rafael Schlatter, liusy182, Giannis Pappas,
ati-ozgur, qbaza, dchoi77, Adam Gerson, Phuc Le, Mark Atwood, christabella, vn09,
Haibin Lin, jjangga0214, RichyChen, noelo, hansent, Giel Dops, dvincent1337, WhiteD3vil,
Peter Kulits, codypenta, joseppinilla, ahmaurya, karolszk, heytitle, Peter Goetz, rigtorp,
Tiep Vu, sfilip, mlxd, Kale-ab Tessera, Sanjar Adilov, MatteoFerrara, hsneto,
Katarzyna Biesialska, Gregory Bruss, Duy–Thanh Doan, paulaurel, graytowne, Duc Pham,
sl7423, Jaedong Hwang, Yida Wang, cys4, clhm, Jean Kaddour, austinmw, trebeljahr, tbaums,
Cuong V. Nguyen, pavelkomarov, vzlamal, NotAnotherSystem, J-Arun-Mani, jancio, eldarkurtic,
the-great-shazbot, doctorcolossus, gducharme, cclauss, Daniel-Mietchen, hoonose, biagiom,
abhinavsp0730, jonathanhrandall, ysraell, Nodar Okroshiashvili, UgurKap, Jiyang Kang,
StevenJokes, Tomer Kaftan, liweiwp, netyster, ypandya, NishantTharani, heiligerl, SportsTHU,
Hoa Nguyen, manuel-arno-korfmann-webentwicklung, aterzis-personal, nxby, Xiaoting He, Josiah Yoder,
mathresearch, mzz2017, jroberayalas, iluu, ghejc, BSharmi, vkramdev, simonwardjones, LakshKD,
TalNeoran, djliden, Nikhil95, Oren Barkan, guoweis, haozhu233, pratikhack, Yue Ying, tayfununal,
steinsag, charleybeller, Andrew Lumsdaine, Jiekui Zhang, Deepak Pathak, Florian Donhauser, Tim Gates,
Adriaan Tijsseling, Ron Medina, Gaurav Saha, Murat Semerci, Lei Mao, Levi McClenny, Joshua Broyde,
jake221, jonbally, zyhazwraith, Brian Pulfer, Nick Tomasino, Lefan Zhang, Hongshen Yang, Vinney Cavallo,
yuntai, Yuanxiang Zhu, amarazov, pasricha, Ben Greenawald, Shivam Upadhyay, Quanshangze Du, Biswajit Sahoo,
Parthe Pandit, Ishan Kumar, HomunculusK, Lane Schwartz, varadgunjal, Jason Wiener, Armin Gholampoor,
Shreshtha13, eigen-arnav, Hyeonggyu Kim, EmilyOng, Bálint Mucsányi, Chase DuBois, Juntian Tao,
Wenxiang Xu, Lifu Huang, filevich, quake2005, nils-werner, Yiming Li, Marsel Khisamutdinov,
Francesco "Fuma" Fumagalli, Peilin Sun, Vincent Gurgul, qingfengtommy, Janmey Shukla, Mo Shan,
Kaan Sancak, regob, AlexSauer, Gopalakrishna Ramachandra, Tobias Uelwer, Chao Wang, Tian Cao,
Nicolas Corthorn, akash5474, kxxt, zxydi1992, Jacob Britton, Shuangchi He, zhmou, krahets, Jie-Han Chen,
Atishay Garg, Marcel Flygare, adtygan, Nik Vaessen, bolded, Louis Schlessinger, Balaji Varatharajan,
atgctg, Kaixin Li, Victor Barbaros, Riccardo Musto, Elizabeth Ho, azimjonn, Guilherme Miotto, Alessandro Finamore,
Joji Joseph, Anthony Biel, Zeming Zhao, shjustinbaek, gab-chen, nantekoto, Yutaro Nishiyama, Oren Amsalem,
Tian-MaoMao, Amin Allahyar, Gijs van Tulder, Mikhail Berkov, iamorphen, Matthew Caseres, Andrew Walsh,
pggPL, RohanKarthikeyan, Ryan Choi és Likun Lei.

Köszönjük az Amazon Web Servicesnek, különösen Wen-Ming Ye-nek, George
Karypisnak, Swami Sivasubramaniannak, Peter DeSantisnak, Adam Selipskynek
és Andrew Jassynak a könyv megírásához nyújtott nagylelkű támogatást.
Az ehhez szükséges idő, erőforrások, kollégákkal folytatott beszélgetések
és folyamatos bátorítás nélkül ez a könyv nem születhetett volna meg. A
könyv kiadásra való előkészítése során a Cambridge University Press
kiváló támogatást nyújtott. Köszönjük felkérő szerkesztőnknek, David
Tranahnak a segítségét és professzionalizmusát.


## Összefoglalás

A deep learning forradalmasította a mintafelismerést, és olyan
technológiát hozott létre, amely ma már számos megoldás alapját adja
olyan különböző területeken, mint a számítógépes látás, a
természetesnyelv-feldolgozás és az automatikus beszédfelismerés. A deep
learning sikeres alkalmazásához meg kell értened, hogyan fogalmazz meg
egy problémát, milyen alapvető matematikára épül a modellezés, milyen
algoritmusokkal lehet a modelleket adatokra illeszteni, és milyen
mérnöki technikák szükségesek mindennek a megvalósításához. Ez a könyv
átfogó forrást kínál, amely egy helyen egyesíti a magyarázó szöveget, az
ábrákat, a matematikát és a kódot.



## Feladatok

1. Regisztrálj fiókot a könyv vitafórumán:
   [discuss.d2l.ai](https://discuss.d2l.ai/).
1. Telepítsd a Pythont a számítógépedre.
1. Kövesd a fejezet alján található linkeket a fórumra, ahol segítséget
   kérhetsz, megbeszélheted a könyv anyagát, és az írók, valamint a
   tágabb közösség bevonásával választ találhatsz a kérdéseidre.

:begin_tab:`mxnet`
[Beszélgetések](https://discuss.d2l.ai/t/18)
:end_tab:

:begin_tab:`pytorch`
[Beszélgetések](https://discuss.d2l.ai/t/20)
:end_tab:

:begin_tab:`tensorflow`
[Beszélgetések](https://discuss.d2l.ai/t/186)
:end_tab:

:begin_tab:`jax`
[Beszélgetések](https://discuss.d2l.ai/t/17963)
:end_tab:
