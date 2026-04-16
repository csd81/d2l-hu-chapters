# Hardver
:label:`sec_hardware`

A kiváló teljesítményű rendszerek felépítéséhez jó ismeretek szükségesek az algoritmusokról és modellekről, hogy a probléma statisztikai aspektusait is figyelembe lehessen venni. Ugyanakkor elengedhetetlen az alapul szolgáló hardver legalább alap szintű ismerete is. Ez a fejezet nem helyettesít egy teljes hardver- és rendszertervezési kurzust. Inkább kiindulópontként szolgálhat annak megértéséhez, hogy egyes algoritmusok miért hatékonyabbak másoknál, és hogyan érhető el jó áteresztőképesség. Egy jó tervezés könnyen egy nagyságrendnyi különbséget hozhat, ami eldöntheti, hogy egy hálózat sikeresen betanítható-e (például egy héten belül) vagy sem (3 hónapig tartana, és lemaradunk a határidőről).
Először a számítógépeket tekintjük át. Majd rázoomolunk a CPU-kra és GPU-kra. Végül visszazoomolunk, hogy megnézzük, hogyan kapcsolódnak össze a számítógépek egy szerverközpontban vagy a felhőben.

![Késleltetési számok, amelyeket minden programozónak ismernie kell.](../img/latencynumbers.png)
:label:`fig_latencynumbers`

A türelmetlen olvasók megelégedhetnek a :numref:`fig_latencynumbers` ábra áttekintésével. Ez Colin Scott [interaktív bejegyzéséből](https://people.eecs.berkeley.edu/%7Ercs/research/interactive_latency.html) származik, amely jó áttekintést nyújt az elmúlt évtized fejlődéséről. Az eredeti számok Jeff Dean [2010-es Stanford-előadásából](https://static.googleusercontent.com/media/research.google.com/en//people/jeff/Stanford-DL-Nov-2010.pdf) valók.
Az alábbi tárgyalás magyarázatot ad ezekre a számokra, és megmutatja, hogyan segíthetnek az algoritmusok tervezésében. Az alábbi tárgyalás magas szintű és vázlatos. Ez egyértelműen *nem helyettesíti* a megfelelő kurzust, hanem csupán elegendő információt kíván nyújtani ahhoz, hogy egy statisztikai modellező megfelelő tervezési döntéseket hozzon. A számítógép-architektúra mélyebb áttekintéséhez :cite:`Hennessy.Patterson.2011`-t ajánljuk, vagy egy közelmúltbeli kurzust erről a témáról, mint például az [Arste Asanovic](http://inst.eecs.berkeley.edu/%7Ecs152/sp19/) által tartott.

## Számítógépek

A legtöbb deep learning kutató és szakember hozzáfér egy számítógéphez, amely megfelelő mennyiségű memóriával, számítási kapacitással és valamilyen gyorsítóval (például GPU-val) rendelkezik, esetleg több ilyen eszközzel. Egy számítógép a következő fő összetevőkből áll:

* Egy processzor (CPU), amely képes végrehajtani a programjainkat (az operációs rendszer és sok egyéb dolog futtatása mellett), általában 8 vagy több maggal.
* Memória (RAM) a számítási eredmények (pl. súlyvektorok, aktivációk) és a tanítási adatok tárolásához és visszakereséséhez.
* Ethernet hálózati kapcsolat (néha több is), amelynek sebessége 1 GB/s és 100 GB/s között mozog. Csúcskategóriás szervereken fejlettebb összeköttetések is találhatók.
* Nagy sebességű bővítőbusz (PCIe) az egy vagy több GPU csatlakoztatásához. A szervereknek akár 8 gyorsítójuk is lehet, amelyek sokszor fejlett topológiában vannak összekötve, míg az asztali rendszereknél 1 vagy 2 db van, a felhasználó költségvetésétől és a tápegység méretétől függően.
* Tartós tárolók, mint a mágneses merevlemez (HDD) vagy a SSD, amelyek sok esetben PCIe buszon keresztül csatlakoznak. Ezek biztosítják a tanítási adatok hatékony átvitelét és a közbenső ellenőrzési pontok tárolását.

![Egy számítógép összetevőinek kapcsolata.](../img/mobo-symbol.svg)
:label:`fig_mobo-symbol`

Ahogy a :numref:`fig_mobo-symbol` ábra mutatja, a legtöbb összetevő (hálózat, GPU, tárolók) PCIe buszon keresztül kapcsolódik a CPU-hoz. Ez több, közvetlenül a CPU-hoz csatlakozó sávból áll. Például az AMD Threadripper 3 64 darab PCIe 4.0 sávval rendelkezik, amelyek mindegyike mindkét irányban 16 Gbit/s adatátvitelre képes. A memória közvetlenül a CPU-hoz csatlakozik, összesített sávszélességgel akár 100 GB/s értékig.

Amikor kódot futtatunk egy számítógépen, adatokat kell mozgatni a processzorokhoz (CPU-k vagy GPU-k), elvégezni a számítást, majd az eredményeket visszamozgatni a RAM-ba és a tartós tárolóra. Ezért a jó teljesítmény érdekében biztosítani kell, hogy ez zökkenőmentesen működjön, anélkül hogy bármelyik rendszer komoly szűk keresztmetszetté válna. Például, ha nem tudjuk elég gyorsan betölteni a képeket, a processzornak nem lesz feladata. Hasonlóképpen, ha nem tudunk elég gyorsan mátrixokat a CPU-ra (vagy GPU-ra) mozgatni, annak feldolgozóegységei éhesek maradnak. Végül, ha több számítógépet kell szinkronizálni a hálózaton keresztül, ez ne lassítsa le a számítást. Az egyik lehetőség a kommunikáció és a számítás összefonása. Nézzük meg részletesebben az egyes összetevőket.


## Memória

Legegyszerűbb formájában a memória a gyorsan elérhető adatok tárolására szolgál. Jelenleg a CPU RAM általában [DDR4](https://en.wikipedia.org/wiki/DDR4_SDRAM) típusú, modulonként 20--25 GB/s sávszélességgel. Minden modul 64 bites buszsal rendelkezik. Jellemzően memóriamodulpárokat használnak a többcsatornás működés lehetővé tételéhez. A CPU-k 2 és 4 memóriacsatorna között rendelkeznek, azaz 40 GB/s és 100 GB/s közötti csúcsmemória-sávszélességgel. Csatornánként általában két memóriabank van. Például az AMD Zen 3 Threadripper 8 foglalattal rendelkezik.

Bár ezek a számok lenyűgözőek, csak részben árulják el a teljes képet. Amikor a memória egy részét szeretnénk olvasni, először meg kell mondanunk a memóriamodulnak, hogy hol találja az információt. Azaz először el kell küldeni a *címet* a RAM-nak. Miután ez megtörtént, dönthetünk úgy, hogy csak egyetlen 64 bites rekordot, vagy egy hosszú sorozatot olvasunk. Az utóbbit *burst olvasásnak* nevezik. Röviden: a cím memóriába küldése és az átvitel beállítása körülbelül 100 ns-t vesz igénybe (a részletek a használt memóriachipek konkrét időzítési együtthatóitól függnek), minden egyes rákövetkező átvitel mindössze 0,2 ns-t vesz igénybe. Röviden: az első olvasás 500-szor drágább, mint a következők! Vegyük figyelembe, hogy másodpercenként akár 10 000 000 véletlen olvasást is elvégezhetünk. Ez azt sugallja, hogy amennyire lehetséges, kerüljük a véletlen memória-hozzáférést, és helyette burst olvasást (és írást) alkalmazzunk.

A dolgok egy kicsit bonyolultabbak, ha figyelembe vesszük, hogy több *bank* is van. Minden bank nagyrészt függetlenül képes olvasni a memóriából. Ez két dolgot jelent.
Egyrészt a véletlen olvasások effektív száma akár 4-szer magasabb, feltéve, hogy egyenletesen oszlanak el a memória között. Ez azt is jelenti, hogy véletlen olvasások végzése még mindig rossz ötlet, mivel a burst olvasások 4-szer gyorsabbak is. Másrészt, a 64 bites határokhoz való memóriaigazítás miatt érdemes az adatstruktúrákat ugyanezekhez a határokhoz igazítani. A fordítók ezt nagyrészt [automatikusan](https://en.wikipedia.org/wiki/Data_structure_alignment) elvégzik, ha a megfelelő jelzők be vannak állítva. A kíváncsi olvasókat bátorítjuk, hogy tekintsék át a DRAM-okról szóló előadásokat, például a [Zeshan Chishti](http://web.cecs.pdx.edu/%7Ezeshan/ece585_lec5.pdf) által tartott előadást.

A GPU memóriára még magasabb sávszélesség-követelmények vonatkoznak, mivel sokkal több feldolgozóelemük van, mint a CPU-knak. Általánosságban két lehetőség van ennek kezelésére. Az első a memóriabusz jelentős kiszélesítése. Például az NVIDIA RTX 2080 Ti 352 bites buszszal rendelkezik. Ez lehetővé teszi, hogy egyszerre sokkal több információt lehessen továbbítani. Másodszor, a GPU-k specifikus, nagy teljesítményű memóriát használnak. A fogyasztói szintű eszközök, mint az NVIDIA RTX és Titan sorozat, általában [GDDR6](https://en.wikipedia.org/wiki/GDDR6_SDRAM) chipeket használnak, összesített sávszélességük meghaladja az 500 GB/s-t. Alternatíva a HBM (nagy sávszélességű memória) modulok használata. Ezek egy teljesen eltérő interfészt alkalmaznak, és közvetlenül a GPU-khoz csatlakoznak egy dedikált szilíciumlapkán. Ez drágává teszi őket, és általában csak csúcskategóriás szerver chipekre korlátozódik a használatuk, mint például az NVIDIA Volta V100 sorozat gyorsítói. Nem meglepő módon a GPU memória általában *sokkal* kisebb a CPU memóriánál az előbbi magasabb ára miatt. Céljaink szempontjából teljesítményi jellemzőik hasonlók, csak jóval gyorsabbak. E könyv céljából biztonságosan figyelmen kívül hagyhatjuk a részleteket. Csak akkor van jelentőségük, ha GPU kerneleket hangolunk a nagy áteresztőképesség érdekében.

## Tárolás

Láttuk, hogy a RAM néhány legfontosabb jellemzője a *sávszélesség* és a *késleltetés*. Ugyanez igaz a tárolóeszközökre is, csak itt a különbségek még szélsőségesebbek lehetnek.

### Merevlemez meghajtók

A *merevlemez meghajtók* (HDD-k) több mint fél évszázada vannak használatban. Röviden, számos forgó lapot tartalmaznak, amelyeken fejek mozognak, és tetszőleges sávon olvashatnak vagy írhatnak. A csúcskategóriás lemezek akár 9 lapon is 16 TB-ot tudnak tárolni. A HDD-k egyik fő előnye, hogy viszonylag olcsók. Számos hátrányuk közé tartozik a katasztrofális meghibásodási mód és a viszonylag magas olvasási késleltetés.

Az utóbbi megértéséhez vegyük figyelembe, hogy a HDD-k körülbelül 7200 RPM (fordulatszám percenként) sebességgel forognak. Ha gyorsabbak lennének, a centrifugális erő szétszakítaná a lapokat. Ennek komoly hátránya van a lemez egy adott szektorának elérésekor: meg kell várnunk, amíg a lap a megfelelő helyzetbe forog (a fejeket mozgathatjuk, de magát a lemezt nem gyorsíthatjuk). Ezért a kért adatok eléréséhez akár 8 ms-t is várni kell. Ezt általában úgy fejezik ki, hogy a HDD-k körülbelül 100 IOP/s (bemeneti/kimeneti műveletek másodpercenként) sebességgel képesek működni. Ez a szám lényegében változatlan maradt az elmúlt két évtizedben. Ráadásul a sávszélesség növelése is nehéz (100--200 MB/s nagyságrendű). Elvégre minden fej bitsorozatot olvas egy sávból, így a bitsebesség csak az információsűrűség négyzetgyökével arányosan növekszik. Ennek eredményeként a HDD-k gyorsan archiválási tárolóvá és nagyon nagy adatkészletek alacsony szintű tárolójává válnak.


### Szilárdtest-meghajtók

A szilárdtest-meghajtók (SSD-k) flash memóriát használnak az információk tartós tárolásához. Ez *sokkal gyorsabb* hozzáférést tesz lehetővé a tárolt rekordokhoz. A modern SSD-k 100 000 és 500 000 IOP/s között képesek működni, azaz akár 3 nagyságrenddel gyorsabbak a HDD-knél. Sávszélességük elérheti az 1--3 GB/s-t, azaz egy nagyságrenddel gyorsabb a HDD-knél. Ezek a fejlesztések szinte túl szépnek tűnnek ahhoz, hogy igazak legyenek. Valójában az SSD-k tervezési módjából adódóan a következő korlátozásokkal járnak.

* Az SSD-k blokkokban tárolják az információt (256 KB vagy nagyobb). Ezek csak egészként írhatók, ami jelentős időt vesz igénybe. Következésképpen a bitenkénti véletlen írások SSD-n nagyon gyenge teljesítményt nyújtanak. Hasonlóképpen, az adatok írása általában jelentős időt vesz igénybe, mivel a blokkot be kell olvasni, törölni, majd új információval újra kell írni. Az SSD-vezérlők és firmware-ek mára algoritmusokat dolgoztak ki ennek mérséklésére. Ennek ellenére az írás sokkal lassabb lehet, különösen a QLC (quad level cell, négyszintű cella) SSD-k esetén. A jobb teljesítmény kulcsa az, hogy *sorban* tartsuk a műveleteket, részesítsük előnyben az olvasásokat, és ha lehetséges, nagy blokkokban írjunk.
* Az SSD-k memóriacellái viszonylag gyorsan elhasználódnak (gyakran már néhány ezer írás után). A kopásszint-védő algoritmusok képesek a degradációt sok cella között elosztani. Ennek ellenére nem ajánlott SSD-ket lapozófájlokhoz vagy nagy mennyiségű naplófájl-összesítéshez használni.
* Végül a sávszélesség masszív növekedése arra kényszerítette a számítógép-tervezőket, hogy az SSD-ket közvetlenül a PCIe buszra csatlakoztassák. Az erre képes meghajtókat NVMe-nek (Non Volatile Memory enhanced) nevezik, és akár 4 PCIe sávot is használhatnak. Ez PCIe 4.0-n akár 8 GB/s-t is jelent.

### Felhőalapú tárolás

A felhőalapú tárolás konfigurálható teljesítménytartományt kínál. Azaz a tárolók virtuális gépekhez való hozzárendelése dinamikus, mind mennyiség, mind sebesség tekintetében, a felhasználók választása szerint. Ajánlott növelni a kiosztott IOP-ok számát, ha a késleltetés túl magas, például sok kis rekordos tanítás során.

## CPU-k

A központi feldolgozóegységek (CPU-k) minden számítógép középpontjai. Számos kulcsfontosságú összetevőből állnak: *processzormagok*, amelyek képesek gépi kód végrehajtására, *busz*, amely összeköti őket (a konkrét topológia jelentősen eltér a processzorok modelljei, generációi és gyártói között), és *gyorsítótárak*, amelyek lehetővé teszik a magasabb sávszélességű és alacsonyabb késleltetésű memória-hozzáférést, mint ami a fő memóriából való olvasással lehetséges. Végül szinte minden modern CPU tartalmaz *vektor feldolgozóegységeket* a nagy teljesítményű lineáris algebra és konvolúciók segítésére, amelyek gyakoriak a médiafeldolgozásban és a gépi tanulásban.

![Intel Skylake fogyasztói négymag CPU.](../img/skylake.svg)
:label:`fig_skylake`

A :numref:`fig_skylake` ábra egy Intel Skylake fogyasztói szintű négymagos CPU-t mutat be. Integrált GPU-val, gyorsítótárakkal és egy gyűrűbuszszal rendelkezik, amely összeköti a négy magot. A perifériák, mint az Ethernet, WiFi, Bluetooth, SSD vezérlő és USB, vagy a chipset részei, vagy közvetlenül (PCIe-n) a CPU-hoz csatlakoznak.


### Mikroarchitektúra

Minden processzormag meglehetősen kifinomult összetevőkből áll. Bár a részletek generációk és gyártók között eltérnek, az alapfunkcionalitás nagyjából szabványos. Az előlap betölti az utasításokat, és megpróbálja előre jelezni, melyik útvonalat veszik (például vezérlési folyamat esetén). Az utasításokat ezután assembly kódból mikroutasításokká dekódolják. Az assembly kód gyakran nem a legalacsonyabb szintű kód, amelyet a processzor végrehajt. Ehelyett az összetett utasítások alacsonyabb szintű műveletek halmazára bonthatók. Ezeket aztán a tényleges végrehajtó mag dolgozza fel. Utóbbi gyakran képes egyszerre sok műveletet elvégezni. Például a :numref:`fig_cortexa77` ARM Cortex A77 magja egyszerre akár 8 műveletet is képes elvégezni.

![ARM Cortex A77 mikroarchitektúra.](../img/a77.svg)
:label:`fig_cortexa77`

Ez azt jelenti, hogy a hatékony programok órajelciklusonként egynél több utasítást is képesek végrehajtani, feltéve, hogy azok egymástól függetlenül hajthatók végre. Nem minden egység egyforma. Néhány egész számokra specializálódott, míg mások a lebegőpontos teljesítményre vannak optimalizálva. Az áteresztőképesség növelése érdekében a processzor egy elágazó utasítás esetén egyszerre több kódútvonalat is követhet, majd elveti a nem követett ágak eredményeit. Ezért fontosak az ágbecslő egységek (az előlapon), hogy csak a legígéretesebb útvonalakat kövessék.

### Vektorizáció

A deep learning rendkívül számításigényes. Ezért ahhoz, hogy a CPU-kat alkalmasakká tegyük a gépi tanulásra, órajelciklusonként sok műveletet kell elvégezni. Ezt vektoregységekkel érik el. Különböző neveken ismertek: ARM-on NEON-nak hívják, x86-on (egy újabb generáció) [AVX2](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions) egységeknek nevezik. Közös jellemzőjük, hogy SIMD (single instruction multiple data, egyetlen utasítás több adaton) műveleteket képesek végrehajtani. A :numref:`fig_neon128` ábra bemutatja, hogyan adható össze 8 rövid egész szám egyetlen órajelciklus alatt ARM-on.

![128 bites NEON vektorizáció.](../img/neon128.svg)
:label:`fig_neon128`

Az architektúra választásától függően az ilyen regiszterek akár 512 bit hosszúak is lehetnek, lehetővé téve akár 64 számpár kombinálását. Például két számot szorozhatunk össze és adhatjuk hozzá egy harmadikhoz, amit fused multiply-add-nak is neveznek. Az Intel [OpenVino](https://01.org/openvinotoolkit) ezeket használja a szerver osztályú CPU-kon végzett deep learning tiszteletre méltó áteresztőképességének eléréséhez. Vegyük figyelembe azonban, hogy ez a szám eltörpül ahhoz képest, amire a GPU-k képesek. Például az NVIDIA RTX 2080 Ti 4352 CUDA maggal rendelkezik, amelyek mindegyike bármikor képes ilyen műveletet feldolgozni.

### Gyorsítótár

Képzeljük el a következő helyzetet: van egy szerény, 4 magos CPU-nk, ahogy azt a fenti :numref:`fig_skylake` ábra mutatja, 2 GHz-es frekvencián futva.
Tegyük fel továbbá, hogy IPC (utasítás órajelciklusonként) számlálónk 1, és az egységeknél engedélyezett a 256 bites szélességű AVX2. Tegyük fel továbbá, hogy az AVX2 műveletek elvégzéséhez szükséges regiszterek legalább egyikét a memóriából kell lekérni. Ez azt jelenti, hogy a CPU $4 \times 256 \textrm{ bit} = 128 \textrm{ bájt}$ adatot fogyaszt órajelciklusonként. Ha nem vagyunk képesek $2 \times 10^9 \times 128 = 256 \times 10^9$ bájtot másodpercenként a processzorhoz juttatni, a feldolgozóelemek éhesek lesznek. Sajnos az ilyen chip memóriainterfésze csak 20--40 GB/s adatátvitelt támogat, azaz egy nagyságrenddel kevesebbet. A megoldás az, hogy amennyire lehetséges, kerüljük az *új* adatok memóriából való betöltését, és inkább helyileg gyorsítótárazzuk azokat a CPU-n. Ez az, ahol a gyorsítótárak hasznosak. A következő neveket vagy fogalmakat szokás használni:

* **A regiszterek** szigorúan véve nem részei a gyorsítótárnak. Segítenek az utasítások előkészítésében. Ennek ellenére a CPU regiszterek olyan memóriahelyek, amelyekhez a CPU órajel sebességgel, bármilyen késleltetési büntetés nélkül hozzáférhet. A CPU-k tíznyi regiszterrel rendelkeznek. A fordítótól (vagy programozótól) függ, hogy hatékonyan használja-e a regisztereket. Például a C programozási nyelvben van egy `register` kulcsszó.
* **Az L1 gyorsítótárak** az első védelmi vonal a magas memória-sávszélesség-igényekkel szemben. Az L1 gyorsítótárak apróak (jellemző méretük 32--64 KB) és sokszor adat- és utasítás-gyorsítótárakra osztódnak. Ha az adatokat az L1 gyorsítótárban találják, a hozzáférés nagyon gyors. Ha nem találják ott, a keresés lejjebb folytatódik a gyorsítótár-hierarchiában.
* **Az L2 gyorsítótárak** a következő megállók. Az architektúra-tervezéstől és a processzorok méretétől függően exkluzívak lehetnek. Elérhetők lehetnek csak egy adott mag, vagy több mag által megosztva. Az L2 gyorsítótárak nagyobbak (jellemzően 256--512 KB magonként) és lassabbak az L1-nél. Ezenkívül az L2-ben való hozzáféréshez először ellenőrizni kell, hogy az adat nincs az L1-ben, ami kis extra késleltetést ad.
* **Az L3 gyorsítótárak** több mag között osztottak, és elég nagyok lehetnek. Az AMD Epyc 3 szerver CPU-knak gigantikus 256 MB gyorsítótáruk van, amely több chiplet között van elosztva. Tipikusabb értékek a 4--8 MB tartományban vannak.

Megjósolni, hogy a memória melyik elemeit lesz szükség a következő lépésben, a chipek tervezésének egyik legfontosabb optimalizálási paramétere. Például célszerű a memóriát *előre haladó* irányban bejárni, mivel a legtöbb gyorsítótár-algoritmus *előre olvas* inkább, mint visszafelé. Hasonlóképpen, a memória-hozzáférési minták helyi tartása jó módja a teljesítmény javításának.

A gyorsítótárak hozzáadása kétélű fegyver. Egyrészt biztosítják, hogy a processzormagok ne éhezzenek adatokra. Ugyanakkor növelik a chip méretét, olyan területet foglalva el, amelyet egyébként a feldolgozási kapacitás növelésére lehetett volna fordítani. Ráadásul a *gyorsítótár-kihagyások* drágák lehetnek. Vegyük a legrosszabb esetet, a *hamis megosztást*, ahogy azt a :numref:`fig_falsesharing` ábra mutatja. Egy memóriahelyet a 0-s processzorban gyorsítótárazzák, amikor az 1-es processzoron lévő szál kéri az adatot. Annak megszerzéséhez a 0-s processzornak le kell állítania a tevékenységét, vissza kell írnia az információt a fő memóriába, majd hagynia kell az 1-es processzort, hogy a memóriából olvassa. Ez idő alatt mindkét processzor vár. Lehetséges, hogy az ilyen kód *lassabban* fut több processzoron, mint egy hatékony egyprocesszoros megvalósítás. Ez egy újabb ok arra, hogy a gyorsítótárak méretének van egy gyakorlati korlátja (fizikai méretükön túl).

![Hamis megosztás (kép az Intel jóvoltából).](../img/falsesharing.svg)
:label:`fig_falsesharing`

## GPU-k és egyéb gyorsítók

Nem túlzás azt állítani, hogy a deep learning nem lett volna sikeres GPU-k nélkül. Ugyanígy meglehetősen ésszerű azzal érvelni, hogy a GPU-gyártók vagyona jelentősen megnőtt a deep learning miatt. A hardver és az algoritmusok ezen együttes fejlődése olyan helyzethez vezetett, ahol a deep learning – jó és rossz értelemben egyaránt – az előnyben részesített statisztikai modellezési paradigma lett. Ezért érdemes megérteni a GPU-k és a kapcsolódó gyorsítók, például a TPU :cite:`Jouppi.Young.Patil.ea.2017` konkrét előnyeit.

Fontos megjegyezni a gyakorlatban sokszor tett megkülönböztetést: a gyorsítókat vagy tanításra, vagy következtetésre optimalizálják. Az utóbbihoz csak a hálózatban a forward propagációt kell kiszámítani. Nincs szükség a közbenső adatok tárolására a visszaterjesztés-höz. Ezenkívül nem feltétlenül van szükség nagyon pontos számításra (az FP16 vagy INT8 általában elegendő). Másrészt a tanítás során minden közbenső eredményt el kell tárolni a gradiensek kiszámításához. Ezenkívül a gradiensek akkumulálásához nagyobb pontosságra van szükség a numerikus alulcsordulás (vagy túlcsordulás) elkerülése érdekében. Ez azt jelenti, hogy az FP16 (vagy vegyes pontosság FP32-vel) a minimális követelmény. Mindez gyorsabb és nagyobb memóriát (HBM2 vs. GDDR6) és több feldolgozási kapacitást igényel. Például az NVIDIA [Turing](https://devblogs.nvidia.com/nvidia-turing-architecture-in-depth/) T4 GPU-k következtetésre vannak optimalizálva, míg a V100 GPU-k tanításhoz előnyösebbek.

Idézzük fel a vektorizációt, ahogyan azt a :numref:`fig_neon128` ábra illusztrálja. A processzormaghoz hozzáadott vektoregységek lehetővé tették az áteresztőképesség jelentős növelését. Például a :numref:`fig_neon128` példában egyszerre 16 műveletet tudtunk elvégezni.
Először, mi lenne, ha olyan műveleteket adnánk hozzá, amelyek nem csak vektorok, hanem mátrixok közötti műveletek elvégzésére is optimalizáltak? Ez a stratégia tenzormagokhoz (amelyeket hamarosan tárgyalunk) vezetett.
Másodszor, mi lenne, ha sokkal több magot adnánk hozzá? Röviden: ez a két stratégia foglalja össze a GPU-k tervezési döntéseit. A :numref:`fig_turing_processing_block` ábra áttekintést ad egy alapvető feldolgozóblokkról. 16 egész szám és 16 lebegőpontos egységet tartalmaz. Ezeken kívül két tenzormag felgyorsítja a deep learningre vonatkozó, szűkebb körű extra műveletek egy részhalmazát. Minden streaming multiprocesszor négy ilyen blokkból áll.

![NVIDIA Turing feldolgozóblokk (kép az NVIDIA jóvoltából).](../img/turing-processing-block.png)
:width:`150px`
:label:`fig_turing_processing_block`

Ezután 12 streaming multiprocesszort grafikus feldolgozó klaszterekbe csoportosítanak, amelyek a csúcskategóriás TU102 processzorokat alkotják. Bőséges memóriacsatornák és egy L2 gyorsítótár egészíti ki a felépítést. A :numref:`fig_turing` ábra tartalmazza a releváns részleteket. Az ilyen eszköz tervezésének egyik oka, hogy az egyes blokkokat szükség szerint hozzá lehet adni vagy el lehet távolítani, lehetővé téve kompaktabb chipek készítését és a hozamproblémák kezelését (a hibás modulok nem aktiválhatók). Szerencsére az ilyen eszközök programozása jól el van rejtve az alkalmi deep learning kutató elől a CUDA és a keretrendszer kódjának rétegei alatt. Különösen, egynél több program is egyidejűleg hajtható végre a GPU-n, feltéve, hogy rendelkezésre állnak szabad erőforrások. Ennek ellenére érdemes tisztában lenni az eszközök korlátaival, hogy elkerüljük az eszköz memóriájába nem illő modellek kiválasztását.

![NVIDIA Turing architektúra (kép az NVIDIA jóvoltából)](../img/turing.png)
:width:`350px`
:label:`fig_turing`

Az utolsó szempont, amelyet részletesebben érdemes megemlíteni, a *tenzormagok*. Ezek egy olyan közelmúltbeli trend példáját képviselik, amikor olyan, kifejezetten a deep learningre hatékony, optimalizált áramköröket adnak hozzá. Például a TPU szisztolikus tömböt :cite:`Kung.1988` adott hozzá a gyors mátrixszorzáshoz. Ott a tervezés egy nagyon kis számú (az első generációs TPU-knál egy) nagy műveletet kívánt támogatni. A tenzormagok ennek az ellentétes végén vannak. $4 \times 4$ és $16 \times 16$ méretek közötti mátrixokat érintő kis műveletekre vannak optimalizálva, numerikus pontosságuktól függően. A :numref:`fig_tensorcore` ábra áttekintést ad az optimalizálásokról.

![NVIDIA tenzormagok a Turingban (kép az NVIDIA jóvoltából).](../img/tensorcore.jpg)
:width:`400px`
:label:`fig_tensorcore`

Nyilvánvalóan, amikor a számítást optimalizáljuk, bizonyos kompromisszumokat kell kötni. Az egyik az, hogy a GPU-k nem nagyon jók a megszakítások és ritka adatok kezelésében. Bár vannak figyelemre méltó kivételek, mint a [Gunrock](https://github.com/gunrock/gunrock) :cite:`Wang.Davidson.Pan.ea.2016`, a ritka mátrixok és vektorok hozzáférési mintái nem illenek jól a nagy sávszélességű burst olvasási műveletekhez, amelyekben a GPU-k kiválóak. Mindkét cél összehangolása aktív kutatási terület. Lásd például a [DGL](http://dgl.ai)-t, egy gráf alapú deep learningre hangolt könyvtárat.


## Hálózatok és buszok

Amikor egyetlen eszköz nem elegendő az optimalizáláshoz, adatokat kell átvinni hozzá és tőle a feldolgozás szinkronizálásához. Ez az, ahol a hálózatok és buszok hasznosak. Számos tervezési paraméterünk van: sávszélesség, költség, távolság és rugalmasság.
Az egyik véglet a WiFi, amely meglehetősen jó hatótávolsággal rendelkezik, nagyon könnyen használható (nincsenek kábelek), olcsó, de viszonylag gyenge sávszélességet és késleltetést kínál. Egyetlen ésszerűen gondolkodó gépi tanulás kutató sem használná szerverfürtök építéséhez. A továbbiakban azokra az összeköttetésekre összpontosítunk, amelyek alkalmasak a deep learningre.

* **A PCIe** egy dedikált busz a nagyon nagy sávszélességű pontok közötti kapcsolatokhoz (PCIe 4.0-n 16 sávos slotban akár 32 GB/s) sávonként. A késleltetés egyjegyű mikroszekundumos nagyságrendű (5 μs). A PCIe linkek értékesek. A processzoroknak csak korlátozott számú van belőlük: az AMD EPYC 3-nak 128 sávja van, az Intel Xeon chipenkénti 48 sávot kínál; asztali szintű CPU-kon ezek a számok 20 (Ryzen 9) és 16 (Core i9). Mivel a GPU-knak általában 16 sávjuk van, ez korlátozza a CPU-hoz teljes sávszélességgel csatlakoztatható GPU-k számát. Elvégre meg kell osztaniuk a linkeket más nagy sávszélességű perifériákkal, mint a tárolók és az Ethernet. Ahogy a RAM-hozzáférésnél, itt is a nagy tömeges átvitelek preferáltak a csökkentett csomag-overhead miatt.
* **Az Ethernet** a számítógépek összekapcsolásának leggyakrabban használt módja. Bár jelentősen lassabb a PCIe-nél, nagyon olcsó, könnyen telepíthető és sokkal hosszabb távolságokat fed le. Az alacsony szintű szerverek tipikus sávszélessége 1 GBit/s. A csúcskategóriás eszközök (például felhőben lévő [C5 példányok](https://aws.amazon.com/ec2/instance-types/c5/)) 10 és 100 GBit/s közötti sávszélességet kínálnak. Mint minden korábbi esetben, az adatátvitelnek jelentős overhead-je van. Vegyük figyelembe, hogy szinte soha nem használjuk közvetlenül a nyers Ethernetet, hanem inkább a fizikai összeköttetésen futó protokollt (mint az UDP vagy a TCP/IP). Ez további overheadet ad. Mint a PCIe, az Ethernet is két eszköz összekapcsolására van tervezve, például egy számítógép és egy switch.
* **A switchek** lehetővé teszik több eszköz csatlakoztatását oly módon, hogy bármely pár egyidejűleg (általában teljes sávszélességű) pontok közötti kapcsolatot létesíthet. Például az Ethernet switchek 40 szervert köthetnek össze magas keresztmetszeti sávszélességgel. Vegyük figyelembe, hogy a switchek nem egyedülállóak a hagyományos számítógép-hálózatokban. Még a PCIe sávok is [kapcsolhatók](https://www.broadcom.com/products/pcie-switches-bridges/pcie-switches). Ez például nagyszámú GPU gazdaprocesszorhoz való csatlakoztatásakor fordul elő, ahogyan az a [P2 példányoknál](https://aws.amazon.com/ec2/instance-types/p2/) is van.
* **Az NVLink** alternatívája a PCIe-nek, ha nagyon nagy sávszélességű összeköttetésekről van szó. Linkenként akár 300 Gbit/s adatátviteli sebességet kínál. A szerver GPU-k (Volta V100) hat linkkel rendelkeznek, míg a fogyasztói szintű GPU-k (RTX 2080 Ti) csak eggyel, csökkentett 100 Gbit/s sebességgel. A GPU-k közötti magas adatátvitel eléréséhez a [NCCL](https://github.com/NVIDIA/nccl) használatát ajánljuk.



## További késleltetési számok

A :numref:`table_latency_numbers` és :numref:`table_latency_numbers_tesla` összefoglalói [Eliot Eshelman](https://gist.github.com/eshelman) munkájából származnak, aki az aktualizált számokat [GitHub gist](https://gist.github.com/eshelman/343a1c46cb3fba142c1afdcdeec17646) formájában tartja naprakészen.

:Általános késleltetési számok.

| Művelet | Idő | Megjegyzés |
| :----------------------------------------- | -----: | :---------------------------------------------- |
| L1 gyorsítótár hivatkozás/találat                     | 1,5 ns | 4 ciklus                                        |
| Lebegőpontos összeadás/szorzás/FMA                | 1,5 ns | 4 ciklus                                        |
| L2 gyorsítótár hivatkozás/találat                     |   5 ns | 12 ~ 17 ciklus                                  |
| Elágazás-hibabecslés                          |   6 ns | 15 ~ 20 ciklus                                  |
| L3 gyorsítótár találat (nem osztott gyorsítótár)              |  16 ns | 42 ciklus                                       |
| L3 gyorsítótár találat (más magban megosztott)      |  25 ns | 65 ciklus                                       |
| Mutex zárolás/feloldás                          |  25 ns |                                                 |
| L3 gyorsítótár találat (más magban módosított)    |  29 ns | 75 ciklus                                       |
| L3 gyorsítótár találat (távoli CPU foglalatban)      |  40 ns | 100 ~ 300 ciklus (40 ~ 116 ns)                  |
| QPI ugrás egy másik CPU-hoz (ugrásonként)         |  40 ns |                                                 |
| 64 MB memóriahivatkozás (helyi CPU)          |  46 ns | TinyMemBench Broadwell E5-2690v4-en             |
| 64 MB memóriahivatkozás (távoli CPU)         |  70 ns | TinyMemBench Broadwell E5-2690v4-en             |
| 256 MB memóriahivatkozás (helyi CPU)         |  75 ns | TinyMemBench Broadwell E5-2690v4-en             |
| Intel Optane véletlen írás                  |  94 ns | UCSD Non-Volatile Systems Lab                   |
| 256 MB memóriahivatkozás (távoli CPU)        | 120 ns | TinyMemBench Broadwell E5-2690v4-en             |
| Intel Optane véletlen olvasás                   | 305 ns | UCSD Non-Volatile Systems Lab                   |
| 4 KB küldése 100 Gbps HPC fabric-on          |   1 μs | MVAPICH2 Intel Omni-Path-on                   |
| 1 KB tömörítése Google Snappy-val            |   3 μs |                                                 |
| 4 KB küldése 10 Gbps Etherneten             |  10 μs |                                                 |
| 4 KB véletlen írás NVMe SSD-re             |  30 μs | DC P3608 NVMe SSD (QOS 99% 500 μs)            |
| 1 MB átvitele NVLink GPU-ra/ról            |  30 μs | ~33 GB/s NVIDIA 40 GB NVLink-en                 |
| 1 MB átvitele PCI-E GPU-ra/ról             |  80 μs | ~12 GB/s PCIe 3.0 x16 linken                  |
| 4 KB véletlen olvasás NVMe SSD-ről            | 120 μs | DC P3608 NVMe SSD (QOS 99%)                     |
| 1 MB szekvenciális olvasás NVMe SSD-ről        | 208 μs | ~4,8 GB/s DC P3608 NVMe SSD                    |
| 4 KB véletlen írás SATA SSD-re             | 500 μs | DC S3510 SATA SSD (QOS 99,9%)                   |
| 4 KB véletlen olvasás SATA SSD-ről            | 500 μs | DC S3510 SATA SSD (QOS 99,9%)                   |
| Oda-vissza út ugyanazon adatközponton belül          | 500 μs | Egyirányú ping ~250 μs                          |
| 1 MB szekvenciális olvasás SATA SSD-ről        |   2 ms | ~550 MB/s DC S3510 SATA SSD                    |
| 1 MB szekvenciális olvasás lemezről            |   5 ms | ~200 MB/s szerver HDD                           |
| Véletlen lemezhozzáférés (keresés+forgás)         |  10 ms |                                                 |
| Csomag küldése CA->Hollandia->CA            | 150 ms |                                                 |
:label:`table_latency_numbers`

:NVIDIA Tesla GPU-k késleltetési számai.

| Művelet | Idő | Megjegyzés |
| :------------------------------ | -----: | :---------------------------------------- |
| GPU megosztott memória hozzáférés        |  30 ns | 30~90 ciklus (banki ütközések késleltetést adnak) |
| GPU globális memória hozzáférés        | 200 ns | 200~800 ciklus                            |
| CUDA kernel indítása GPU-n       |  10 μs | A gazdagép CPU utasítja a GPU-t a kernel indítására    |
| 1 MB átvitele NVLink GPU-ra/ról |  30 μs | ~33 GB/s NVIDIA 40 GB NVLink-en           |
| 1 MB átvitele PCI-E GPU-ra/ról  |  80 μs | ~12 GB/s PCI-Express x16 linken         |
:label:`table_latency_numbers_tesla`

## Összefoglalás

* Az eszközöknek overhead-jük van a műveletekhez. Ezért fontos, hogy kevés nagy átvitelre törekedjünk, nem sok kis átvitelre. Ez vonatkozik a RAM-ra, SSD-kre, hálózatokra és GPU-kra.
* A vektorizáció kulcsfontosságú a teljesítményhez. Győződjön meg arról, hogy ismeri a gyorsítója konkrét képességeit. Például egyes Intel Xeon CPU-k különösen jók az INT8 műveleteknél, az NVIDIA Volta GPU-k kiemelkednek az FP16 mátrix-mátrix műveleteknél, az NVIDIA Turing pedig az FP16, INT8 és INT4 műveleteknél jeleskedik.
* A kis adattípusokból eredő numerikus túlcsordulás problémát okozhat tanítás közben (és kisebb mértékben következtetés közben).
* Az aliasing jelentősen ronthatja a teljesítményt. Például a 64 bites CPU-kon a memóriaigazítást 64 bites határokhoz képest kell elvégezni. GPU-kon jó ötlet a konvolúciók méretét igazítani, például a tenzormagokhoz.
* Illeszkedjen az algoritmusokhoz a hardverhez (pl. memóriaigény és sávszélesség). Nagy sebességnövekedés (nagyságrendek) érhető el, ha a paramétereket gyorsítótárakba helyezzük.
* Javasoljuk, hogy egy új algoritmus teljesítményét papíron vázolja fel, mielőtt az kísérleti eredményeket ellenőrizné. Az egy nagyságrendet meghaladó eltérések aggodalomra adnak okot.
* Profilozók segítségével derítse fel a teljesítmény szűk keresztmetszeteit.
* A tanítási és következtetési hardvereknek eltérő ár-teljesítmény optimumpontjaik vannak.

## Feladatok

1. Írj C kódot annak tesztelésére, hogy van-e sebességkülönbség a külső memóriainterfészhez képest igazított vagy nem igazított memória-hozzáférés között. Tipp: légy óvatos a gyorsítótárazási hatásokkal.
1. Teszteld a sebességkülönbséget a memória szekvenciálisan vagy adott lépésközzel való hozzáférése között.
1. Hogyan mérhető a CPU-n lévő gyorsítótárak mérete?
1. Hogyan osztanád el az adatokat több memóriacsatornán a maximális sávszélesség érdekében? Hogyan osztanád el, ha sok kis szálra lenne szükség?
1. Egy vállalati szintű merevlemez 10 000 RPM-en forog. Mi az a feltétlenül szükséges minimális idő, amelyet a merevlemeznek a legrosszabb esetben el kell töltenie, mielőtt adatokat olvashat (feltételezve, hogy a fejek szinte azonnal mozognak)? Miért válnak a 2,5"-os HDD-k egyre népszerűbbé a kereskedelmi szervereknél (a 3,5"-os és 5,25"-os meghajtókhoz képest)?
1. Tegyük fel, hogy egy merevlemez-gyártó 1 Tbit per négyzetinchről 5 Tbit per négyzetinchre növeli a tárolási sűrűséget. Mennyi információt tárolhat egy 2,5"-os HDD gyűrűjén? Van-e különbség a belső és a külső sávok között?
1. A 8 bitesről 16 bites adattípusra való áttérés körülbelül négyszeresére növeli a szilícium mennyiségét. Miért? Miért adhatott az NVIDIA INT4 műveleteket a Turing GPU-ikhoz?
1. Mennyivel gyorsabb a memória előre olvasása, mint visszafelé olvasása? Eltér-e ez a szám különböző számítógépek és CPU-gyártók között? Miért? Írj C kódot és kísérletezz.
1. Meg tudod-e mérni a lemez gyorsítótárának méretét? Mi az egy tipikus HDD esetén? Szükségük van-e az SSD-knek gyorsítótárra?
1. Mérd meg a csomagok overhead-jét, amikor üzeneteket küld az Etherneten keresztül. Keresd meg az UDP és a TCP/IP kapcsolatok közötti különbséget.
1. A közvetlen memória-hozzáférés (DMA) lehetővé teszi a CPU-n kívüli eszközök számára, hogy közvetlenül írjanak (és olvassanak) a memóriába (memóriából). Miért jó ötlet ez?
1. Tekintsd meg a Turing T4 GPU teljesítményszámait. Miért "csak" duplázódik a teljesítmény, ahogy az FP16-ról az INT8-ra és INT4-re megy?
1. Mi a legrövidebb idő, amelyre egy csomagnak szüksége van San Francisco és Amszterdam közötti oda-vissza útra? Tipp: feltételezheti, hogy a távolság 10 000 km.


[Discussions](https://discuss.d2l.ai/t/363)
