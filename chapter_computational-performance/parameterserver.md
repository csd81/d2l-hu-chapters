# Paraméterkiszolgálók
:label:`sec_parameterserver`

Ahogy egyetlen GPU-ról több GPU-ra, majd több GPU-t tartalmazó szerverekre váltunk, amelyek akár több rackben és hálózati kapcsolón keresztül is szétoszolhatnak, az elosztott és párhuzamos tanításhoz használt algoritmusainknak sokkal kifinomultabbá kell válniuk. A részletek számítanak, mert a különböző összeköttetések nagyon eltérő sávszélességgel rendelkeznek (például megfelelő beállításban az NVLink akár 100 GB/s-ot is kínálhat 6 linken keresztül, a PCIe 4.0 (16 sávos) 32 GB/s-ot nyújt, míg még a nagy sebességű 100GbE Ethernet is csak 10 GB/s-ot jelent). Ugyanakkor ésszerűtlen elvárni, hogy egy statisztikai modellező hálózati és rendszertechnikai szakértő legyen.

A paraméterkiszolgáló alapötletét :citet:`Smola.Narayanamurthy.2010` vezette be elosztott látensváltozó-modellek kontextusában. Ezt követte a `push` és `pull` szemantika leírása :citet:`Ahmed.Aly.Gonzalez.ea.2012`-ben, majd a rendszer és egy nyílt forráskódú könyvtár bemutatása :citet:`Li.Andersen.Park.ea.2014`-ben. Az alábbiakban azokat az összetevőket mutatjuk be, amelyekre a hatékonysághoz szükség van.


## Adatpárhuzamos tanítás

Nézzük át az adatpárhuzamos tanítás megközelítését az elosztott tanításhoz. Ebben a szakaszban kizárólag ezt használjuk, mivel a gyakorlatban lényegesen egyszerűbb megvalósítani. Manapság gyakorlatilag nincs olyan eset, ahol más párhuzamosítási stratégia előnyösebb lenne (a gráfokon végzett mélytanulást kivéve), mert a GPU-k ma már bőséges memóriával rendelkeznek. A :numref:`fig_parameterserver` ábra azt az adatpárhuzamos változatot mutatja, amelyet a :numref:`sec_multi_gpu` szakaszban valósítottunk meg. Ennek kulcseleme, hogy a grádiensek összegyűjtése egyetlen GPU-n (a 0-s GPU-n) történik, mielőtt a frissített paramétereket újra szétküldenénk az összes GPU-ra.

![Balra: egyetlen GPU-s tanítás. Jobbra: a több GPU-s tanítás egy változata: (1) kiszámítjuk a veszteséget és a gradienst, (2) az összes gradienst egy GPU-n gyűjtjük össze, (3) a paraméterfrissítés megtörténik, és a paramétereket újra szétosztjuk az összes GPU-ra.](../img/ps.svg)
:label:`fig_parameterserver`

Visszatekintve a 0-s GPU-ra történő összegyűjtés meglehetősen önkényes döntésnek tűnik. Végső soron ugyanígy összegyűjthetnénk őket a CPU-n is. Sőt, akár úgy is dönthetnénk, hogy egyes paramétereket az egyik GPU-n, másokat pedig egy másikon aggregálunk. Ha az optimalizálási algoritmus ezt támogatja, valójában nincs ok arra, hogy ezt ne tehetnénk meg. Például ha négy paramétervektorunk van a hozzájuk tartozó $\mathbf{g}_1, \ldots, \mathbf{g}_4$ gradiensekkel, akkor az egyes $\mathbf{g}_i$-khez ($i = 1, \ldots, 4$) külön GPU-n gyűjthetnénk össze a gradienseket.


Ez az érvelés önkényesnek és könnyednek tűnhet. Végső soron a matematika mindenhol ugyanaz. Ugyanakkor valódi fizikai hardverrel dolgozunk, ahol a különböző buszok eltérő sávszélességgel rendelkeznek, ahogy azt a :numref:`sec_hardware` szakaszban tárgyaltuk.
Vegyünk egy valós, négy GPU-s szervert a :numref:`fig_bw_hierarchy` ábrán látható módon. Ha különösen jól kapcsolódik, lehet rajta 100 GbE hálózati kártya. Jellemzőbbek az 1--10 GbE tartományba eső értékek, 100 MB/s és 1 GB/s közötti effektív sávszélességgel.
Mivel a CPU-knak túl kevés PCIe sávjuk van ahhoz, hogy minden GPU-hoz közvetlenül kapcsolódjanak (például a fogyasztói Intel CPU-k 24 sávval rendelkeznek), szükségünk van egy [multiplexelőre](https://www.broadcom.com/products/pcie-switches-bridges/pcie-switches). A CPU és egy 16x Gen3 kapcsolat sávszélessége 16 GB/s. Ez az a sebesség is, amellyel *mindegyik* GPU kapcsolódik a kapcsolóhoz. Vagyis hatékonyabb a kommunikációt az eszközök között bonyolítani.

![Egy 4 GPU-s szerver.](../img/bw-hierarchy.svg)
:label:`fig_bw_hierarchy`

Tegyük fel, hogy a gradiensek mérete 160 MB. Ebben az esetben 30 ms-ba kerül, hogy a három másik GPU-ról a gradienseket a negyedikre küldjük (egy átvitel 10 ms = 160 MB / 16 GB/s). Ha további 30 ms-ot adunk a súlyvektorok visszaküldésére, összesen 60 ms-t kapunk.
Ha minden adatot a CPU-ra küldünk, 40 ms-os büntetést fizetünk, mivel *mindegyik* GPU-nak el kell küldenie az adatot a CPU-ra, ami összesen 80 ms-ot jelent. Tegyük fel végül, hogy a gradienseket négy, egyenként 40 MB-os részre tudjuk bontani. Ekkor az egyes részeket *egyszerre* összegyűjthetjük különböző GPU-kon, mivel a PCIe kapcsoló teljes sávszélességű működést biztosít az összes kapcsolat között. 30 ms helyett ez 7.5 ms-ot vesz igénybe, vagyis a szinkronizációs művelet teljes ideje 15 ms lesz. Röviden: attól függően, hogyan szinkronizáljuk a paramétereket, ugyanaz a művelet 15 ms és 80 ms között is eltarthat. A :numref:`fig_ps_distributed` ábra a különböző paramétercsere-stratégiákat mutatja.

![Paraméter-szinkronizációs stratégiák.](../img/ps-distributed.svg)
:label:`fig_ps_distributed`

Érdemes megjegyezni, hogy a teljesítmény javítására még egy eszköz áll rendelkezésünkre: egy mély hálózatban időbe telik az összes gradiens kiszámítása felülről lefelé. Egyes paramétercsoportok gradienseit már akkor is elkezdhetjük szinkronizálni, amikor mások számítása még folyamatban van. A részletekhez lásd például :citet:`Sergeev.Del-Balso.2018`-at a [Horovod](https://github.com/horovod/horovod) használatáról.

## Gyűrűs szinkronizáció

Ha modern mélytanulási hardveren szinkronizálunk, gyakran erősen testre szabott hálózati kapcsolódással találkozunk. Például az AWS p3.16xlarge és az NVIDIA DGX-2 példányok a :numref:`fig_nvlink` ábrán látható kapcsolódási szerkezetet használják. Minden GPU egy PCIe kapcsolaton keresztül csatlakozik a gazda CPU-hoz, amely legfeljebb 16 GB/s sebességet biztosít. Emellett minden GPU-nak 6 NVLink kapcsolata is van, amelyek közül mindegyik 300 Gbit/s kétirányú átvitelre képes. Ez nagyjából 18 GB/s-t jelent kapcsolatként, irányonként. Röviden: az összesített NVLink-sávszélesség jelentősen nagyobb, mint a PCIe-sávszélesség. A kérdés az, hogyan használjuk ezt a leghatékonyabban.

![NVLink összeköttetés 8 V100 GPU-s szervereken (kép az NVIDIA jóvoltából).](../img/nvlink.svg)
:label:`fig_nvlink`

Kiderül, hogy az optimális szinkronizációs stratégia a hálózat két gyűrűre bontása és az adatok közvetlen szinkronizálása :cite:`Wang.Li.Liberty.ea.2018`. A :numref:`fig_nvlink_twoloop` ábra azt szemlélteti, hogy a hálózat felbontható egy kétszeres NVLink-sávszélességű gyűrűre (1-2-3-4-5-6-7-8-1), valamint egy szabványos sávszélességű gyűrűre (1-4-6-3-5-8-2-7-1). Hatékony szinkronizációs protokoll tervezése ebben az esetben nem triviális.

![Az NVLink hálózat felbontása két gyűrűre.](../img/nvlink-twoloop.svg)
:label:`fig_nvlink_twoloop`


Gondoljunk a következő gondolatkísérletre: adott egy $n$ számítási csomópontból (vagy GPU-ból) álló gyűrű, amelyben a gradiens az első csomópontról a másodikra küldhető. Ott hozzáadódik a helyi gradienshez, majd a harmadik csomópontra továbbítódik, és így tovább. $n-1$ lépés után az összegzett gradiens az utoljára érintett csomópontban található. Vagyis a grádiensek összegyűjtéséhez szükséges idő lineárisan nő a csomópontok számával. Ha ezt így csináljuk, az algoritmus meglehetősen nem hatékony, hiszen egy időpillanatban csak egy csomópont kommunikál. Mi lenne, ha a gradienseket $n$ részre bontanánk, és az $i$-edik darab szinkronizálását az $i$-edik csomópontnál kezdenénk?
Mivel minden darab mérete $1/n$, a teljes idő most $(n-1)/n \approx 1$. Más szóval a grádiensek összegyűjtésére fordított idő *nem nő* a gyűrű méretének növelésével. Ez meglepő eredmény. A :numref:`fig_ringsync` ábra szemlélteti a lépések sorozatát $n=4$ csomópont esetén.

![Gyűrűs szinkronizáció 4 csomóponton. Minden csomópont elkezdi a gradiens részeit a bal oldali szomszédjának küldeni, amíg az összegyűjtött gradiens megtalálható a jobb oldali szomszédban.](../img/ringsync.svg)
:label:`fig_ringsync`

Ha ugyanazt a példát vesszük, vagyis 160 MB szinkronizálását 8 V100 GPU között, akkor körülbelül $2 \cdot 160 \textrm{MB} / (3 \cdot 18 \textrm{GB/s}) \approx 6 \textrm{ms}$-t kapunk. Ez jobb, mint a PCIe busz használata, még akkor is, ha most már 8 GPU-t használunk. Megjegyzendő, hogy a gyakorlatban ezek az értékek valamivel rosszabbak, mivel a mélytanulási keretrendszerek gyakran nem képesek a kommunikációt nagy, összefüggő átviteli csomagokká összegyűjteni.

Gyakori tévhit, hogy a gyűrűs szinkronizáció alapvetően különbözik más szinkronizációs algoritmusoktól. Valójában az egyetlen különbség az, hogy a szinkronizációs út valamivel összetettebb egy egyszerű fához képest.

## Többgépes tanítás

Az elosztott tanítás több gépen további kihívást jelent: olyan kiszolgálókkal kell kommunikálnunk, amelyek csak viszonylag alacsonyabb sávszélességű hálózaton keresztül kapcsolódnak, és ez bizonyos esetekben nagyságrendekkel lassabb lehet.
Az eszközök közötti szinkronizáció bonyolult. Végül is a tanító kódot futtató különböző gépek sebessége kissé eltérő lesz. Ezért *szinkronizálnunk* kell őket, ha szinkron elosztott optimalizálást akarunk használni. A :numref:`fig_ps_multimachine` ábra szemlélteti az elosztott párhuzamos tanítás folyamatát.

1. Minden gépen egy (különböző) adathalmazbeli batch kerül beolvasásra, több GPU között szétosztva, majd GPU-memóriába továbbítva. Ott minden GPU-batchen külön-külön számítjuk ki az előrejelzéseket és a gradienseket.
2. Az összes helyi GPU gradienseit egy GPU-n összegyűjtjük (vagy azokat több GPU között részben összegyűjtjük).
3. A gradienseket elküldjük a CPU-knak.
4. A CPU-k a gradienseket egy központi paraméterkiszolgálóra küldik, amely minden gradienst összegyűjt.
5. Az összegzett gradienseket ezután a paraméterek frissítésére használjuk, a frissített paramétereket pedig visszasugározzuk az egyes CPU-kra.
6. Az információt egy (vagy több) GPU-ra továbbítjuk.
7. A frissített paraméterek szétterjednek az összes GPU között.

![Többgépes, több GPU-s elosztott párhuzamos tanítás.](../img/ps-multimachine.svg)
:label:`fig_ps_multimachine`

Minden egyes művelet meglehetősen egyszerűnek tűnik. És valóban, egyetlen gépen belül hatékonyan is végrehajthatók. Amikor azonban több gépre tekintünk, láthatóvá válik, hogy a központi paraméterkiszolgáló válik a szűk keresztmetszetté. Hiszen a szerverenkénti sávszélesség korlátozott, ezért $m$ dolgozó esetén minden gradiens szerverre küldésének ideje $\mathcal{O}(m)$. Ezt a korlátot úgy törhetjük át, ha a szerverek számát $n$-re növeljük. Ekkor minden szervernek csak a paraméterek $\mathcal{O}(1/n)$ részét kell tárolnia, így a frissítések és az optimalizálás teljes ideje $\mathcal{O}(m/n)$ lesz.
Mindkét szám összehangolása konstans skálázódást ad, függetlenül attól, hány dolgozóval dolgozunk. A gyakorlatban ugyanazokat a gépeket használjuk dolgozóként és szerverként is. A :numref:`fig_ps_multips` ábra ezt a kialakítást mutatja (részletekért lásd még :cite:`Li.Andersen.Park.ea.2014`).
Különösen nem triviális annak biztosítása, hogy több gép elfogadhatatlan késlekedések nélkül működjön. 

![Fent: egyetlen paraméterkiszolgáló szűk keresztmetszet, mivel a sávszélessége véges. Lent: több paraméterkiszolgáló tárolja a paraméterek részeit összesített sávszélességgel.](../img/ps-multips.svg)
:label:`fig_ps_multips`

## Kulcs-érték tárolók

Az elosztott, több GPU-s tanításhoz szükséges lépések gyakorlati megvalósítása nem triviális.
Ezért hasznos egy közös absztrakciót használni, mégpedig egy újradefiniált frissítési szemantikájú *kulcs-érték tárolót*.


Sok dolgozó és sok GPU esetén az $i$-edik gradiens számítása a következőképpen definiálható:

$$\mathbf{g}_{i} = \sum_{k \in \textrm{workers}} \sum_{j \in \textrm{GPUs}} \mathbf{g}_{ijk},$$

ahol $\mathbf{g}_{ijk}$ az $i$-edik gradiens azon része, amely a $k$-adik dolgozó $j$-edik GPU-ján lett felosztva.
Ennek a műveletnek a lényege, hogy *kommutatív redukció*, vagyis sok vektort egyetlen vektorrá alakít, és a művelet alkalmazásának sorrendje nem számít. Ez kiválóan megfelel céljainkra, mivel nem szükséges (és nem is kell) pontosan szabályoznunk, mikor érkezik melyik gradiens. Megjegyzendő továbbá, hogy ez a művelet különböző $i$-kre egymástól függetlenül zajlik.

Ez lehetővé teszi a következő két művelet definiálását: a *push*, amely összegyűjti a gradienseket, és a *pull*, amely kinyeri az összegzett gradienseket. Mivel sok különböző gradienssel dolgozunk (végül is sok rétegünk van), a gradienseket egy $i$ kulccsal kell indexelni. Ez a hasonlóság a kulcs-érték tárolókhoz, mint például a Dynamóban bevezetett :cite:`DeCandia.Hastorun.Jampani.ea.2007`, nem véletlen. Azok is sok hasonló tulajdonsággal rendelkeznek, különösen a paraméterek több kiszolgáló közötti elosztása tekintetében.


A kulcs-érték tárolók `push` és `pull` műveletei a következők:

* **push(key, value)** egy adott gradienst (az értéket) küld el egy dolgozótól egy közös tárolóba. Ott az érték összegződik, például összeadással.
* **pull(key, value)** egy összegzett értéket olvas ki a közös tárolóból, például miután az összes dolgozó gradienseit egyesítették.

Ha a szinkronizáció minden bonyolultságát egy egyszerű `push` és `pull` művelet mögé rejtjük, szétválaszthatjuk a statisztikai modellezők igényeit, akik egyszerűen szeretnék kifejezni az optimalizálást, és a rendszerfejlesztők feladatait, akiknek az elosztott szinkronizáció belső bonyolultságával kell megküzdeniük.

## Összefoglalás

* A szinkronizációnak erősen alkalmazkodnia kell az adott hálózati infrastruktúrához és a szerveren belüli kapcsolódáshoz. Ez jelentősen befolyásolhatja a szinkronizáció idejét.
* A gyűrűs szinkronizáció a p3 és DGX-2 szervereknél optimális lehet. Más rendszereknél nem feltétlenül.
* Hierarchikus szinkronizációs stratégia jól működik, amikor több paraméterkiszolgálót adunk hozzá a nagyobb sávszélesség érdekében.


## Gyakorlatok

1. Tovább gyorsítható a gyűrűs szinkronizáció? Tipp: az üzeneteket mindkét irányba is lehet küldeni.
1. Lehetséges aszinkron kommunikációt engedélyezni, miközben a számítás még folyamatban van? Hogyan hat ez a teljesítményre?
1. Mi történik, ha egy hosszú futású számítás közben elveszítünk egy szervert? Hogyan tervezhetnénk olyan *hibatűrő* mechanizmust, amely megakadályozza a teljes újraindítást?


[Discussions](https://discuss.d2l.ai/t/366)
