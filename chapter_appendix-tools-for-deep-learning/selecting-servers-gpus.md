# Szerverek és GPU-k kiválasztása
:label:`sec_buy_gpu`

A deep learning tanítás általában nagy mennyiségű számítást igényel. Jelenleg a GPU-k a legköltséghatékonyabb hardveres gyorsítók a deep learning számára. Különösen a CPU-khoz képest a GPU-k olcsóbbak és nagyobb teljesítményt nyújtanak, gyakran egy nagyságrenddel is. Ezenkívül egyetlen szerver több GPU-t is támogathat, akár 8-at a csúcskategóriás szerverek esetén. A mérnöki munkaállomásoknál általánosabb szám legfeljebb 4 GPU, mivel a hő-, hűtési és teljesítményigények gyorsan meghaladják azt, amit egy irodaépület képes kiszolgálni. Nagyobb telepítéseknél a felhőalapú számítástechnika (pl. az Amazon [P3](https://aws.amazon.com/ec2/instance-types/p3/) és [G4](https://aws.amazon.com/blogs/aws/in-the-works-ec2-instances-g4-with-nvidia-t4-gpus/) példányai) sokkal praktikusabb megoldás.


## Szerverek kiválasztása

Általában nincs szükség csúcskategóriás, sok szálú CPU-k vásárlására, mivel a számítások nagy része a GPU-kon zajlik. Ugyanakkor a Python globális értelmező zárja (GIL) miatt a CPU egyszálú teljesítménye számíthat olyan helyzetekben, ahol 4--8 GPU-nk van. Egyenlő feltételek mellett ez azt sugallja, hogy a kevesebb maggal, de magasabb órajel-frekvenciával rendelkező CPU-k gazdaságosabb választás lehetnek. Például egy 6 magos 4 GHz-es és egy 8 magos 3.5 GHz-es CPU között választva az előbbi sokkal előnyösebb, még akkor is, ha összesített sebessége kisebb.
Fontos szempont, hogy a GPU-k sok energiát fogyasztanak, és ezáltal sok hőt termelnek. Ez nagyon jó hűtést és elég nagy házat igényel a GPU-k elhelyezéséhez. Ha lehet, kövesd az alábbi irányelveket:

1. **Tápegység**. A GPU-k jelentős mennyiségű energiát fogyasztanak. Eszközönként legfeljebb 350 W-tal számolj (a tipikus fogyasztás helyett a grafikus kártya *csúcsigényét* ellenőrizd, mivel a hatékony kód sok energiát fogyaszthat). Ha a tápegységed nem felel meg az igénynek, a rendszer instabillá válhat.
1. **Ház mérete**. A GPU-k nagyok, és a kiegészítő tápcsatlakozók gyakran extra helyet igényelnek. Ezenkívül a nagyobb házak könnyebben hűthetők.
1. **GPU-hűtés**. Ha sok GPU-d van, érdemes lehet vízhűtésbe beruházni. Ezenkívül törekedj *referencia dizájnokra*, még ha kevesebb ventilátoruk is van, mivel elég vékonyak ahhoz, hogy levegő áramolhasson az eszközök között. Ha többventilátoros modellt vásárolsz, túl vastag lehet ahhoz, hogy több GPU felszerelésekor elegendő levegőt kapjon, és hőmérsékleti fojtásba futhatsz.
1. **PCIe-helyek**. Az adatok GPU-ra való mozgatása, és a GPU-k közötti adatcsere is, nagy sávszélességet igényel. 16 sávos PCIe 3.0 helyeket ajánlunk. Ha több GPU-t szerelsz be, gondosan olvasd el az alaplap leírását, hogy megbizonyosodj arról: a 16$\times$ sávszélesség akkor is rendelkezésre áll, amikor egyszerre több GPU-t használsz, és a további helyeken is PCIe 3.0-t kapsz, nem PCIe 2.0-t. Egyes alaplapok több GPU telepítésekor 8$\times$ vagy akár 4$\times$ sávszélességre csökkentik ezt. Ez részben attól függ, hány PCIe sávot kínál a CPU.

Röviden összefoglalva, íme néhány ajánlás egy deep learning szerver felépítéséhez:

* **Kezdőknek**. Vásárolj alacsony teljesítményfogyasztású, belépő szintű GPU-t, a deep learningre alkalmas olcsó játék-GPU-k 150--200 W-ot fogyasztanak. Ha szerencséd van, a jelenlegi számítógéped is támogatja.
* **1 GPU**. Egy 4 magos belépő szintű CPU elegendő lesz, és a legtöbb alaplap megfelel. Törekedj legalább 32 GB DRAM-ra, és fektess be SSD-be a helyi adathozzáférés érdekében. Egy 600 W-os tápegység elegendő lesz. Vásárolj sok ventilátorral rendelkező GPU-t.
* **2 GPU**. Egy 4-6 magos belépő szintű CPU elegendő lesz. Törekedj 64 GB DRAM-ra, és fektess be SSD-be. Két csúcskategóriás GPU-hoz körülbelül 1000 W-ra lesz szükséged. Alaplapok tekintetében győződj meg arról, hogy *két* PCIe 3.0 x16 hellyel rendelkeznek. Ha teheted, szerezz be olyan alaplapot, amelyen két szabad hely, azaz 60 mm-es távolság van a PCIe 3.0 x16 helyek között a jobb légáramlás érdekében. Ebben az esetben vásárolj két sok ventilátorral rendelkező GPU-t.
* **4 GPU**. Győződj meg arról, hogy viszonylag gyors egyszálú sebességű CPU-t vásárolsz, vagyis magas órajel-frekvenciájút. Valószínűleg szükséged lesz olyan CPU-ra, amely több PCIe sávval rendelkezik, mint például az AMD Threadripper. Valószínűleg viszonylag drága alaplapokra is szükséged lesz ahhoz, hogy 4 PCIe 3.0 x16 helyet kapj, mivel ehhez valószínűleg PLX kell a PCIe sávok multiplexeléséhez. Vásárolj referencia dizájnú GPU-kat, amelyek keskenyek, és levegőt engednek a GPU-k közé. 1600--2000 W-os tápegységre lesz szükséged, és az irodai konnektor ezt esetleg nem támogatja. Ez a szerver valószínűleg *hangos és forró* lesz. Nem akarod majd az íróasztalod alatt tartani. 128 GB DRAM ajánlott. Szerezz be SSD-t, 1--2 TB NVMe-t, a helyi tároláshoz, és néhány merevlemezt RAID-konfigurációban az adatok tárolásához.
* **8 GPU**. Dedikált multi-GPU szerverházat kell vásárolnod több redundáns tápegységgel, például 2+1-et 1600 W-os tápegységenként. Ehhez kettős foglalatú szerver-CPU-kra, 256 GB ECC DRAM-ra, gyors hálózati kártyára, 10 GBE ajánlott, lesz szükséged, és ellenőrizned kell, hogy a szerverek támogatják-e a GPU-k *fizikai formátumát*. A légáramlás és a kábelezés elhelyezése jelentősen eltér a fogyasztói és a szerveres GPU-k esetén, például RTX 2080 vs. Tesla V100. Ez azt jelenti, hogy esetleg nem tudod telepíteni a fogyasztói GPU-t egy szerverbe a tápkábel elégtelen szabad helye vagy a megfelelő kábelköteg hiánya miatt, ahogy azt az egyik társszerző fájdalmasan tapasztalta.


## GPU-k kiválasztása

Jelenleg az AMD és az NVIDIA a dedikált GPU-k két fő gyártója. Az NVIDIA volt az első, aki belépett a deep learning területére, és a CUDA révén jobb támogatást nyújt a deep learning keretrendszerekhez. Ezért a legtöbb vásárló NVIDIA GPU-kat választ.

Az NVIDIA kétféle GPU-t kínál, egyéni felhasználóknak (pl. a GTX és RTX sorozaton keresztül) és vállalati felhasználóknak (a Tesla sorozatán keresztül). A kétféle GPU összehasonlítható számítási teljesítményt nyújt. A vállalati GPU-k azonban általában (passzív) kényszerített hűtést, több memóriát és ECC (hibajavító) memóriát használnak. Ezek a GPU-k alkalmasabbak adatközpontokhoz, és általában tízszer annyiba kerülnek, mint a fogyasztói GPU-k.

Ha egy nagy vállalatnál 100+ szerverrel dolgozol, érdemes megfontolni az NVIDIA Tesla sorozatot, vagy alternatívaként a felhőben lévő GPU-s szerverek használatát. Egy laboratórium vagy kis-közepes vállalat számára 10+ szerver esetén az NVIDIA RTX sorozat valószínűleg a legköltséghatékonyabb. Vásárolhatsz Supermicro vagy Asus házzal előre konfigurált szervereket, amelyek hatékonyan befogadnak 4--8 GPU-t.

A GPU-gyártók általában másfél-két évente adnak ki új generációt, mint például a 2017-ben megjelent GTX 1000 (Pascal) sorozat és a 2019-ben megjelent RTX 2000 (Turing) sorozat. Minden sorozat számos különböző modellt kínál, amelyek különböző teljesítményszinteket nyújtanak. A GPU teljesítménye elsősorban a következő három paraméter kombinációja:

1. **Számítási teljesítmény**. Általában 32 bites lebegőpontos számítási teljesítményt keresünk. A 16 bites lebegőpontos tanítás, FP16, is egyre elterjedtebb. Ha csak az előrejelzés érdekel, 8 bites egész számokat is használhatsz. A Turing GPU-k legújabb generációja 4 bites gyorsítást kínál. Sajnos az írás idején az alacsony pontosságú hálózatok tanítására szolgáló algoritmusok még nem terjedtek el széles körben.
1. **Memória mérete**. Ahogy a modellek nagyobbak lesznek, vagy a tanítás során használt kötegek nőnek, több GPU-memóriára lesz szükséged. Nézd meg a HBM2, High Bandwidth Memory, és a GDDR6, Graphics DDR, memóriát is. A HBM2 gyorsabb, de sokkal drágább.
1. **Memória sávszélessége**. Csak akkor tudod kiaknázni a számítási teljesítményt, ha elegendő memória-sávszélességgel rendelkezel. GDDR6 használatakor keress széles memóriabuszokat.

A legtöbb felhasználó számára elegendő a számítási teljesítményt figyelni. Vedd figyelembe, hogy sok GPU különböző típusú gyorsítást kínál. Például az NVIDIA TensorCores az operátorok egy részhalmazát 5$\times$-ös sebességgel gyorsítja. Győződj meg arról, hogy a könyvtáraid támogatják ezt. A GPU-memória ne legyen kevesebb 4 GB-nál, a 8 GB pedig sokkal jobb. Próbáld elkerülni, hogy a GPU-t GUI megjelenítésére is használd, inkább a beépített grafikát használd erre. Ha ezt nem tudod elkerülni, adj hozzá extra 2 GB RAM-ot a biztonság kedvéért.

A :numref:`fig_flopsvsprice` összehasonlítja a különböző GTX 900, GTX 1000 és RTX 2000 sorozatú modellek 32 bites lebegőpontos számítási teljesítményét és árát. A javasolt árak azok, amelyeket a Wikipédián találtunk az írás idején.

![Lebegőpontos számítási teljesítmény és ár összehasonlítása. ](../img/flopsvsprice.svg)
:label:`fig_flopsvsprice`


Számos dolgot figyelhetünk meg:

1. Az egyes sorozatokon belül az ár és a teljesítmény nagyjából arányos. A Titan modellek jelentős felárral rendelkeznek a nagyobb GPU-memória előnyéért. Az újabb modellek azonban jobb költséghatékonyságot kínálnak, ahogy az a 980 Ti és az 1080 Ti összehasonlításában látható. Az RTX 2000 sorozatnál az ár nem tűnik sokat javulni. Ez azonban annak köszönhető, hogy sokkal jobb alacsony pontosságú teljesítményt (FP16, INT8 és INT4) kínálnak.
2. A GTX 1000 sorozat teljesítmény-ár aránya körülbelül kétszerese a 900-as sorozatnak.
3. Az RTX 2000 sorozatnál a teljesítmény (GFLOPs-ban) az ár *affin* függvénye.

![Lebegőpontos számítási teljesítmény és energiafogyasztás. ](../img/wattvsprice.svg)
:label:`fig_wattvsprice`


A :numref:`fig_wattvsprice` mutatja, hogy az energiafogyasztás nagyjából lineárisan skálázódik a számítás mennyiségével. Másodszor, a későbbi generációk hatékonyabbak. Ezt látszólag cáfolja az RTX 2000 sorozatnak megfelelő grafikon. Ez azonban a TensorCores következménye, amelyek aránytalanul sok energiát fogyasztanak.


## Összefoglalás

* Figyelj a teljesítményre, a PCIe busz sávjaira, a CPU egyszálú sebességére és a hűtésre szerver építésekor.
* Ha lehet, a legújabb GPU-generációt vásárold meg.
* Nagyméretű telepítésekhez használd a felhőt.
* A nagy sűrűségű szerverek esetleg nem kompatibilisek minden GPU-val. Vásárlás előtt ellenőrizd a mechanikai és hűtési specifikációkat.
* A magas hatékonyság érdekében használj FP16-ot vagy alacsonyabb pontosságot.


[Discussions](https://discuss.d2l.ai/t/425)
