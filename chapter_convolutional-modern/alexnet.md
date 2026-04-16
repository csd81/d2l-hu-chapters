```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Mély Konvolúciós Neurális Hálózatok (AlexNet)
:label:`sec_alexnet`


Bár a CNN-ek jól ismertek voltak a számítógépes látás és a gépi tanulás közösségében a LeNet :cite:`LeCun.Jackel.Bottou.ea.1995` bevezetése óta, nem uralták azonnal a területet.
Bár a LeNet korai kis adathalmazokon jó eredményeket ért el, a CNN-ek nagyobb, valószerűbb adathalmazokon való tanításának teljesítménye és megvalósíthatósága még nem volt bizonyított.
Valójában az 1990-es évek elejétől a 2012-es áttörő eredményekig :cite:`Krizhevsky.Sutskever.Hinton.2012` a neurális hálózatokat gyakran felülmúlták más gépi tanulási módszerek, mint például a kernel módszerek :cite:`Scholkopf.Smola.2002`, az ensemble módszerek :cite:`Freund.Schapire.ea.1996` és a strukturált becslés :cite:`Taskar.Guestrin.Koller.2004`.

A számítógépes látás terén ez az összehasonlítás talán nem teljesen helyes.
Bár a konvolúciós hálózatok bemenetei nyers vagy enyhén feldolgozott (pl. középre igazított) pixelértékekből állnak, a szakemberek soha nem tápláltak nyers pixeleket a hagyományos modellekbe.
Ehelyett a tipikus számítógépes látási feldolgozóláncok kézileg megtervezett jellemzőkinyerő csővezetékekből álltak, mint például SIFT :cite:`Lowe.2004`, SURF :cite:`Bay.Tuytelaars.Van-Gool.2006` és vizuális szavak zsákjai :cite:`Sivic.Zisserman.2003`.
A jellemzőket nem *tanulták*, hanem *kézzel alkották meg*.
A legtöbb fejlődés egyrészt az okosabb jellemzőkinyerési ötletekből, másrészt a geometria mély ismeretéből :cite:`Hartley.Zisserman.2000` fakadt. A tanulási algoritmust sokszor csak mellékesnek tekintették.

Bár az 1990-es években már léteztek neurális hálózat-gyorsítók, ezek még nem voltak elég erősek ahhoz, hogy sok paramétert tartalmazó, mély, többcsatornás, többrétegű CNN-eket futtassanak. Például az NVIDIA 1999-es GeForce 256-os chipje másodpercenként legfeljebb 480 millió lebegőpontos műveletet (MFLOPS) tudott elvégezni, és nem állt rendelkezésre értelmes programozási keretrendszer a játékokon túli műveletekhez. A mai gyorsítók eszközönként meghaladják az 1000 TFLOPs teljesítményt.
Ráadásul az adathalmazok viszonylag kicsik voltak: az OCR 60 000 alacsony felbontású, $28 \times 28$ pixeles képen rendkívül nehéz feladatnak számított.
Mindezekhez hozzájárult, hogy a neurális hálózatok tanításának kulcstechnikái — köztük a paraméter-inicializálási heurisztikák :cite:`Glorot.Bengio.2010`, a sztochasztikus gradienscsökkenés okos változatai :cite:`Kingma.Ba.2014`, a nem összenyomó aktivációs függvények :cite:`Nair.Hinton.2010` és a hatékony regularizációs technikák :cite:`Srivastava.Hinton.Krizhevsky.ea.2014` — még hiányoztak.

Így az *end-to-end* (pixeltől a besorolásig) rendszerek tanítása helyett a klasszikus feldolgozóláncok inkább így néztek ki:

1. Szerezz be egy érdekes adathalmazt. A korai időkben ezek drága érzékelőket igényeltek. Az 1994-es [Apple QuickTake 100](https://en.wikipedia.org/wiki/Apple_QuickTake) például egy hatalmasnak számító 0,3 megapixeles (VGA) felbontással büszkélkedhetett, és legfeljebb 8 képet tudott tárolni, mindezt 1000 dollárért.
1. Dolgozd fel az adathalmazt kézzel kialakított jellemzőkkel, optikai, geometriai és más analitikai ismeretek, valamint olykor szerencsés doktoranduszok véletlenszerű felfedezései alapján.
1. Vezesd az adatokat szabványos jellemzőkinyerőkön keresztül, mint a SIFT (skálainvariáns jellemzőtranszformáció) :cite:`Lowe.2004`, a SURF (gyorsított robusztus jellemzők) :cite:`Bay.Tuytelaars.Van-Gool.2006` vagy más kézzel hangolt csővezetékek. Az OpenCV még ma is biztosít SIFT-kinyerőket!
1. Az eredményül kapott reprezentációkat táplálj be kedvenc osztályozódba — valószínűleg egy lineáris modellbe vagy kernel módszerbe — a tanításhoz.

Ha gépi tanulással foglalkozó kutatókkal beszéltél, azt mondták volna, hogy a gépi tanulás fontos és szép.
Elegáns elméletek bizonyították a különböző osztályozók tulajdonságait :cite:`boucheron2005theory`, és a konvex optimalizálás :cite:`Boyd.Vandenberghe.2004` vált a fő eszközzé ezek előállításához.
A gépi tanulás területe virágzott, szigorú volt és rendkívül hasznos. Ha azonban számítógépes látással foglalkozó kutatókkal beszéltél, egészen más történetet hallottál.
A képfelismerés szennyes igazsága — mondták — az, hogy a jellemzők, a geometria :cite:`Hartley.Zisserman.2000,hartley2009global` és a mérnöki megközelítés, nem pedig az új tanulási algoritmusok, hajtották előre a fejlődést.
A számítógépes látás kutatói joggal hitték, hogy egy valamivel nagyobb vagy tisztább adathalmaz vagy valamivel jobb jellemzőkinyerési csővezeték sokkal nagyobb hatással van a végső pontosságra, mint bármelyik tanulási algoritmus.

```{.python .input  n=2}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, init, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input  n=4}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

## Reprezentációs Tanulás

A helyzet másképp is megfogalmazható: a csővezeték legfontosabb eleme a reprezentáció volt.
2012-ig a reprezentációt többnyire mechanikusan számolták ki.
Valójában az újabb jellemzőfüggvény-készletek megtervezése, az eredmények javítása és a módszer leírása mind hangsúlyos szerepet kapott a cikkekben.
A SIFT :cite:`Lowe.2004`, a SURF :cite:`Bay.Tuytelaars.Van-Gool.2006`, a HOG (orientált gradiens hisztogramok) :cite:`Dalal.Triggs.2005`, a vizuális szavak zsákjai :cite:`Sivic.Zisserman.2003` és hasonló jellemzőkinyerők uralták a területet.

Egy másik kutatócsoport — köztük Yann LeCun, Geoff Hinton, Yoshua Bengio, Andrew Ng, Shun-ichi Amari és Juergen Schmidhuber — más terveket dédelgetett.
Ők úgy vélték, hogy maguknak a jellemzőknek kell tanulhatóknak lenniük.
Sőt, azt hitték, hogy a kellően összetett jellemzőket hierarchikusan kell felépíteni, közösen tanult rétegek sorozatával, amelyek mindegyike tanulható paraméterekkel rendelkezik.
Egy kép esetén az alacsonyabb rétegek az éleket, a színeket és a textúrákat detektálnák, hasonlóan ahhoz, ahogy az állatok vizuális rendszere feldolgozza a bemenetet. Különösen az olyan vizuális jellemzők automatikus tervezése, mint amelyeket a ritka kódolás :cite:`olshausen1996emergence` által nyernek, nyitott kihívás maradt a modern CNN-ek megjelenéséig.
Csak :citet:`Dean.Corrado.Monga.ea.2012,le2013building` munkái után nyert jelentős teret az a gondolat, hogy a jellemzőket automatikusan állítsuk elő képadatokból.

Az első modern CNN :cite:`Krizhevsky.Sutskever.Hinton.2012`, amelyet az egyik feltalálójáról, Alex Krizhevsky-ről neveztek el *AlexNet*-nek, nagyrészt evolúciós fejlesztés a LeNet felett. A 2012-es ImageNet-kihíváson kiváló teljesítményt ért el.

![Az AlexNet első rétege által tanult képszűrők. Reprodukció: :citet:`Krizhevsky.Sutskever.Hinton.2012`.](../img/filters.png)
:width:`400px`
:label:`fig_filters`

Érdekesség, hogy a hálózat alacsonyabb rétegeiben a modell olyan jellemzőkinyerőket tanult, amelyek hasonlítottak néhány hagyományos szűrőre.
A :numref:`fig_filters` ábra az alacsonyabb szintű képleírókat mutatja.
A hálózat magasabb rétegei ezekre a reprezentációkra építve nagyobb struktúrákat, például szemeket, orrokat, fűszálakat és hasonlókat képesek ábrázolni.
Még magasabb rétegek egész objektumokat — embereket, repülőket, kutyákat vagy frizbiket — képesek ábrázolni.
Végül az utolsó rejtett állapot a kép kompakt reprezentációját tanulja meg, amely összefoglalja a tartalmát, így a különböző kategóriákhoz tartozó adatok könnyen elkülöníthetők.

Az AlexNet (2012) és elődje, a LeNet (1995) sok architektúrális elemet oszt. Ez felveti a kérdést: miért tartott ennyi ideig? A fő különbség az volt, hogy az előző két évtizedben az elérhető adat és számítási kapacitás jelentősen megnövekedett. Ezért az AlexNet jóval nagyobb volt: sokkal több adaton és sokkal gyorsabb GPU-kon tanították, mint amilyen CPU-k 1995-ben elérhetők voltak.

### Hiányzó összetevő: Adatok

A sok réteggel rendelkező mély modellek nagy mennyiségű adatot igényelnek ahhoz, hogy abba a tartományba kerüljenek, ahol jelentősen felülmúlják a konvex optimalizáláson alapuló hagyományos módszereket (pl. lineáris és kernel módszerek).
Az 1990-es évek korlátozott tárolókapacitása, az (képalkotó) érzékelők viszonylag magas ára és a szorosabb kutatási keretek miatt azonban a legtöbb kutatás apró adathalmazokra támaszkodott.
Számos cikk az UCI adathalmaz-gyűjteményre támaszkodott, amelyek közül sok csak több száz vagy (néhány) ezer alacsony felbontású, mesterségesen tiszta hátterű képet tartalmazott.

2009-ben kiadták az ImageNet adathalmazt :cite:`Deng.Dong.Socher.ea.2009`, amely arra buzdította a kutatókat, hogy 1 millió példából, 1000 különböző objektumkategóriánként 1000-ből tanuljanak modelleket. A kategóriák maguk a WordNet :cite:`Miller.1995` legnépszerűbb főnévi csomópontjain alapultak.
Az ImageNet-csapat a Google Képkeresőt használta az egyes kategóriák nagy jelölthalmazainak előszűrésére, és az Amazon Mechanical Turk crowdsourcing platformját alkalmazta annak megerősítésére, hogy az egyes képek a hozzájuk rendelt kategóriához tartoznak.
Ez a skála példátlan volt, más adathalmazokat több mint egy nagyságrenddel felülmúlva (pl. a CIFAR-100 60 000 képet tartalmaz). Egy másik szempont az volt, hogy a képek viszonylag magas, $224 \times 224$ pixeles felbontásúak voltak, ellentétben a 80 milliós TinyImages adathalmazzal :cite:`Torralba.Fergus.Freeman.2008`, amely $32 \times 32$ pixeles bélyegképekből állt.
Ez lehetővé tette a magasabb szintű jellemzők kialakítását.
A kapcsolódó verseny, az ImageNet Large Scale Visual Recognition Challenge :cite:`russakovsky2015imagenet`, előrevitte a számítógépes látás és a gépi tanulás kutatását, és arra ösztönözte a kutatókat, hogy azonosítsák, mely modellek teljesítenek a legjobban olyan méretarányban, amelyet a tudományos szféra korábban nem mérlegelt. A legnagyobb látási adathalmazok, mint a LAION-5B :cite:`schuhmann2022laion`, milliárdnyi képet tartalmaznak további metaadatokkal.

### Hiányzó összetevő: Hardver

A deep learning modellek mohó fogyasztói a számítási ciklusoknak.
A tanítás száz epoch-on is keresztül mehet, és minden iterációban az adatokat sok, számítási szempontból drága lineáris algebrai műveletet tartalmazó rétegen kell átvezetni.
Ez az egyik fő oka annak, hogy az 1990-es és a 2000-es évek elején a hatékonyabban optimalizált, konvex célok alapján működő egyszerűbb algoritmusokat részesítették előnyben.

A *grafikus feldolgozó egységek* (GPU-k) játékmegváltoztató tényezőnek bizonyultak a deep learning megvalósíthatóvá tételében.
Ezeket a chipeket eredetileg a számítógépes játékok grafikus feldolgozásának gyorsítására fejlesztették ki.
Különösen a nagy áteresztőképességű $4 \times 4$-es mátrix–vektor szorzatokra optimalizálták őket, amelyekre számos számítógépes grafikai feladatban szükség van.
Szerencsére a matematika meglepően hasonló a konvolúciós rétegek kiszámításához szükségeshez.
Ebben az időszakban az NVIDIA és az ATI elkezdte optimalizálni a GPU-kat általános célú számítási műveletekre :cite:`Fernando.2004`, egészen odáig menve, hogy *általános célú GPU-k* (GPGPU-k) néven forgalmazták őket.

Az összehasonlítás kedvéért gondoljuk át egy modern mikroprocesszor (CPU) magját.
Minden egyes mag viszonylag erős, nagy órajelen fut, és nagy gyorsítótárakkal rendelkezik (akár több megabájtos L3).
Minden egyes mag alkalmas sokféle utasítás végrehajtására: ágprediktáló, mély csővezeték, specializált végrehajtó egységek, spekulatív végrehajtás és sok más kiegészítő funkció teszi lehetővé, hogy sokféle, bonyolult vezérlési folyamatú programot futtasson.
Ez a látszólagos erősség azonban egyben az Achilles-sarka is: az általános célú magok nagyon drágán gyárthatók. Kiválóak általános célú kódban, amely sok vezérlési folyamatot tartalmaz.
Ehhez sok chip-területre van szükség, nem csak a tényleges ALU (aritmetikai logikai egység) számára, ahol a számítás történik, hanem az összes említett kiegészítő funkcióhoz, plusz a memória interfészekhez, a magok közötti gyorsítótár-logikához, a nagy sebességű összekapcsolásokhoz stb. A CPU-k viszonylag rosszak bármely egyes feladatban, ha dedikált hardverrel vetjük össze őket.
A modern laptopoknak 4–8 magjuk van, és még a csúcsszerverek is ritkán haladják meg a foglalatonkénti 64 magot, mert egyszerűen nem gazdaságos.

Ezzel szemben a GPU-k több ezer kis feldolgozó elemből állhatnak (az NVIDIA legújabb Ampere chipjeinek akár 6912 CUDA magjuk is lehet), amelyeket általában nagyobb csoportokba (az NVIDIA ezeket warp-nak nevezi) szerveznek.
A részletek némileg eltérnek az NVIDIA, az AMD, az ARM és más chipgyártók között. Bár minden egyes mag viszonylag gyenge, kb. 1 GHz-es órajelen fut, az ilyen magok összesített száma teszi a GPU-kat nagyságrendekkel gyorsabbá a CPU-knál.
Például az NVIDIA legújabb Ampere A100 GPU-ja chipenkénti 300 TFLOPs feletti teljesítményt kínál a specializált 16 bites pontosságú (BFLOAT16) mátrix-mátrix szorzatokhoz, és akár 20 TFLOPs-ot az általánosabb célú lebegőpontos műveletekhez (FP32).
Ugyanakkor a CPU-k lebegőpontos teljesítménye ritkán haladja meg az 1 TFLOPs-t. Például az Amazon Graviton 3 csúcs-teljesítménye 2 TFLOPs a 16 bites precíziós műveletekhez, ami hasonló az Apple M1 processzor GPU-teljesítményéhez.

Számos oka van annak, hogy a GPU-k FLOPs szempontjából sokkal gyorsabbak a CPU-knál.
Először is, a fogyasztás az órajellel *négyzetes arányban* nő.
Így egy négyszer gyorsabb CPU-maghoz szükséges teljesítményen 16 GPU-magot lehet működtetni $\frac{1}{4}$-es sebességen, ami $16 \times \frac{1}{4} = 4$-szeres teljesítményt eredményez.
Másodszor, a GPU-magok sokkal egyszerűbbek (valójában hosszú ideig még *nem is voltak képesek* általános célú kód végrehajtására), ami energiahatékonyabbá teszi őket. Például (i) általában nem támogatják a spekulatív kiértékelést, (ii) jellemzően nem lehet minden egyes feldolgozó elemet egyénileg programozni, és (iii) az egyes magok gyorsítótárai jóval kisebbek szoktak lenni.
Végül a deep learning számos művelete nagy memória-sávszélességet igényel.
A GPU-k itt is jeleskednek, busszal, amely legalább 10-szer szélesebb, mint sok CPU-é.

Visszatérve 2012-höz: a nagy áttörés akkor következett be, amikor Alex Krizhevsky és Ilya Sutskever implementált egy mély CNN-t, amely GPU-kon futott.
Rájöttek, hogy a CNN-ek számítási szűk keresztmetszetei — a konvolúciók és a mátrixszorzatok — mind hardveresen párhuzamosítható műveletek.
Két NVIDIA GTX 580-as GPU-val, egyenként 3 GB memóriával (amelyek mindegyike 1,5 TFLOPs teljesítményre volt képes, ami még egy évtizeddel később is kihívást jelent a legtöbb CPU számára), gyors konvolúciókat implementáltak.
A [cuda-convnet](https://code.google.com/archive/p/cuda-convnet/) kód olyan jó volt, hogy több éven át iparági szabványnak számított, és a deep learning boom első néhány évét hajtotta.

## AlexNet

Az AlexNet, amely egy 8 rétegű CNN-t alkalmazott, nagy különbséggel nyerte meg a 2012-es ImageNet Large Scale Visual Recognition Challenge-t :cite:`Russakovsky.Deng.Huang.ea.2013`.
Ez a hálózat elsőként mutatta meg, hogy a tanulással nyert jellemzők felülmúlhatják a kézzel tervezett jellemzőket, megdöntve a számítógépes látás korábbi paradigmáját.

Az AlexNet és a LeNet architektúrája meglepően hasonlít egymásra, ahogy azt a :numref:`fig_alexnet` ábra szemlélteti.
Megjegyezzük, hogy az AlexNet egy kissé egyszerűsített változatát mutatjuk be, amelyből elhagytuk azokat a tervezési különlegességeket, amelyek 2012-ben szükségesek voltak ahhoz, hogy a modell elférjen két kis GPU-n.

![A LeNet-től (balra) az AlexNet-ig (jobbra).](../img/alexnet.svg)
:label:`fig_alexnet`

Az AlexNet és a LeNet között jelentős különbségek is vannak.
Először is, az AlexNet sokkal mélyebb, mint a viszonylag kis LeNet-5.
Az AlexNet nyolc rétegből áll: öt konvolúciós rétegből, két teljesen összekötött rejtett rétegből és egy teljesen összekötött kimeneti rétegből.
Másodszor, az AlexNet a sigmoid helyett a ReLU-t használja aktivációs függvényként. Nézzük meg a részleteket!

### Architektúra

Az AlexNet első rétegében a konvolúciós ablak alakja $11\times11$.
Mivel az ImageNet képei nyolcszor magasabbak és szélebbek, mint az MNIST képek, az ImageNet adatokban az objektumok több pixelt foglalnak el, több vizuális részlettel.
Ezért nagyobb konvolúciós ablakra van szükség az objektum megragadásához.
A második réteg konvolúciós ablakának alakja $5\times5$-re csökken, ezt követi a $3\times3$.
Ezenkívül az első, második és ötödik konvolúciós réteg után a hálózat max-pooling rétegeket ad hozzá $3\times3$-as ablakmérettel és 2-es lépésközzel.
Ráadásul az AlexNet tízszer több konvolúciós csatornával rendelkezik, mint a LeNet.

Az utolsó konvolúciós réteg után két hatalmas, teljesen összekötött réteg következik 4096 kimenettel.
Ezek a rétegek majdnem 1 GB modellparamétert igényelnek.
A korai GPU-k korlátozott memóriája miatt az eredeti AlexNet kettős adatfolyam-tervezést alkalmazott, így minden GPU felelős volt a modell felének tárolásáért és kiszámításáért.
Szerencsére a GPU-memória ma már viszonylag bőséges, ezért manapság ritkán kell modelleket GPU-k között felosztani (az AlexNet modellünk ebben a tekintetben eltér az eredeti cikktől).

### Aktivációs Függvények

Az AlexNet a sigmoid aktivációs függvényt egy egyszerűbb ReLU aktivációs függvényre cserélte. Egyrészt a ReLU aktivációs függvény számítása egyszerűbb — például nem tartalmaz a sigmoid aktivációs függvényben szereplő exponenciálási műveletet.
Másrészt a ReLU aktivációs függvény megkönnyíti a modelltanítást különböző paraméter-inicializálási módszerek esetén. Ennek oka, hogy amikor a sigmoid aktivációs függvény kimenete nagyon közel van a 0-hoz vagy az 1-hez, ezeknek a tartományoknak a gradiense majdnem 0, így a visszaterjesztés nem tudja frissíteni a modell néhány paraméterét. Ezzel szemben a ReLU aktivációs függvény gradiense a pozitív tartományban mindig 1 (:numref:`subsec_activation-functions`). Ezért ha a modell paraméterei nincsenek megfelelően inicializálva, a sigmoid függvény a pozitív tartományban majdnem 0 gradienst kaphat, ami azt jelenti, hogy a modell nem tanítható hatékonyan.

### Kapacitásvezérlés és Előfeldolgozás

Az AlexNet a teljesen összekötött réteg modellkomplexitását dropout segítségével szabályozza (:numref:`sec_dropout`), míg a LeNet csak súlycsökkentést (weight decay) alkalmaz.
Az adatok még tovább augmentálása érdekében az AlexNet tanítási ciklusa rengeteg képaugmentálást adott hozzá, például tükrözést, vágást és szín módosításokat.
Ez robusztusabbá teszi a modellt, és a nagyobb mintaméret hatékonyan csökkenti a túlillesztést.
Lásd :citet:`Buslaev.Iglovikov.Khvedchenya.ea.2020` alapos áttekintését az ilyen előfeldolgozási lépésekről.

```{.python .input  n=5}
%%tab pytorch, mxnet, tensorflow
class AlexNet(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Sequential()
            self.net.add(
                nn.Conv2D(96, kernel_size=11, strides=4, activation='relu'),
                nn.MaxPool2D(pool_size=3, strides=2),
                nn.Conv2D(256, kernel_size=5, padding=2, activation='relu'),
                nn.MaxPool2D(pool_size=3, strides=2),
                nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
                nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
                nn.Conv2D(256, kernel_size=3, padding=1, activation='relu'),
                nn.MaxPool2D(pool_size=3, strides=2),
                nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
                nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
                nn.Dense(num_classes))
            self.net.initialize(init.Xavier())
        if tab.selected('pytorch'):
            self.net = nn.Sequential(
                nn.LazyConv2d(96, kernel_size=11, stride=4, padding=1),
                nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2),
                nn.LazyConv2d(256, kernel_size=5, padding=2), nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
                nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
                nn.LazyConv2d(256, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
                nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(p=0.5),
                nn.LazyLinear(4096), nn.ReLU(),nn.Dropout(p=0.5),
                nn.LazyLinear(num_classes))
            self.net.apply(d2l.init_cnn)
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4,
                                       activation='relu'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
                tf.keras.layers.Conv2D(filters=256, kernel_size=5, padding='same',
                                       activation='relu'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
                tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',
                                       activation='relu'),
                tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',
                                       activation='relu'),
                tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same',
                                       activation='relu'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(4096, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(4096, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(num_classes)])
```

```{.python .input}
%%tab jax
class AlexNet(d2l.Classifier):
    lr: float = 0.1
    num_classes: int = 10
    training: bool = True

    def setup(self):
        self.net = nn.Sequential([
            nn.Conv(features=96, kernel_size=(11, 11), strides=4, padding=1),
            nn.relu,
            lambda x: nn.max_pool(x, window_shape=(3, 3), strides=(2, 2)),
            nn.Conv(features=256, kernel_size=(5, 5)),
            nn.relu,
            lambda x: nn.max_pool(x, window_shape=(3, 3), strides=(2, 2)),
            nn.Conv(features=384, kernel_size=(3, 3)), nn.relu,
            nn.Conv(features=384, kernel_size=(3, 3)), nn.relu,
            nn.Conv(features=256, kernel_size=(3, 3)), nn.relu,
            lambda x: nn.max_pool(x, window_shape=(3, 3), strides=(2, 2)),
            lambda x: x.reshape((x.shape[0], -1)),  # simítás
            nn.Dense(features=4096),
            nn.relu,
            nn.Dropout(0.5, deterministic=not self.training),
            nn.Dense(features=4096),
            nn.relu,
            nn.Dropout(0.5, deterministic=not self.training),
            nn.Dense(features=self.num_classes)
        ])
```

[**Egységnyi csatornájú adatpéldát hozunk létre**] 224-es magassággal és szélességgel (**az egyes rétegek kimeneti alakjának megfigyeléséhez**). Ez megfelel az AlexNet architektúrájának a :numref:`fig_alexnet` ábrán.

```{.python .input  n=6}
%%tab pytorch, mxnet
AlexNet().layer_summary((1, 1, 224, 224))
```

```{.python .input  n=7}
%%tab tensorflow
AlexNet().layer_summary((1, 224, 224, 1))
```

```{.python .input}
%%tab jax
AlexNet(training=False).layer_summary((1, 224, 224, 1))
```

## Tanítás

Bár az AlexNet-et :citet:`Krizhevsky.Sutskever.Hinton.2012` az ImageNet-en tanították, itt a Fashion-MNIST-et használjuk, mivel egy ImageNet-modell konvergenciáig való tanítása akár órákat vagy napokat is igénybe vehet modern GPU-n is.
Az AlexNet közvetlen alkalmazásának egyik problémája a [**Fashion-MNIST**]-en az, hogy (**a képek felbontása alacsonyabb**) ($28 \times 28$ pixel), (**mint az ImageNet képeké.**)
Hogy a dolgok működjenek, (**felskálázzuk őket $224 \times 224$-re**).
Ez általában nem okos megközelítés, mivel csak növeli a számítási bonyolultságot anélkül, hogy információt adna hozzá. Mindazonáltal itt megtesszük, hogy hűek maradjunk az AlexNet architektúrájához.
Ezt az átméretezést a `d2l.FashionMNIST` konstruktor `resize` argumentumával végezzük.

Most [**elindíthatjuk az AlexNet tanítását.**]
A :numref:`sec_lenet`-beli LeNet-hez képest a fő változás itt a kisebb tanulási ráta alkalmazása és a sokkal lassabb tanítás a mélyebb és szélesebb hálózat, a nagyobb képfelbontás és a drágább konvolúciók miatt.

```{.python .input  n=8}
%%tab pytorch, mxnet, jax
model = AlexNet(lr=0.01)
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
trainer.fit(model, data)
```

```{.python .input  n=9}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
with d2l.try_gpu():
    model = AlexNet(lr=0.01)
    trainer.fit(model, data)
```

## Vita

Az AlexNet szerkezete feltűnően hasonlít a LeNet-re, számos kritikus fejlesztéssel mind a pontosság (dropout), mind a tanítás könnyűsége (ReLU) terén. Ugyanilyen meglepő a deep learning eszközkészlete terén elért fejlődés mértéke. Ami 2012-ben több hónap munkát igényelt, az ma bármely modern keretrendszerrel egy tucat kódsorral elvégezhető.

Az architektúrát áttekintve látható, hogy az AlexNetnek van egy gyenge pontja a hatékonyság terén: az utolsó két rejtett réteg $6400 \times 4096$-os és $4096 \times 4096$-os méretű mátrixokat igényel. Ez 164 MB memóriának és 81 MFLOPs számítási költségnek felel meg, ami nem elhanyagolható kiadás, különösen kisebb eszközökön, mint például a mobiltelefonok. Ez az egyik oka annak, hogy az AlexNet-et sokkal hatékonyabb architektúrák váltották fel, amelyeket a következő szakaszokban tárgyalunk. Mindazonáltal kulcsfontosságú lépés a sekély hálózatoktól a ma használt mély hálózatokig. Megjegyezzük, hogy bár a paraméterek száma messze meghaladja a kísérleteinkben lévő tanítóadatok mennyiségét (az utolsó két rétegnek több mint 40 millió paramétere van, amelyeket 60 ezer képből álló adathalmazon tanítanak), alig tapasztalható túlillesztés: a tanítási és validációs veszteség a tanítás során szinte azonos. Ez a modern mély hálózati tervezésben rejlő jobb regularizációnak, például a dropoutnak köszönhető.

Bár úgy tűnik, hogy az AlexNet implementációja csak néhány sorral hosszabb a LeNet-énél, az akadémiai közösségnek sok évébe telt befogadni ezt a fogalmi változást és kiaknázni kiváló kísérleti eredményeit. Ez részben a hatékony számítási eszközök hiányának is köszönhető. Akkor még nem létezett sem a DistBelief :cite:`Dean.Corrado.Monga.ea.2012`, sem a Caffe :cite:`Jia.Shelhamer.Donahue.ea.2014`, és a Theano :cite:`Bergstra.Breuleux.Bastien.ea.2010` is nélkülözött még sok megkülönböztető funkciót. A TensorFlow :cite:`Abadi.Barham.Chen.ea.2016` elérhetővé válása döntő mértékben megváltoztatta a helyzetet.

## Feladatok

1. Folytatva a fenti megbeszélést, elemezd az AlexNet számítási tulajdonságait.
    1. Számítsd ki a konvolúciók és a teljesen összekötött rétegek memóriaigényét. Melyik dominál?
    1. Számítsd ki a konvolúciók és a teljesen összekötött rétegek számítási költségét.
    1. Hogyan befolyásolja a memória (olvasási és írási sávszélesség, késleltetés, méret) a számítást? Van-e különbség a tanítás és a következtetés hatásai között?
1. Chiptervező vagy, és el kell döntened, hogyan oszd el az erőforrásokat a számítás és a memória sávszélesség között. Például egy gyorsabb chip több energiát és esetleg nagyobb chip-területet igényel. Több memória sávszélesség több csatlakozót és vezérlési logikát igényel, így szintén több területet. Hogyan optimalizálsz?
1. Miért nem végeznek a mérnökök teljesítményteszteket az AlexNet-en?
1. Próbáld meg növelni az epoch-ok számát az AlexNet tanításakor. Hogyan különböznek az eredmények a LeNet-hez képest? Miért?
1. Az AlexNet talán túl összetett a Fashion-MNIST adathalmazhoz, különösen a kezdeti képek alacsony felbontása miatt.
    1. Próbáld egyszerűsíteni a modellt a tanítás gyorsítása érdekében, miközben biztosítod, hogy a pontosság ne csökkenjen jelentősen.
    1. Tervezz jobb modellt, amely közvetlenül $28 \times 28$-as képeken működik.
1. Módosítsd a batch méretet, és figyeld meg az áteresztőképesség (képek/s), a pontosság és a GPU memória változásait.
1. Alkalmazz dropoutot és ReLU-t a LeNet-5-re. Javul-e? Tovább javíthatsz-e az előfeldolgozással, hogy kihasználd a képekben rejlő invarianciákat?
1. Lehet-e az AlexNet-et túlilleszteni? Melyik jellemzőt kell eltávolítani vagy megváltoztatni a tanítás megzavarásához?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/75)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/76)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/276)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18001)
:end_tab:
