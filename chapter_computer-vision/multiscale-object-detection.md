# Többléptékű objektumdetektálás
:label:`sec_multiscale-object-detection`


A :numref:`sec_anchor` fejezetben a bemeneti kép minden pixelét középpontként véve több horgonydobozt generáltunk.
Lényegében ezek a horgonydobozok a kép különböző régióinak mintái.
Azonban ha *minden* pixelre generáljuk őket, a kiszámítandó horgonydobozok száma túl naggyá válhat.
Gondoljunk egy $561 \times 728$ pixeles bemeneti képre.
Ha minden pixelhez öt különböző alakú horgonydobozt generálunk középpontként,
több mint kétmillió horgonydobozt ($561 \times 728 \times 5$) kell felcímkézni és megjósolni a képen.

## Többléptékű horgonydobozok
:label:`subsec_multiscale-anchor-boxes`

Belátható, hogy nem nehéz csökkenteni a horgonydobozok számát a képen.
Például egyszerűen egyenletesen mintavételezhetünk néhány pixelt a bemeneti képből, és ezeket középpontként véve generálhatunk horgonydobozokat.
Emellett különböző léptékeken különböző számú és különböző méretű horgonydobozt is generálhatunk.
Intuitív módon a kisebb objektumok valószínűbben jelennek meg a képen, mint a nagyobbak.
Példaként: $1 \times 1$, $1 \times 2$ és $2 \times 2$ méretű objektumok rendre 4, 2 és 1 lehetséges módon jelenhetnek meg egy $2 \times 2$ méretű képen.
Ezért kisebb horgonydobozokkal kisebb objektumokat felismerve több régiót mintavételezhetünk, míg nagyobb objektumoknál kevesebb régiót.

Annak bemutatásához, hogyan generálunk horgonydobozokat több léptéken, olvassunk be egy képet.
Magassága és szélessége rendre 561 és 728 pixel.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import image, np, npx

npx.set_np()

img = image.imread('../img/catdog.jpg')
h, w = img.shape[:2]
h, w
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

img = d2l.plt.imread('../img/catdog.jpg')
h, w = img.shape[:2]
h, w
```

Emlékeztetjük, hogy a :numref:`sec_conv_layer` fejezetben egy konvolúciós réteg kétdimenziós tömb kimenetét jellemzőtérképnek neveztük.
A jellemzőtérkép alakjának meghatározásával meghatározhatjuk az egyenletesen mintavételezett horgonydobozok középpontjait bármely képen.


A `display_anchors` függvényt az alábbiakban definiáljuk.
[**A jellemzőtérképen (`fmap`) minden egységet (pixelt) horgonydoboz-középpontként véve generálunk horgonydobozokat (`anchors`).**]
Mivel a horgonydobozokban (`anchors`) az $(x, y)$-tengelykoordináta értékeket elosztják a jellemzőtérkép (`fmap`) szélességével és magasságával,
ezek az értékek 0 és 1 között vannak,
jelezve a horgonydobozok relatív pozícióit a jellemzőtérképen.

Mivel a horgonydobozok (`anchors`) középpontjai a jellemzőtérkép (`fmap`) összes egységén el vannak osztva,
ezeknek a középpontoknak *egyenletesen* kell eloszlaniuk bármely bemeneti képen
relatív térbeli pozícióik szempontjából.
Konkrétabban, adott a jellemzőtérkép `fmap_w` szélességével és `fmap_h` magasságával,
a következő függvény *egyenletesen* mintavételezi
a pixeleket `fmap_h` sorból és `fmap_w` oszlopból
bármely bemeneti képen.
Az egyenletesen mintavételezett pixeleken középpontként,
`s` méretű (feltéve, hogy az `s` lista hossza 1) és különböző képarányú (`ratios`) horgonydobozok generálódnak.

```{.python .input}
#@tab mxnet
def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # Az első két dimenzió értékei nem befolyásolják a kimenetet
    fmap = np.zeros((1, 10, fmap_h, fmap_w))
    anchors = npx.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = np.array((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img.asnumpy()).axes,
                    anchors[0] * bbox_scale)
```

```{.python .input}
#@tab pytorch
def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # Az első két dimenzió értékei nem befolyásolják a kimenetet
    fmap = d2l.zeros((1, 10, fmap_h, fmap_w))
    anchors = d2l.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = d2l.tensor((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img).axes,
                    anchors[0] * bbox_scale)
```

Először [**vegyük figyelembe a kis objektumok felismerését**].
Annak érdekében, hogy megjelenítéskor könnyebb legyen megkülönböztetni, a különböző középpontú horgonydobozok itt nem fedik át egymást:
a horgonydoboz mérete 0.15-re van állítva, a jellemzőtérkép magassága és szélessége pedig 4. Láthatjuk, hogy a kép 4 sorában és 4 oszlopában lévő horgonydobozok középpontjai egyenletesen oszlanak el.

```{.python .input}
#@tab all
display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
```

Folytatjuk [**a jellemzőtérkép magasságának és szélességének felére csökkentésével, és nagyobb horgonydobozok használatával nagyobb objektumok felismeréséhez**]. Amikor a méret 0.4-re van állítva, néhány horgonydoboz átfedésbe kerülhet egymással.

```{.python .input}
#@tab all
display_anchors(fmap_w=2, fmap_h=2, s=[0.4])
```

Végül [**tovább csökkentjük a jellemzőtérkép magasságát és szélességét felére, és a horgonydoboz méretét 0.8-ra növeljük**]. Most a horgonydoboz középpontja a kép középpontja.

```{.python .input}
#@tab all
display_anchors(fmap_w=1, fmap_h=1, s=[0.8])
```

## Többléptékű felismerés


Mivel többléptékű horgonydobozokat generáltunk, ezeket különböző léptékeken különböző méretű objektumok felismerésére fogjuk használni.
A következőkben bemutatunk egy konvolúciós neurális hálózaton alapuló többléptékű objektumdetektálási módszert, amelyet a :numref:`sec_ssd` fejezetben implementálunk.

Bizonyos léptéken tegyük fel, hogy $c$ jellemzőtérképünk van $h \times w$ alakban.
A :numref:`subsec_multiscale-anchor-boxes` fejezetben ismertetett módszer alkalmazásával
$hw$ horgonydoboz-halmazt generálunk,
ahol minden halmaz $a$ azonos középpontú horgonydobozt tartalmaz.
Például a :numref:`subsec_multiscale-anchor-boxes` fejezetben végzett kísérletek első léptékén,
adott tíz (csatornaszám) $4 \times 4$-es jellemzőtérképből
16 horgonydoboz-halmazt generáltunk,
ahol minden halmaz 3 azonos középpontú horgonydobozt tartalmaz.
Ezután minden horgonydobozt a valódi befoglaló téglalapok alapján osztállyal és eltolással látnak el. Az aktuális léptéken az objektumdetektálási modellnek meg kell jósolni a bemeneti képen lévő $hw$ horgonydoboz-halmaz osztályait és eltolásait, ahol a különböző halmazoknak különböző középpontjaik vannak.


Feltéve, hogy a $c$ jellemzőtérkép itt
a konvolúciós neurális hálózat előreterjesztésével a bemeneti kép alapján kapott közbenső kimenetek.
Mivel minden jellemzőtérképen $hw$ különböző térbeli pozíció van,
ugyanaz a térbeli pozíció $c$ egységgel rendelkezőnek tekinthető.
A :numref:`sec_conv_layer` fejezetben bemutatott receptív mező definíciója szerint
a jellemzőtérképek ugyanazon térbeli pozíciójában lévő $c$ egységnek
ugyanaz a receptív mezeje van a bemeneti képen:
ugyanabban a receptív mezőben képviselik a bemeneti kép információit.
Ezért a jellemzőtérképek ugyanazon térbeli pozíciójának $c$ egységét átalakíthatjuk az ezen térbeli pozíció segítségével generált $a$ horgonydoboz osztályaivá és eltolásaivá.
Lényegében a bemeneti kép egy bizonyos receptív mezőben lévő információit használjuk arra, hogy megjósoljuk a bemeneti képen ahhoz a receptív mezőhöz közel elhelyezkedő horgonydobozok osztályait és eltolásait.


Amikor a különböző rétegeken lévő jellemzőtérképeknek különböző méretű receptív mezőik vannak a bemeneti képen, felhasználhatók különböző méretű objektumok felismerésére.
Például tervezhetünk egy neurális hálózatot, amelyben a kimeneti réteghez közelebb lévő jellemzőtérképek egységeinek szélesebb receptív mezői vannak, így nagyobb objektumokat tudnak felismerni a bemeneti képből.

Röviden összefoglalva, a mély neurális hálózatok segítségével kihasználhatjuk a képek rétegenkénti, több szintű reprezentációit a többléptékű objektumdetektáláshoz.
Megmutatjuk, hogyan működik ez egy konkrét példán keresztül a :numref:`sec_ssd` fejezetben.


## Összefoglalás

* Több léptéken különböző méretű horgonydobozokat generálhatunk különböző méretű objektumok felismeréséhez.
* A jellemzőtérképek alakjának meghatározásával meghatározhatjuk az egyenletesen mintavételezett horgonydobozok középpontjait bármely képen.
* A bemeneti kép egy bizonyos receptív mezőben lévő információit a bemeneti képen ahhoz a receptív mezőhöz közel elhelyezkedő horgonydobozok osztályainak és eltolásainak megjóslására használjuk.
* mélytanulás segítségével kihasználhatjuk a képek rétegenkénti, több szintű reprezentációit a többléptékű objektumdetektáláshoz.


## Feladatok

1. A :numref:`sec_alexnet` fejezetben folytatott megbeszéléseink alapján a mély neurális hálózatok hierarchikus jellemzőket tanulnak egyre növekvő absztrakciós szinteken a képekhez. A többléptékű objektumdetektálásban a különböző léptékeken lévő jellemzőtérképek különböző absztrakciós szinteknek felelnek meg? Miért vagy miért nem?
1. A :numref:`subsec_multiscale-anchor-boxes` fejezetben végzett kísérletek első léptékén (`fmap_w=4, fmap_h=4`) generálj egyenletesen elosztott, esetleg átfedő horgonydobozokat.
1. Adott egy $1 \times c \times h \times w$ alakú jellemzőtérkép változó, ahol $c$, $h$ és $w$ rendre a jellemzőtérképek csatornáinak száma, magassága és szélessége. Hogyan alakítható át ez a változó a horgonydobozok osztályaivá és eltolásaivá? Mi a kimenet alakja?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/371)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1607)
:end_tab:
