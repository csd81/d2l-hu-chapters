# Geometria és lineáris algebrai műveletek
:label:`sec_geometry-linear-algebraic-ops`

A :numref:`sec_linear-algebra` fejezetben megismerkedtünk a lineáris algebra alapjaival,
és láttuk, hogyan lehet ezeket felhasználni az adatok átalakítására szolgáló általános műveletek leírásához.
A lineáris algebra az egyik legfontosabb matematikai alappillér,
amelyre a mélytanulás és tágabb értelemben a gépi tanulás munkájának nagy része épül.
Míg a :numref:`sec_linear-algebra` fejezet elegendő eszközt tartalmazott
a modern mélytanulás modellek mechanikájának bemutatásához,
a témának ennél jóval több oldala is van.
Ebben a szakaszban mélyebbre ásunk:
kiemeljük a lineáris algebrai műveletek néhány geometriai értelmezését,
és bevezetünk néhány alapfogalmat, köztük a sajátértékeket és a sajátvektorokat.

## A vektorok geometriája
Először a vektorok két elterjedt geometriai értelmezéséről kell szólnunk:
a vektort tekinthetjük pontnak vagy iránynak a térben.
Alapvetően egy vektor számok listája, mint az alábbi Python-lista.

```{.python .input}
#@tab all
v = [1, 7, 0, 1]
```

A matematikusok ezt leggyakrabban *oszlop-* vagy *sorvektorként* írják le, azaz vagy

$$
\mathbf{x} = \begin{bmatrix}1\\7\\0\\1\end{bmatrix},
$$

vagy

$$
\mathbf{x}^\top = \begin{bmatrix}1 & 7 & 0 & 1\end{bmatrix}.
$$

Ezeknek általában eltérő értelmezésük van:
az adatpéldányokat oszlopvektorként,
a súlyozott összegek kialakításához használt súlyokat sorvektorként szokás kezelni.
Ugyanakkor rugalmasnak lenni hasznos lehet.
Ahogy a :numref:`sec_linear-algebra` fejezetben leírtuk,
bár egy önálló vektor alapértelmezett iránya az oszlopvektor,
táblázatos adathalmazt tároló mátrix esetén
elfogadottabb minden adatpéldányt sorvektorként kezelni
a mátrixban.

Egy vektornak az első értelmezése, amelyet adnunk kell,
a tér egy pontjaként való felfogás.
Két vagy három dimenzióban ezeket a pontokat vizualizálhatjuk,
ha a vektorok összetevőit felhasználjuk a pontok helyzetének meghatározásához
egy rögzített referenciához képest, amelyet *origónak* nevezünk. Ez látható a :numref:`fig_grid` ábrán.

![Illusztráció a vektorok mint síkbeli pontok vizualizálásához. A vektor első összetevője az $\mathit{x}$-koordinátát, a második összetevő az $\mathit{y}$-koordinátát adja meg. Magasabb dimenziókban ez analóg módon működik, bár sokkal nehezebb vizualizálni.](../img/grid-points.svg)
:label:`fig_grid`

Ez a geometriai nézőpont lehetővé teszi, hogy a problémát elvontabb szinten vizsgáljuk.
Ahelyett, hogy egy látszólag leküzdhetetlen feladattal – például képek macskákká vagy kutyákká való besorolásával – néznénk szembe,
elvontan kezdhetünk gondolkodni a feladatokról
mint térbeli ponthalmazokról, és elképzelhetjük a célt
mint két különálló pontcsoport szétválasztásának felfedezését.

Ezzel párhuzamosan létezik a vektorok egy másik, szintén elterjedt nézőpontja:
a tér irányaiként való értelmezésük.
A $\mathbf{v} = [3,2]^\top$ vektort nemcsak
az origótól jobbra $3$, felfelé $2$ egységre lévő pontnak tekinthetjük,
hanem magának az iránynak is – amely szerint $3$ lépést kell jobbra és $2$ lépést felfelé tenni.
Ily módon a :numref:`fig_arrow` ábra összes vektorát azonosnak tekintjük.

![Bármely vektor nyílként ábrázolható a síkban. Ebben az esetben minden rajzolt vektor a $(3,2)^\top$ vektor egy-egy ábrázolása.](../img/par-vec.svg)
:label:`fig_arrow`

Ennek a váltásnak az egyik előnye,
hogy a vektorösszeadás aktusát vizuálisan is értelmezhetjük.
Konkrétan: követjük az egyik vektor által kijelölt irányt,
majd a másikét, ahogyan azt a :numref:`fig_add-vec` ábra mutatja.

![A vektorösszeadást úgy vizualizálhatjuk, hogy először az egyik vektort követjük, majd a másikat.](../img/vec-add.svg)
:label:`fig_add-vec`

A vektorkivonásnak hasonló értelmezése van.
A $\mathbf{u} = \mathbf{v} + (\mathbf{u}-\mathbf{v})$ azonosságból kiindulva
látjuk, hogy a $\mathbf{u}-\mathbf{v}$ vektor az az irány,
amely a $\mathbf{v}$ pontból a $\mathbf{u}$ pontba vezet.


## Skaláris szorzat és szögek
Ahogy a :numref:`sec_linear-algebra` fejezetben láttuk,
ha adott két $\mathbf{u}$ és $\mathbf{v}$ oszlopvektor,
akkor skaláris szorzatuk kiszámítható az alábbi módon:

$$\mathbf{u}^\top\mathbf{v} = \sum_i u_i\cdot v_i.$$
:eqlabel:`eq_dot_def`

Mivel a :eqref:`eq_dot_def` szimmetrikus, tükrözni fogjuk
a klasszikus szorzás jelölését, és így írjuk:

$$
\mathbf{u}\cdot\mathbf{v} = \mathbf{u}^\top\mathbf{v} = \mathbf{v}^\top\mathbf{u},
$$

kiemelve, hogy a vektorok sorrendjének felcserélése ugyanolyan eredményt ad.

A :eqref:`eq_dot_def` skaláris szorzatnak geometriai értelmezése is van: szorosan összefügg két vektor szögével. Tekintsük a :numref:`fig_angle` ábrán látható szöget.

![Bármely két síkbeli vektor között jól meghatározott $\theta$ szög adódik. Látni fogjuk, hogy ez a szög szorosan kapcsolódik a skaláris szorzathoz.](../img/vec-angle.svg)
:label:`fig_angle`

Kezdésként vizsgáljunk meg két konkrét vektort:

$$
\mathbf{v} = (r,0) \; \textrm{and} \; \mathbf{w} = (s\cos(\theta), s \sin(\theta)).
$$

A $\mathbf{v}$ vektor $r$ hosszú és párhuzamos az $x$-tengellyel,
a $\mathbf{w}$ vektor $s$ hosszú és $\theta$ szöget zár be az $x$-tengellyel.
Ha kiszámítjuk e két vektor skaláris szorzatát, azt kapjuk, hogy

$$
\mathbf{v}\cdot\mathbf{w} = rs\cos(\theta) = \|\mathbf{v}\|\|\mathbf{w}\|\cos(\theta).
$$

Néhány egyszerű algebrai átalakítással a tagokat átrendezve kapjuk:

$$
\theta = \arccos\left(\frac{\mathbf{v}\cdot\mathbf{w}}{\|\mathbf{v}\|\|\mathbf{w}\|}\right).
$$

Röviden: e két konkrét vektornál
a skaláris szorzat és a normák együttesen megadják a két vektor által bezárt szöget. Ez az állítás általánosan érvényes. A levezetést itt nem végezzük el, azonban
ha $\|\mathbf{v} - \mathbf{w}\|^2$-t kétféleképpen írjuk fel –
egyszer a skaláris szorzat segítségével, egyszer geometriailag a koszinusz-tétel alapján –,
megkapjuk a teljes összefüggést.
Valóban, bármely két $\mathbf{v}$ és $\mathbf{w}$ vektorra
a köztük lévő szög:

$$\theta = \arccos\left(\frac{\mathbf{v}\cdot\mathbf{w}}{\|\mathbf{v}\|\|\mathbf{w}\|}\right).$$
:eqlabel:`eq_angle_forumla`

Ez szép eredmény, mivel a számításban semmi nem hivatkozik kétdimenziós térre.
Valóban, háromban vagy akár hárommillió dimenzióban is gond nélkül alkalmazható.

Egyszerű példaként nézzük meg, hogyan számítható ki két vektor szöge:

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mxnet import gluon, np, npx
npx.set_np()

def angle(v, w):
    return np.arccos(v.dot(w) / (np.linalg.norm(v) * np.linalg.norm(w)))

angle(np.array([0, 1, 2]), np.array([2, 3, 4]))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import torch
from torchvision import transforms
import torchvision

def angle(v, w):
    return torch.acos(v.dot(w) / (torch.norm(v) * torch.norm(w)))

angle(torch.tensor([0, 1, 2], dtype=torch.float32), torch.tensor([2.0, 3, 4]))
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import tensorflow as tf

def angle(v, w):
    return tf.acos(tf.tensordot(v, w, axes=1) / (tf.norm(v) * tf.norm(w)))

angle(tf.constant([0, 1, 2], dtype=tf.float32), tf.constant([2.0, 3, 4]))
```

Most még nem lesz szükségünk rá, de érdemes tudni,
hogy azokat a vektorokat, amelyek szöge $\pi/2$
(azaz $90^{\circ}$), *ortogonálisnak* nevezzük.
A fenti egyenletet vizsgálva látjuk, hogy ez akkor teljesül, ha $\theta = \pi/2$,
ami megegyezik azzal, hogy $\cos(\theta) = 0$.
Ez csakis akkor lehetséges, ha maga a skaláris szorzat nulla,
és két vektor pontosan akkor ortogonális, ha $\mathbf{v}\cdot\mathbf{w} = 0$.
Ez a képlet hasznosnak bizonyul az objektumok geometriai megértésekor.

Jogos a kérdés: miért hasznos a szög kiszámítása?
A válasz abban a fajta invarianciában rejlik, amelyet az adatoktól elvárunk.
Képzeljük el, hogy van egy kép és egy másolata,
amelyen minden pixelérték ugyanolyan, de az eredeti fényesség $10\%$-a.
Az egyes pixelek értékei általában messze vannak az eredeti értékektől.
Ezért ha kiszámítanánk az eredeti kép és a sötétebb kép közötti távolságot,
az nagy lehet.
Azonban a legtöbb gépi tanulási alkalmazásban a *tartalom* ugyanaz – egy macska/kutya osztályozó számára ez még mindig egy macska képe.
Ha viszont a szöget vizsgáljuk, könnyen belátható,
hogy bármely $\mathbf{v}$ vektorra a $\mathbf{v}$ és a $0.1\cdot\mathbf{v}$ közötti szög nulla.
Ez annak felel meg, hogy a vektorok skálázása
megtartja az irányt, és csak a hosszt változtatja meg.
A szög alapján a sötétebb kép azonosnak minősül.

Az ilyen példák mindenütt megtalálhatók.
Szöveg esetén előfordulhat, hogy a tárgyalt témának nem szabad megváltoznia,
ha kétszer olyan hosszú dokumentumot írunk ugyanarról a dologról.
Bizonyos kódolások esetén (például a szavak előfordulásainak megszámlálásánál egy szótárban)
ez a dokumentumot kódoló vektor megkettőzésének felel meg,
így itt is használhatjuk a szöget.

### Koszinusz-hasonlóság
Azokban a gépi tanulási kontextusokban, ahol a szöget
két vektor közelségének mérésére alkalmazzák,
a szakemberek a *koszinusz-hasonlóság* kifejezést használják
a következő mennyiségre:
$$
\cos(\theta) = \frac{\mathbf{v}\cdot\mathbf{w}}{\|\mathbf{v}\|\|\mathbf{w}\|}.
$$

A koszinusz értéke $1$, ha a két vektor azonos irányba mutat,
$-1$, ha ellentétes irányba mutatnak,
és $0$, ha a két vektor ortogonális.
Fontos megjegyezni: ha a nagy dimenziójú vektorok összetevőit
véletlenszerűen, $0$ várható értékkel mintavételezzük,
koszinuszuk majdnem mindig közel lesz $0$-hoz.


## Hipersíkok

A vektorokkal való munka mellett egy másik kulcsfontosságú objektum,
amelyet a lineáris algebrában el kell sajátítani,
a *hipersík*: magasabb dimenziókba általánosítása
az egyenesnek (két dimenzióban) vagy a síknak (három dimenzióban).
Egy $d$ dimenziós vektortérben egy hipersíknak $d-1$ dimenziója van,
és a teret két félrészre osztja.

Kezdjük egy példával.
Tegyük fel, hogy adott a $\mathbf{w}=[2,1]^\top$ oszlopvektor. Arra keressük a választ: „Melyek azok a $\mathbf{v}$ pontok, amelyekre $\mathbf{w}\cdot\mathbf{v} = 1$?"
A skaláris szorzat és a szögek közötti fenti összefüggés felidézésével (:eqref:`eq_angle_forumla`),
látjuk, hogy ez ekvivalens az alábbi feltétellel:
$$
\|\mathbf{v}\|\|\mathbf{w}\|\cos(\theta) = 1 \; \iff \; \|\mathbf{v}\|\cos(\theta) = \frac{1}{\|\mathbf{w}\|} = \frac{1}{\sqrt{5}}.
$$

![A trigonometria felidézésével belátjuk, hogy a $\|\mathbf{v}\|\cos(\theta)$ kifejezés a $\mathbf{v}$ vektor $\mathbf{w}$ irányára vetített vetítésének hossza.](../img/proj-vec.svg)
:label:`fig_vector-project`

Ha megvizsgáljuk ennek a kifejezésnek a geometriai jelentését,
látjuk, hogy ez ekvivalens azzal,
hogy a $\mathbf{v}$ vetülete a $\mathbf{w}$ irányára pontosan $1/\|\mathbf{w}\|$ hosszú,
ahogyan az a :numref:`fig_vector-project` ábrán látható.
Azon pontok halmaza, amelyekre ez teljesül, egy egyenes,
amely merőleges a $\mathbf{w}$ vektorra.
Ha akarnánk, megkereshetnénk ennek az egyenesnek az egyenletét,
és megkapnánk, hogy az $2x + y = 1$, azaz $y = 1 - 2x$.

Ha most megvizsgáljuk, mi történik, amikor azokat a pontokat keressük, amelyekre
$\mathbf{w}\cdot\mathbf{v} > 1$ vagy $\mathbf{w}\cdot\mathbf{v} < 1$,
azt látjuk, hogy ezek azok az esetek, amelyekben a vetületek
rendre hosszabbak vagy rövidebbek $1/\|\mathbf{w}\|$-nél.
Így a két egyenlőtlenség az egyenes két oldalát határolja.
Ily módon egy módszert találtunk arra, hogy a terünket két részre osszuk:
az egyik oldalon lévő összes pontnak a skaláris szorzata a küszöb alatt van,
a másik oldalon pedig felette, amint az a :numref:`fig_space-division` ábrán látható.

![Ha most a kifejezés egyenlőtlenség-változatát vizsgáljuk, látjuk, hogy hipersíkunk (ebben az esetben: egyszerűen egy egyenes) két félre osztja a teret.](../img/space-division.svg)
:label:`fig_space-division`

Magasabb dimenzióban az elképzelés lényegében ugyanez.
Ha most $\mathbf{w} = [1,2,3]^\top$ és
azokat a háromdimenziós pontokat keressük, amelyekre $\mathbf{w}\cdot\mathbf{v} = 1$,
akkor egy, a $\mathbf{w}$ vektorra merőleges síkot kapunk.
A két egyenlőtlenség ismét a sík két oldalát határolja,
amint az a :numref:`fig_higher-division` ábrán látható.

![A hipersíkok bármely dimenzióban két félre osztják a teret.](../img/space-division-3d.svg)
:label:`fig_higher-division`

Bár a vizualizálási képességeink itt véget érnek,
semmi sem akadályoz meg minket abban, hogy ezt tíz, száz vagy akár milliárd dimenzióban alkalmazzuk.
Ez az igény gépileg tanult modelleknél gyakran felmerül.
Például a lineáris osztályozási modelleket –
mint a :numref:`sec_softmax` fejezetben bemutatottakat –
értelmezhetjük olyan módszerekként, amelyek megtalálják a különböző célcsoportokat elválasztó hipersíkokat.
Ebben a kontextusban az ilyen hipersíkokat gyakran *döntési síkoknak* is nevezik.
A mélytanulás alapú osztályozási modellek többsége egy softmax-szal táplált lineáris réteggel végződik,
ezért a mély neurális hálózat szerepét úgy is értelmezhetjük,
mint egy nemlineáris reprezentáció megtalálását, amelynek segítségével a célcsoportok
hipersíkok által tisztán szétválaszthatók.

Kézzel készített példaként figyeljük meg, hogy ésszerű modellt kaphatunk
a Fashion-MNIST adathalmaz (lásd: :numref:`sec_fashion_mnist`) apró pólóképeinek és nadrágképeinek osztályozásához,
ha pusztán a két átlag közötti vektort vesszük a döntési sík meghatározásához,
és szemre becsülünk egy nyers küszöbértéket. Először betöltjük az adatokat és kiszámítjuk az átlagokat.

```{.python .input}
#@tab mxnet
# Az adathalmaz betöltése
train = gluon.data.vision.FashionMNIST(train=True)
test = gluon.data.vision.FashionMNIST(train=False)

X_train_0 = np.stack([x[0] for x in train if x[1] == 0]).astype(float)
X_train_1 = np.stack([x[0] for x in train if x[1] == 1]).astype(float)
X_test = np.stack(
    [x[0] for x in test if x[1] == 0 or x[1] == 1]).astype(float)
y_test = np.stack(
    [x[1] for x in test if x[1] == 0 or x[1] == 1]).astype(float)

# Átlagok kiszámítása
ave_0 = np.mean(X_train_0, axis=0)
ave_1 = np.mean(X_train_1, axis=0)
```

```{.python .input}
#@tab pytorch
# Az adathalmaz betöltése
trans = []
trans.append(transforms.ToTensor())
trans = transforms.Compose(trans)
train = torchvision.datasets.FashionMNIST(root="../data", transform=trans,
                                          train=True, download=True)
test = torchvision.datasets.FashionMNIST(root="../data", transform=trans,
                                         train=False, download=True)

X_train_0 = torch.stack(
    [x[0] * 256 for x in train if x[1] == 0]).type(torch.float32)
X_train_1 = torch.stack(
    [x[0] * 256 for x in train if x[1] == 1]).type(torch.float32)
X_test = torch.stack(
    [x[0] * 256 for x in test if x[1] == 0 or x[1] == 1]).type(torch.float32)
y_test = torch.stack([torch.tensor(x[1]) for x in test
                      if x[1] == 0 or x[1] == 1]).type(torch.float32)

# Átlagok kiszámítása
ave_0 = torch.mean(X_train_0, axis=0)
ave_1 = torch.mean(X_train_1, axis=0)
```

```{.python .input}
#@tab tensorflow
# Az adathalmaz betöltése
((train_images, train_labels), (
    test_images, test_labels)) = tf.keras.datasets.fashion_mnist.load_data()


X_train_0 = tf.cast(tf.stack(train_images[[i for i, label in enumerate(
    train_labels) if label == 0]] * 256), dtype=tf.float32)
X_train_1 = tf.cast(tf.stack(train_images[[i for i, label in enumerate(
    train_labels) if label == 1]] * 256), dtype=tf.float32)
X_test = tf.cast(tf.stack(test_images[[i for i, label in enumerate(
    test_labels) if label == 0]] * 256), dtype=tf.float32)
y_test = tf.cast(tf.stack(test_images[[i for i, label in enumerate(
    test_labels) if label == 1]] * 256), dtype=tf.float32)

# Átlagok kiszámítása
ave_0 = tf.reduce_mean(X_train_0, axis=0)
ave_1 = tf.reduce_mean(X_train_1, axis=0)
```

Tanulságos lehet ezeket az átlagokat részletesen megvizsgálni, ezért ábrázoljuk, hogyan néznek ki. Ebben az esetben azt látjuk, hogy az átlag valóban egy elmosódott póló képére hasonlít.

```{.python .input}
#@tab mxnet, pytorch
# Az átlagos póló ábrázolása
d2l.set_figsize()
d2l.plt.imshow(ave_0.reshape(28, 28).tolist(), cmap='Greys')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
# Az átlagos póló ábrázolása
d2l.set_figsize()
d2l.plt.imshow(tf.reshape(ave_0, (28, 28)), cmap='Greys')
d2l.plt.show()
```

A második esetben szintén azt látjuk, hogy az átlag egy elmosódott nadrág képére hasonlít.

```{.python .input}
#@tab mxnet, pytorch
# Az átlagos nadrág ábrázolása
d2l.plt.imshow(ave_1.reshape(28, 28).tolist(), cmap='Greys')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
# Az átlagos nadrág ábrázolása
d2l.plt.imshow(tf.reshape(ave_1, (28, 28)), cmap='Greys')
d2l.plt.show()
```

Egy teljesen gépileg tanult megoldás esetén a küszöbértéket az adathalmazból tanulnánk meg. Ebben az esetben én egyszerűen szemre becsültem egy küszöbértéket, amely a tanítási adatokon jónak tűnt.

```{.python .input}
#@tab mxnet
# A teszthalmaz pontosságának kiíratása szemre becsült küszöbértékkel
w = (ave_1 - ave_0).T
predictions = X_test.reshape(2000, -1).dot(w.flatten()) > -1500000

# Pontosság
np.mean(predictions.astype(y_test.dtype) == y_test, dtype=np.float64)
```

```{.python .input}
#@tab pytorch
# A teszthalmaz pontosságának kiíratása szemre becsült küszöbértékkel
w = (ave_1 - ave_0).T
# Az `@` operátor a mátrixszorzás jele a PyTorchban.
predictions = X_test.reshape(2000, -1) @ (w.flatten()) > -1500000

# Pontosság
torch.mean((predictions.type(y_test.dtype) == y_test).float(), dtype=torch.float64)
```

```{.python .input}
#@tab tensorflow
# A teszthalmaz pontosságának kiíratása szemre becsült küszöbértékkel
w = tf.transpose(ave_1 - ave_0)
predictions = tf.reduce_sum(X_test * tf.nest.flatten(w), axis=0) > -1500000

# Pontosság
tf.reduce_mean(
    tf.cast(tf.cast(predictions, y_test.dtype) == y_test, tf.float32))
```

## Lineáris transzformációk geometriája

A :numref:`sec_linear-algebra` fejezet és a fenti tárgyalás alapján
jól értjük a vektorok, hosszak és szögek geometriáját.
Van azonban egy fontos objektum, amelyről még nem szóltunk,
mégpedig a mátrixok által képviselt lineáris transzformációk geometriai értelmezése. Teljes mértékben elsajátítani, hogy a mátrixok hogyan képesek átalakítani az adatokat
két esetlegesen eltérő magas dimenziós tér között, jelentős gyakorlást igényel,
és meghaladja e függelék kereteit.
Azonban két dimenzióban már el lehet kezdeni fejleszteni az intuíciót.

Tegyük fel, hogy adott egy mátrix:

$$
\mathbf{A} = \begin{bmatrix}
a & b \\ c & d
\end{bmatrix}.
$$

Ha ezt egy tetszőleges $\mathbf{v} = [x, y]^\top$ vektorra alkalmazzuk,
szorozzuk meg, és azt kapjuk, hogy

$$
\begin{aligned}
\mathbf{A}\mathbf{v} & = \begin{bmatrix}a & b \\ c & d\end{bmatrix}\begin{bmatrix}x \\ y\end{bmatrix} \\
& = \begin{bmatrix}ax+by\\ cx+dy\end{bmatrix} \\
& = x\begin{bmatrix}a \\ c\end{bmatrix} + y\begin{bmatrix}b \\d\end{bmatrix} \\
& = x\left\{\mathbf{A}\begin{bmatrix}1\\0\end{bmatrix}\right\} + y\left\{\mathbf{A}\begin{bmatrix}0\\1\end{bmatrix}\right\}.
\end{aligned}
$$

Ez talán furcsának tűnő számításnak látszik,
ahol valami világos valamelyest átláthatatlanná válik.
Mégis azt mondja nekünk, hogy le tudjuk írni,
hogyan transzformál egy mátrix *bármely* vektort,
*két konkrét vektor* – a $[1,0]^\top$ és a $[0,1]^\top$ – transzformációja segítségével.
Ezt érdemes egy pillanatra végiggondolni.
Lényegében egy végtelen problémát (mi történik bármely valós számpárral)
redukáltuk egy végesre (mi történik ezekkel a konkrét vektorokkal).
Ezek a vektorok egy *bázis* példái,
amellyel segítségével terünk bármely vektora felírható
ezen *bázisvektorok* súlyozott összegeként.

Rajzoljuk le, mi történik, ha a konkrét

$$
\mathbf{A} = \begin{bmatrix}
1 & 2 \\
-1 & 3
\end{bmatrix}
$$

mátrixot alkalmazzuk.

Ha megnézzük a konkrét $\mathbf{v} = [2, -1]^\top$ vektort,
látjuk, hogy ez $2\cdot[1,0]^\top + -1\cdot[0,1]^\top$,
ezért tudjuk, hogy az $A$ mátrix ezt a következőbe viszi:
$2(\mathbf{A}[1,0]^\top) + -1(\mathbf{A}[0,1])^\top = 2[1, -1]^\top - [2,3]^\top = [0, -5]^\top$.
Ha gondosan végiggondoljuk ezt a logikát –
például az összes egész koordinátájú pont rácsát vizsgálva –,
látjuk, hogy a mátrixszorzás nyírhatja, forgathatja és skálázhatja a rácsot,
de a rácsstruktúra megmarad, ahogyan az a :numref:`fig_grid-transform` ábrán látható.

![Az $\mathbf{A}$ mátrix hat a megadott bázisvektorokra. Figyeljük meg, hogyan mozdul el az egész rács együtt.](../img/grid-transform.svg)
:label:`fig_grid-transform`

Ez a legfontosabb intuitív pont,
amelyet a mátrixok által képviselt lineáris transzformációkról el kell sajátítani.
A mátrixok képtelenek a tér egyes részeit másképp torzítani, mint más részeit.
Mindössze annyit tehetnek, hogy a tér eredeti koordinátáit
nyírják, forgatják és skálázzák.

Egyes torzítások igen erőteljesek lehetnek. Például a

$$
\mathbf{B} = \begin{bmatrix}
2 & -1 \\ 4 & -2
\end{bmatrix}
$$

mátrix az egész kétdimenziós síkot egyetlen egyenessé zsugorítja.
Az ilyen transzformációk azonosítása és kezelése egy későbbi szakasz témája,
de geometriailag láthatjuk, hogy ez alapvetően különbözik
a fent látott transzformációktól.
Például az $\mathbf{A}$ mátrix eredménye „visszahajlítható" az eredeti rácsra. A $\mathbf{B}$ mátrix eredménye nem,
mert soha nem tudhatjuk, honnan jött a $[1,2]^\top$ vektor – vajon $[1,1]^\top$-ból vagy $[0, -1]^\top$-ból?

Bár ezt a képet egy $2\times2$-es mátrixhoz mutattuk,
semmi sem akadályoz meg minket abban, hogy a tanultakat magasabb dimenziókba is átvigyük.
Ha hasonló bázisvektorokat veszünk, mint a $[1,0, \ldots,0]$,
és megnézzük, hova viszi ezeket a mátrix,
kezdhetünk érzéket kapni arra, hogyan torzítja a mátrixszorzás
az egész teret – akármilyen dimenziójú térről legyen szó.

## Lineáris függőség

Vizsgáljuk meg ismét a

$$
\mathbf{B} = \begin{bmatrix}
2 & -1 \\ 4 & -2
\end{bmatrix}
$$

mátrixot.

Ez az egész síkot az $y = 2x$ egyenesen élő egyenessé zsugorítja.
Most felmerül a kérdés: van-e mód arra, hogy ezt csupán a mátrix vizsgálatával felismerjük?
A válasz igenlő.
Legyen $\mathbf{b}_1 = [2,4]^\top$ és $\mathbf{b}_2 = [-1, -2]^\top$
a $\mathbf{B}$ két oszlopa.
Emlékezzünk, hogy minden, a $\mathbf{B}$ mátrix által átalakított vektor
felírható a mátrix oszlopainak súlyozott összegeként:
például $a_1\mathbf{b}_1 + a_2\mathbf{b}_2$.
Ezt *lineáris kombinációnak* nevezzük.
Az a tény, hogy $\mathbf{b}_1 = -2\cdot\mathbf{b}_2$,
azt jelenti, hogy a két oszlop bármely lineáris kombinációja
kifejezhető kizárólag $\mathbf{b}_2$ segítségével, mivel

$$
a_1\mathbf{b}_1 + a_2\mathbf{b}_2 = -2a_1\mathbf{b}_2 + a_2\mathbf{b}_2 = (a_2-2a_1)\mathbf{b}_2.
$$

Ez azt jelenti, hogy az egyik oszlop bizonyos értelemben felesleges,
mivel nem határoz meg egyedi irányt a térben.
Ez nem igazán lephet meg minket,
hiszen már láttuk, hogy ez a mátrix az egész síkot egyetlen egyenessé lapítja.
Ráadásul látjuk, hogy a $\mathbf{b}_1 = -2\cdot\mathbf{b}_2$ lineáris függőség tükrözi ezt.
A két vektor közötti szimmetria érdekében ezt így írjuk:

$$
\mathbf{b}_1  + 2\cdot\mathbf{b}_2 = 0.
$$

Általánosan: azt mondjuk, hogy a $\mathbf{v}_1, \ldots, \mathbf{v}_k$ vektorok halmaza *lineárisan függő*,
ha léteznek olyan $a_1, \ldots, a_k$ együtthatók, amelyek *nem mindegyike nulla*, és amelyekre

$$
\sum_{i=1}^k a_i\mathbf{v_i} = 0.
$$

Ebben az esetben az egyik vektort ki tudjuk fejezni a többiek valamely kombinációjaként,
és lényegében feleslegessé válik.
Ezért a mátrix oszlopai közötti lineáris függőség
annak tanúbizonysága, hogy mátrixunk
a teret valamilyen alacsonyabb dimenzióba sűríti.
Ha nincs lineáris függőség, azt mondjuk, hogy a vektorok *lineárisan függetlenek*.
Ha egy mátrix oszlopai lineárisan függetlenek,
nem következik be tömörítés, és a művelet visszafordítható.

## Rang

Ha adott egy általános $n\times m$-es mátrix,
ésszerű a kérdés: milyen dimenziójú térbe képezi le a mátrix a vektorokat?
Erre az ún. *rang* fogalma ad választ.
Az előző szakaszban megjegyeztük, hogy a lineáris függőség
a tér alacsonyabb dimenzióba való tömörítésének tanúbizonysága,
és ezt felhasználhatjuk a rang fogalmának meghatározásához.
Konkrétan: az $\mathbf{A}$ mátrix rangja
az oszlopok azon részhalmazai közül a legnagyobb méretű lineárisan független részhalmaz mérete. Például a

$$
\mathbf{B} = \begin{bmatrix}
2 & 4 \\ -1 & -2
\end{bmatrix}
$$

mátrixra $\textrm{rang}(B)=1$, mivel a két oszlop lineárisan függő,
de bármelyik oszlop önmagában nem az.
Nehezebb példaként tekintsük a következőt:

$$
\mathbf{C} = \begin{bmatrix}
1& 3 & 0 & -1 & 0 \\
-1 & 0 & 1 & 1 & -1 \\
0 & 3 & 1 & 0 & -1 \\
2 & 3 & -1 & -2 & 1
\end{bmatrix},
$$

és mutassuk meg, hogy $\mathbf{C}$ rangja kettő, mivel például
az első két oszlop lineárisan független,
de a három oszlopból alkotott bármely négyes kombináció már függő.

Ez az eljárás a leírás szerint nagyon nem hatékony.
Megköveteli a mátrix összes oszlopkombinációjának vizsgálatát,
így az oszlopok számában potenciálisan exponenciális.
Később megismerünk egy számításilag hatékonyabb módszert
a mátrix rangjának meghatározására, de egyelőre
ennyi elegendő ahhoz, hogy lássuk: a fogalom jól definiált, és megértsük a jelentését.

## Invertálhatóság

Fentebb láttuk, hogy lineárisan függő oszlopú mátrixszal való szorzás
nem fordítható vissza, azaz nincs olyan inverz művelet, amely mindig vissza tudja állítani a bemenetet. Azonban teljes rangú mátrixszal való szorzás
(azaz valamely $n \times n$-es, $n$ rangú $\mathbf{A}$ mátrixszal) mindig visszafordítható. Tekintsük az

$$
\mathbf{I} = \begin{bmatrix}
1 & 0 & \cdots & 0 \\
0 & 1 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 1
\end{bmatrix}
$$

mátrixot,

amelynek főátlójában egyes, máshol nullás értékek szerepelnek.
Ezt *egységmátrixnak* nevezzük.
Ez az a mátrix, amely alkalmazva nem változtatja meg az adatainkat.
Ahhoz, hogy megtaláljuk azt a mátrixot, amely visszacsinája az $\mathbf{A}$ mátrix hatását,
keresünk egy $\mathbf{A}^{-1}$ mátrixot, amelyre

$$
\mathbf{A}^{-1}\mathbf{A} = \mathbf{A}\mathbf{A}^{-1} =  \mathbf{I}.
$$

Ha ezt egyenletrendszerként tekintjük, $n \times n$ ismeretlenünk van
(az $\mathbf{A}^{-1}$ elemei) és $n \times n$ egyenletünk
(a szükséges egyenlőség az $\mathbf{A}^{-1}\mathbf{A}$ szorzat és az $\mathbf{I}$ minden eleme között),
ezért általánosan várhatjuk, hogy létezik megoldás.
Valóban, a következő szakaszban megismerünk egy *determináns* nevű mennyiséget,
amelynek az a tulajdonsága, hogy amíg a determináns nem nulla, megoldás létezik. Az ilyen $\mathbf{A}^{-1}$ mátrixot *inverz mátrixnak* nevezzük.
Például, ha $\mathbf{A}$ az általános $2 \times 2$-es mátrix

$$
\mathbf{A} = \begin{bmatrix}
a & b \\
c & d
\end{bmatrix},
$$

akkor az inverz mátrix:

$$
 \frac{1}{ad-bc}  \begin{bmatrix}
d & -b \\
-c & a
\end{bmatrix}.
$$

A fenti képlettel adott inverzzal való szorzás ellenőrzéséhez
megvizsgálhatjuk, hogy ez a gyakorlatban is működik-e.

```{.python .input}
#@tab mxnet
M = np.array([[1, 2], [1, 4]])
M_inv = np.array([[2, -1], [-0.5, 0.5]])
M_inv.dot(M)
```

```{.python .input}
#@tab pytorch
M = torch.tensor([[1, 2], [1, 4]], dtype=torch.float32)
M_inv = torch.tensor([[2, -1], [-0.5, 0.5]])
M_inv @ M
```

```{.python .input}
#@tab tensorflow
M = tf.constant([[1, 2], [1, 4]], dtype=tf.float32)
M_inv = tf.constant([[2, -1], [-0.5, 0.5]])
tf.matmul(M_inv, M)
```

### Numerikus problémák
Bár a mátrix inverze elméletileg hasznos,
meg kell jegyeznünk, hogy a legtöbb esetben
a mátrix inverzét *nem* érdemes közvetlenül felhasználni egy probléma gyakorlati megoldásához.
Általánosan sokkal numerikusan stabilabb algoritmusok léteznek
az olyan lineáris egyenletek megoldására, mint

$$
\mathbf{A}\mathbf{x} = \mathbf{b},
$$

mint az inverz kiszámítása és megszorzása az alábbi módon:

$$
\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}.
$$

Ahogyan egy kis számmal való osztás numerikus instabilitáshoz vezethet,
úgy az alacsony ranghoz közelítő mátrix invertálása is.

Ráadásul gyakori, hogy az $\mathbf{A}$ mátrix *ritka*,
azaz csak kevés nem nulla értéket tartalmaz.
Ha megvizsgálnánk a példákat, láthatnánk,
hogy ez nem jelenti azt, hogy az inverz is ritka.
Még ha $\mathbf{A}$ egy $1$ millió $\times$ $1$ millió-os mátrix lenne
mindössze $5$ millió nem nulla elemmel
(így csak azokat az $5$ millió elemet kell tárolni),
az inverz tipikusan szinte minden elemében nem negatív lenne,
és az összes $1\textrm{M}^2$ elemet kellene tárolni – ez $1$ billió elem!

Bár nincs időnk teljesen belemerülni a lineáris algebrával való munkában
gyakran felmerülő numerikus problémák bogozásába,
szeretnénk némi intuíciót adni arról, mikor kell óvatosan eljárni,
és általánosságban az invertálás kerülése a gyakorlatban jó ökölszabálynak bizonyul.

## Determináns
A lineáris algebra geometriai szemlélete intuitív módot kínál
egy alapvető mennyiség, a *determináns* értelmezésére.
Tekintsük a korábbi rácsos képet, most egy kiemelt területtel (:numref:`fig_grid-filled`).

![Az $\mathbf{A}$ mátrix ismét torzítja a rácsot. Ezúttal különös figyelmet szeretnék fordítani arra, mi történik a kiemelt négyzettel.](../img/grid-transform-filled.svg)
:label:`fig_grid-filled`

Figyeljük meg a kiemelt négyzetet. Ez egy $(0, 1)$ és $(1, 0)$ oldalakkal rendelkező négyzet, amelynek területe egységnyi.
Miután $\mathbf{A}$ átalakítja ezt a négyzetet,
azt látjuk, hogy egy paralelogrammá válik.
Nincs ok azt feltételezni, hogy ennek a paralelogrammának ugyanakkora területe van,
mint amivel kezdtük, és valóban az itt bemutatott konkrét esetben, ahol

$$
\mathbf{A} = \begin{bmatrix}
1 & 2 \\
-1 & 3
\end{bmatrix},
$$

koordinátageometriával elvégezhető a számítás,
hogy a paralelogramma területe $5$.

Általánosan, ha adott a

$$
\mathbf{A} = \begin{bmatrix}
a & b \\
c & d
\end{bmatrix}
$$

mátrix,

némi számítással belátható, hogy a kapott paralelogramma területe $ad-bc$.
Ezt a területet *determinánsnak* nevezzük.

Ellenőrizzük ezt gyorsan néhány példakóddal.

```{.python .input}
#@tab mxnet
import numpy as np
np.linalg.det(np.array([[1, -1], [2, 3]]))
```

```{.python .input}
#@tab pytorch
torch.det(torch.tensor([[1, -1], [2, 3]], dtype=torch.float32))
```

```{.python .input}
#@tab tensorflow
tf.linalg.det(tf.constant([[1, -1], [2, 3]], dtype=tf.float32))
```

Az élesebbek észreveszik,
hogy ez a kifejezés lehet nulla vagy akár negatív is.
A negatív eset a matematikában általánosan elfogadott konvenció kérdése:
ha a mátrix megfordítja az alakzatot,
azt mondjuk, hogy a terület negatív előjelű.
Nézzük meg most, mit tudunk meg, amikor a determináns nulla.

Tekintsük a következőt:

$$
\mathbf{B} = \begin{bmatrix}
2 & 4 \\ -1 & -2
\end{bmatrix}.
$$

Ha kiszámítjuk ennek a mátrixnak a determinánsát,
$2\cdot(-2 ) - 4\cdot(-1) = 0$-t kapunk.
A fenti megértésünk alapján ez érthető.
$\mathbf{B}$ az eredeti kép négyzetét egy vonaldarabbá zsugorítja, amelynek területe nulla.
És valóban, az alacsonyabb dimenziójú térbe való tömörítés
az egyetlen mód arra, hogy a transzformáció után nulla terület adódjék.
Ezért látjuk, hogy a következő állítás igaz:
egy $A$ mátrix pontosan akkor invertálható, ha a determinánsa nem nulla.

Végső megjegyzésként képzeljük el, hogy a síkon rajzolt tetszőleges alakzatunk van.
Informatikai szemlélettel gondolkodva
az alakzatot felbonthatjuk kis négyzetekre,
így az alakzat területe lényegében
a felbontásban szereplő négyzetek számával egyenlő.
Ha most ezt az alakzatot egy mátrixszal átalakítjuk,
minden egyes négyzetet paralelogrammává viszünk,
amelyek mindegyikének területe a determináns.
Látjuk tehát, hogy bármely alakzat esetén a determináns azt az (előjeles) számot adja meg,
amellyel a mátrix skálázza bármely alakzat területét.

Nagyobb mátrixoknál a determináns kiszámítása fáradságos lehet,
de az intuíció ugyanaz marad.
A determináns marad az a szorzófaktor,
amellyel az $n\times n$-es mátrixok az $n$-dimenziós köteteket skálázzák.

## Tenzorok és általános lineáris algebrai műveletek

A :numref:`sec_linear-algebra` fejezetben bevezettük a tenzorok fogalmát.
Ebben a szakaszban mélyebben megvizsgáljuk a tenzorkontrakciót
(a mátrixszorzás tenzoros megfelelőjét),
és megmutatjuk, hogyan nyújt ez egységes képet
számos mátrix- és vektorműveletről.

A mátrixokkal és vektorokkal tudtuk, hogyan szorozzuk meg őket az adatok átalakításához.
Ehhez hasonló definícióra van szükségünk a tenzorokhoz is, ha hasznosak akarunk lenni velük.
Gondoljunk a mátrixszorzásra:

$$
\mathbf{C} = \mathbf{A}\mathbf{B},
$$

vagy ekvivalensen:

$$ c_{i, j} = \sum_{k} a_{i, k}b_{k, j}.$$

Ezt a mintát tenzorokra is megismételhetjük.
Tenzorok esetén nem lehet általánosan egyetlen összeadandó halmazt választani,
ezért pontosan meg kell adnunk, mely indexek felett összegzünk.
Például tekinthetjük a következőt:

$$
y_{il} = \sum_{jk} x_{ijkl}a_{jk}.
$$

Az ilyen transzformációt *tenzorkontrak­ciónak* nevezzük.
Lényegesen rugalmasabb transzformációcsaládot képvisel,
mint a mátrixszorzás önmagában.

Egy sokat használt jelölési egyszerűsítésként megfigyelhető,
hogy az összeg pontosan azok felett az indexek felett fut,
amelyek egynél többször fordulnak elő a kifejezésben.
Ezért az emberek gyakran az *Einstein-jelöléssel* dolgoznak,
amelyben az összeadás implicit módon az összes ismétlődő index felett értendő.
Ez a tömör kifejezést adja:

$$
y_{il} = x_{ijkl}a_{jk}.
$$

### Szokásos példák a lineáris algebrából

Nézzük meg, hogy a korábban látott lineáris algebrai definíciók közül hányat lehet
ebben a tömörített tenzoros jelölésben kifejezni:

* $\mathbf{v} \cdot \mathbf{w} = \sum_i v_iw_i$
* $\|\mathbf{v}\|_2^{2} = \sum_i v_iv_i$
* $(\mathbf{A}\mathbf{v})_i = \sum_j a_{ij}v_j$
* $(\mathbf{A}\mathbf{B})_{ik} = \sum_j a_{ij}b_{jk}$
* $\textrm{tr}(\mathbf{A}) = \sum_i a_{ii}$

Ily módon számos speciális jelölést helyettesíthetünk rövid tenzoros kifejezésekkel.

### Kifejezés kódban
A tenzorokon kódban is rugalmasan végezhetők műveletek.
Ahogy a :numref:`sec_linear-algebra` fejezetben láttuk,
az alábbiakban bemutatott módon hozhatunk létre tenzorokat.

```{.python .input}
#@tab mxnet
# Tenzorok definiálása
B = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
A = np.array([[1, 2], [3, 4]])
v = np.array([1, 2])

# Az alakok kiíratása
A.shape, B.shape, v.shape
```

```{.python .input}
#@tab pytorch
# Tenzorok definiálása
B = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
A = torch.tensor([[1, 2], [3, 4]])
v = torch.tensor([1, 2])

# Az alakok kiíratása
A.shape, B.shape, v.shape
```

```{.python .input}
#@tab tensorflow
# Tenzorok definiálása
B = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
A = tf.constant([[1, 2], [3, 4]])
v = tf.constant([1, 2])

# Az alakok kiíratása
A.shape, B.shape, v.shape
```

Az Einstein-összegzés közvetlenül implementálva van.
Az Einstein-összegzésben szereplő indexek karakterláncként adhatók át,
amelyet a műveletet végrehajtó tenzorok követnek.
Például a mátrixszorzás megvalósításához
tekintsük a fent látott Einstein-összegzést
($\mathbf{A}\mathbf{v} = a_{ij}v_j$),
és nyerjük ki az indexeket az implementációhoz:

```{.python .input}
#@tab mxnet
# A mátrixszorzás újraimplementálása
np.einsum("ij, j -> i", A, v), A.dot(v)
```

```{.python .input}
#@tab pytorch
# A mátrixszorzás újraimplementálása
torch.einsum("ij, j -> i", A, v), A@v
```

```{.python .input}
#@tab tensorflow
# A mátrixszorzás újraimplementálása
tf.einsum("ij, j -> i", A, v), tf.matmul(A, tf.reshape(v, (2, 1)))
```

Ez egy rendkívül rugalmas jelölés.
Ha például azt szeretnénk kiszámítani, amit hagyományosan így írunk:

$$
c_{kl} = \sum_{ij} \mathbf{b}_{ijk}\mathbf{a}_{il}v_j.
$$

az Einstein-összegzéssel így valósítható meg:

```{.python .input}
#@tab mxnet
np.einsum("ijk, il, j -> kl", B, A, v)
```

```{.python .input}
#@tab pytorch
torch.einsum("ijk, il, j -> kl", B, A, v)
```

```{.python .input}
#@tab tensorflow
tf.einsum("ijk, il, j -> kl", B, A, v)
```

Ez a jelölés emberek számára olvasható és hatékony,
azonban nehézkes, ha bármilyen okból
programszerűen kell tenzorkontrakciót előállítani.
Emiatt az `einsum` alternatív jelölést is kínál:
minden tenzorhoz egész indexeket adhatunk meg.
Például ugyanaz a tenzorkontrakció a következőképpen is felírható:

```{.python .input}
#@tab mxnet
np.einsum(B, [0, 1, 2], A, [0, 3], v, [1], [2, 3])
```

```{.python .input}
#@tab pytorch
# A PyTorch nem támogatja ezt a jelölést.
```

```{.python .input}
#@tab tensorflow
# A TensorFlow nem támogatja ezt a jelölést.
```

Mindkét jelölés tömör és hatékony módot biztosít a tenzorkontrakciók kódban történő megjelenítésére.

## Összefoglalás
* A vektorok geometriailag értelmezhetők mint pontok vagy irányok a térben.
* A skaláris szorzat definiálja a szög fogalmát tetszőlegesen magas dimenziójú terekre.
* A hipersíkok az egyenesek és síkok magas dimenziójú általánosításai. Felhasználhatók döntési síkok meghatározásához, amelyeket osztályozási feladatokban az utolsó lépésként szokás alkalmazni.
* A mátrixszorzás geometriailag az alapvető koordináták egyenletes torzításaként értelmezhető. Ez egy erősen korlátozott, de matematikailag elegáns módja a vektorok átalakításának.
* A lineáris függőség módot ad annak felismerésére, ha vektorok halmaza alacsonyabb dimenziójú térben helyezkedik el, mint várható lenne (például $3$ vektor $2$ dimenziós térben). Egy mátrix rangja az oszlopok lineárisan független részhalmazának maximális mérete.
* Ha egy mátrix inverze értelmezett, a mátrixinverzió lehetővé teszi egy másik mátrix megtalálását, amely visszacsinája az első hatását. A mátrixinverzió elméletileg hasznos, de a gyakorlatban óvatosságot igényel a numerikus instabilitás miatt.
* A determinánsok lehetővé teszik annak mérését, mennyire tágítja vagy összenyomja a mátrix a teret. A nem nulla determináns invertálható (nem szinguláris) mátrixot jelent, a nulla determináns pedig azt, hogy a mátrix nem invertálható (szinguláris).
* A tenzorkontrakciók és az Einstein-összegzés elegáns és tiszta jelölést biztosítanak a gépi tanulásban előforduló számos számítás kifejezéséhez.

## Feladatok
1. Mekkora a szög a következő vektorok között?
$$
\vec v_1 = \begin{bmatrix}
1 \\ 0 \\ -1 \\ 2
\end{bmatrix}, \qquad \vec v_2 = \begin{bmatrix}
3 \\ 1 \\ 0 \\ 1
\end{bmatrix}?
$$
2. Igaz vagy hamis: $\begin{bmatrix}1 & 2\\0&1\end{bmatrix}$ és $\begin{bmatrix}1 & -2\\0&1\end{bmatrix}$ egymás inverzei?
3. Tegyük fel, hogy a síkon rajzolunk egy $100\textrm{m}^2$ területű alakzatot. Mekkora a terület az alakzat alábbi mátrixszal való transzformálása után?
$$
\begin{bmatrix}
2 & 3\\
1 & 2
\end{bmatrix}.
$$
4. Az alábbi vektorhalmazok közül melyek lineárisan függetlenek?
 * $\left\{\begin{pmatrix}1\\0\\-1\end{pmatrix}, \begin{pmatrix}2\\1\\-1\end{pmatrix}, \begin{pmatrix}3\\1\\1\end{pmatrix}\right\}$
 * $\left\{\begin{pmatrix}3\\1\\1\end{pmatrix}, \begin{pmatrix}1\\1\\1\end{pmatrix}, \begin{pmatrix}0\\0\\0\end{pmatrix}\right\}$
 * $\left\{\begin{pmatrix}1\\1\\0\end{pmatrix}, \begin{pmatrix}0\\1\\-1\end{pmatrix}, \begin{pmatrix}1\\0\\1\end{pmatrix}\right\}$
5. Tegyük fel, hogy adott egy $A = \begin{bmatrix}c\\d\end{bmatrix}\cdot\begin{bmatrix}a & b\end{bmatrix}$ alakú mátrix valamely $a, b, c, d$ értékekre. Igaz vagy hamis: az ilyen mátrix determinánsa mindig $0$?
6. Az $e_1 = \begin{bmatrix}1\\0\end{bmatrix}$ és $e_2 = \begin{bmatrix}0\\1\end{bmatrix}$ vektorok ortogonálisak. Mi a feltétele egy $A$ mátrixra, hogy $Ae_1$ és $Ae_2$ ortogonálisak legyenek?
7. Hogyan írható $\textrm{tr}(\mathbf{A}^4)$ Einstein-jelöléssel tetszőleges $A$ mátrixra?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/410)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1084)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1085)
:end_tab:
