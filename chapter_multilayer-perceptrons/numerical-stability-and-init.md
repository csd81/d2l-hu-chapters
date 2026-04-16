```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Numerikus stabilitás és inicializálás
:label:`sec_numerical_stability`


Eddig minden általunk implementált modell megkövetelte,
hogy paramétereit valamely előre meghatározott eloszlás szerint inicializáljuk.
Eddig az inicializálási sémát adottnak vettük,
és nem foglalkoztunk azzal, hogyan születnek ezek a döntések.
Talán az volt a benyomásod, hogy ezek a döntések
nem különösen fontosak.
Épp ellenkezőleg: az inicializálási séma megválasztása
fontos szerepet játszik a neurális hálózatok tanulásában,
és kritikus lehet a numerikus stabilitás megőrzése szempontjából.
Ráadásul ezek a döntések érdekesen összefonódhatnak
a nemlineáris aktivációs függvény megválasztásával.
Az, hogy melyik függvényt választjuk, és hogyan inicializáljuk a paramétereket,
meghatározhatja, milyen gyorsan konvergál az optimalizálási algoritmusunk.
Helytelen döntések esetén tanítás közben felrobbanó vagy elhaló gradiensekkel találkozhatunk.
Ebben a részben mélyebben megvizsgáljuk ezeket a témákat,
és néhány hasznos heurisztikát tárgyalunk,
amelyek hasznosnak bizonyulnak majd
a mély tanulási karriered során.

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
from jax import grad, vmap
```

## Elhaló és felrobbanó gradiensek

Tekintsünk egy $L$ rétegű mély hálózatot,
$\mathbf{x}$ bemenettel és $\mathbf{o}$ kimenettel.
Ha minden $l$ réteget egy $f_l$ transzformáció definiál
$\mathbf{W}^{(l)}$ súlyokkal parametrizálva,
amelynek rejtett réteg kimenete $\mathbf{h}^{(l)}$ (legyen $\mathbf{h}^{(0)} = \mathbf{x}$),
a hálózatunk a következőképpen fejezhető ki:

$$\mathbf{h}^{(l)} = f_l (\mathbf{h}^{(l-1)}) \textrm{ és így } \mathbf{o} = f_L \circ \cdots \circ f_1(\mathbf{x}).$$

Ha az összes rejtett réteg kimenet és a bemenet vektorok,
az $\mathbf{o}$ gradiensét bármely $\mathbf{W}^{(l)}$ paraméterkészletre
a következőképpen írhatjuk:

$$\partial_{\mathbf{W}^{(l)}} \mathbf{o} = \underbrace{\partial_{\mathbf{h}^{(L-1)}} \mathbf{h}^{(L)}}_{ \mathbf{M}^{(L)} \stackrel{\textrm{def}}{=}} \cdots \underbrace{\partial_{\mathbf{h}^{(l)}} \mathbf{h}^{(l+1)}}_{ \mathbf{M}^{(l+1)} \stackrel{\textrm{def}}{=}} \underbrace{\partial_{\mathbf{W}^{(l)}} \mathbf{h}^{(l)}}_{ \mathbf{v}^{(l)} \stackrel{\textrm{def}}{=}}.$$

Más szóval ez a gradiens
$L-l$ mátrix szorzata
$\mathbf{M}^{(L)} \cdots \mathbf{M}^{(l+1)}$
és a $\mathbf{v}^{(l)}$ gradiensvektorral.
Így ugyanolyan numerikus alulcsordulási problémáknak vagyunk kitéve,
amelyek akkor jelentkeznek, ha túl sok valószínűséget szorzunk össze.
Valószínűségek esetén általánosan alkalmazott trükk,
hogy log-térbe váltunk, azaz a numerikus reprezentáció
mantisszájáról a kitevőre helyezzük az értéket.
Sajnos a mi problémánk komolyabb:
kezdetben a $\mathbf{M}^{(l)}$ mátrixoknak nagyon különböző sajátértékeik lehetnek.
Lehetnek kicsik vagy nagyok,
és szorzatuk *nagyon nagy* vagy *nagyon kicsi* lehet.

Az instabil gradiensek által okozott kockázatok
messze túlmutatnak a numerikus reprezentáción.
A kiszámíthatatlan nagyságú gradiensek
veszélyeztetik optimalizálási algoritmusaink stabilitását is.
Előfordulhat, hogy paraméterfrissítéseink vagy
(i) túlságosan nagyok, tönkretéve a modellünket
(*felrobbanó gradiens* probléma);
vagy (ii) túlságosan kicsik
(*elhaló gradiens* probléma),
ami lehetetlenné teszi a tanulást, mivel a paraméterek
alig mozdulnak el minden egyes frissítésnél.


### (**Elhaló gradiensek**)

Az elhaló gradiens problémát okozó egyik gyakori bűnös
az egyes rétegek lineáris műveleteinek végén alkalmazott
$\sigma$ aktivációs függvény megválasztása.
Történelmileg a sigmoid függvény
$1/(1 + \exp(-x))$ (amelyet a :numref:`sec_mlp` részben mutattunk be)
népszerű volt, mert hasonlít egy küszöbfüggvényre.
Mivel a korai mesterséges neurális hálózatokat
biológiai neurális hálózatok ihlették,
az ötlet, hogy a neuronok vagy *teljesen tüzelnek*, vagy *egyáltalán nem*
(mint a biológiai neuronok), vonzónak tűnt.
Vizsgáljuk meg közelebbről a sigmoidt,
hogy miért okozhat elhaló gradienseket.

```{.python .input}
%%tab mxnet
x = np.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    y = npx.sigmoid(x)
y.backward()

d2l.plot(x, [y, x.grad], legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

```{.python .input}
%%tab pytorch
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))

d2l.plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

```{.python .input}
%%tab tensorflow
x = tf.Variable(tf.range(-8.0, 8.0, 0.1))
with tf.GradientTape() as t:
    y = tf.nn.sigmoid(x)
d2l.plot(x.numpy(), [y.numpy(), t.gradient(y, x).numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

```{.python .input}
%%tab jax
x = jnp.arange(-8.0, 8.0, 0.1)
y = jax.nn.sigmoid(x)
grad_sigmoid = vmap(grad(jax.nn.sigmoid))
d2l.plot(x, [y, grad_sigmoid(x)],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

Mint látható, (**a sigmoid gradiense eltűnik
mind nagy, mind kis bemenetei esetén**).
Ráadásul, ha sok rétegen keresztül haladunk vissza,
hacsak nem vagyunk az „arany közepén", ahol
sok sigmoid bemenete közel van a nullához,
az összesített szorzat gradiense eltűnhet.
Ha a hálózatunk sok réteggel rendelkezik,
óvatlanság esetén a gradiens valószínűleg
valamely rétegnél leállítódik.
Valóban, ez a probléma korábban gyötörte a mély hálózatok tanítását.
Ennek következtében a ReLU-k, amelyek stabilabbak
(de neuronbiológiailag kevésbé megalapozottak),
a szakemberek alapértelmezett választásává váltak.


### [**Felrobbanó gradiensek**]

Az ellentétes probléma, amikor a gradiensek felrobbannak,
hasonlóan zavaró lehet.
Ennek szemléltetésére
100 Gauss-féle véletlenszerű mátrixot rajzolunk
és megszorozzuk azokat egy kezdeti mátrixszal.
Az általunk választott skálán
(a $\sigma^2=1$ variancia megválasztása esetén)
a mátrixszorzat felrobban.
Ha ez a mély hálózat inicializálása miatt történik,
nincs esélyünk arra, hogy egy gradienscsökkenés optimalizáló konvergáljon.

```{.python .input}
%%tab mxnet
M = np.random.normal(size=(4, 4))
print('a single matrix', M)
for i in range(100):
    M = np.dot(M, np.random.normal(size=(4, 4)))
print('after multiplying 100 matrices', M)
```

```{.python .input}
%%tab pytorch
M = torch.normal(0, 1, size=(4, 4))
print('a single matrix \n',M)
for i in range(100):
    M = M @ torch.normal(0, 1, size=(4, 4))
print('after multiplying 100 matrices\n', M)
```

```{.python .input}
%%tab tensorflow
M = tf.random.normal((4, 4))
print('a single matrix \n', M)
for i in range(100):
    M = tf.matmul(M, tf.random.normal((4, 4)))
print('after multiplying 100 matrices\n', M.numpy())
```

```{.python .input}
%%tab jax
get_key = lambda: jax.random.PRNGKey(d2l.get_seed())  # PRNG-kulcsok generálása
M = jax.random.normal(get_key(), (4, 4))
print('a single matrix \n', M)
for i in range(100):
    M = jnp.matmul(M, jax.random.normal(get_key(), (4, 4)))
print('after multiplying 100 matrices\n', M)
```

### A szimmetria megtörése

A neurális hálózatok tervezésének egy másik problémája
a parametrizációjukban rejlő szimmetria.
Tegyük fel, hogy van egy egyszerű MLP-nk
egy rejtett réteggel és két egységgel.
Ebben az esetben felcserélhetnénk az első réteg
$\mathbf{W}^{(1)}$ súlyait, és hasonlóképpen felcserélhetnénk
a kimeneti réteg súlyait,
és ugyanazt a függvényt kapnánk.
Nincs semmi különleges, ami megkülönbözteti
az első és a második rejtett egységet.
Más szóval permutációs szimmetria áll fenn
az egyes rétegek rejtett egységei között.

Ez több, mint puszta elméleti kellemetlenség.
Tekintsük a fent említett egyetlen rejtett réteges MLP-t
két rejtett egységgel.
Szemléltetésül tegyük fel, hogy a kimeneti réteg
a két rejtett egységet egyetlen kimeneti egységgé transzformálja.
Képzeljük el, mi történne, ha a rejtett réteg összes paraméterét
$\mathbf{W}^{(1)} = c$ értékre inicializálnánk valamely $c$ konstanssal.
Ebben az esetben az előre irányú terjesztés során
mindkét rejtett egység ugyanazokat a bemeneteket és paramétereket kapja,
ugyanolyan aktivációt produkálva,
amelyet a kimeneti egységbe táplálnak.
A visszaterjesztés során a kimeneti egység $\mathbf{W}^{(1)}$ paraméterekre vonatkozó differenciálása
olyan gradienseket ad, amelyek összes eleme azonos értéket vesz fel.
Így a gradiens alapú iteráció (pl. mini-batch sztochasztikus gradienscsökkenés) után
a $\mathbf{W}^{(1)}$ összes eleme még mindig azonos értéket vesz fel.
Az ilyen iterációk sohasem *törnék meg a szimmetriát* önállóan,
és lehet, hogy soha nem tudnánk realizálni
a hálózat kifejezőerejét.
A rejtett réteg úgy viselkedne,
mintha csak egyetlen egységből állna.
Vegyük észre, hogy bár a mini-batch sztochasztikus gradienscsökkenés nem törné meg ezt a szimmetriát,
a dropout regularizáció (amelyet később mutatunk be) megtörné!


## Paraméter-inicializálás

A fent felvetett kérdések kezelésének --- vagy legalábbis enyhítésének --- egyik módja
a gondos inicializálás.
Amint később látni fogjuk,
az optimalizálás során végzett további gondosság
és a megfelelő regularizáció tovább fokozhatja a stabilitást.


### Alapértelmezett inicializálás

Az előző részekben, pl. a :numref:`sec_linear_concise` részben,
normális eloszlást használtunk
súlyaink értékeinek inicializálásához.
Ha nem adjuk meg az inicializálási módszert, a keretrendszer
alapértelmezett véletlenszerű inicializálási módszert fog használni,
amely a mérsékelt méretű problémák esetén általában jól működik a gyakorlatban.


### Xavier inicializálás
:label:`subsec_xavier`

Vizsgáljuk meg valamely teljesen összekötött réteg
egy $o_{i}$ kimenetének skálázási eloszlását
*nemlinearitások nélkül*.
$n_\textrm{in}$ bemenettel $x_j$
és a megfelelő $w_{ij}$ súlyokkal ennél a rétegnél,
a kimenet:

$$o_{i} = \sum_{j=1}^{n_\textrm{in}} w_{ij} x_j.$$

A $w_{ij}$ súlyok mind ugyanabból az eloszlásból
egymástól függetlenül vannak húzva.
Tegyük fel továbbá, hogy ennek az eloszlásnak
nulla a várható értéke és $\sigma^2$ a varianciája.
Vegyük észre, hogy ez nem jelenti azt, hogy az eloszlásnak Gauss-eloszlásnak kell lennie,
csupán azt, hogy a várható értéknek és a varianciának kell léteznie.
Egyelőre tegyük fel, hogy a réteg $x_j$ bemenetei is
nulla várható értékűek és $\gamma^2$ varianciájúak,
és egymástól, valamint $w_{ij}$-tól is függetlenek.
Ebben az esetben kiszámíthatjuk az $o_i$ várható értékét:

$$
\begin{aligned}
    E[o_i] & = \sum_{j=1}^{n_\textrm{in}} E[w_{ij} x_j] \\&= \sum_{j=1}^{n_\textrm{in}} E[w_{ij}] E[x_j] \\&= 0, \end{aligned}$$

és a varianciát:

$$
\begin{aligned}
    \textrm{Var}[o_i] & = E[o_i^2] - (E[o_i])^2 \\
        & = \sum_{j=1}^{n_\textrm{in}} E[w^2_{ij} x^2_j] - 0 \\
        & = \sum_{j=1}^{n_\textrm{in}} E[w^2_{ij}] E[x^2_j] \\
        & = n_\textrm{in} \sigma^2 \gamma^2.
\end{aligned}
$$

A variancia rögzítésének egyik módja
az $n_\textrm{in} \sigma^2 = 1$ feltétel teljesítése.
Most tekintsük a visszaterjesztést.
Ott hasonló problémával szembesülünk,
bár a gradiensek a kimenethez közelebbi rétegekből terjednek.
Ugyanolyan érveléssel, mint az előre irányú terjesztés esetén,
láthatjuk, hogy a gradiensek varianciája felrobbanhat,
hacsak nem teljesül az $n_\textrm{out} \sigma^2 = 1$ feltétel,
ahol $n_\textrm{out}$ ennek a rétegnek a kimeneteinek száma.
Ez dilemma elé állít minket:
egyszerre mindkét feltételt nem lehet teljesíteni.
Ehelyett egyszerűen megpróbáljuk teljesíteni:

$$
\begin{aligned}
\frac{1}{2} (n_\textrm{in} + n_\textrm{out}) \sigma^2 = 1 \textrm{ vagy egyenértékűen }
\sigma = \sqrt{\frac{2}{n_\textrm{in} + n_\textrm{out}}}.
\end{aligned}
$$

Ez az érvelés áll a mára már standard és gyakorlatilag hasznos
*Xavier inicializálás* mögött,
amelyet alkotói közül az elsőről neveztek el :cite:`Glorot.Bengio.2010`.
Az Xavier inicializálás általában
nulla várható értékű és $\sigma^2 = \frac{2}{n_\textrm{in} + n_\textrm{out}}$ varianciájú
Gauss-eloszlásból mintavételezi a súlyokat.
Ezt egyenletesen eloszlott súlyok mintavételezéséhez is
adaptálhatjuk.
Vegyük észre, hogy az $U(-a, a)$ egyenletes eloszlás varianciája $\frac{a^2}{3}$.
Az $\frac{a^2}{3}$-ot behelyettesítve a $\sigma^2$-ra vonatkozó feltételünkbe,
a következő inicializáláshoz jutunk:

$$U\left(-\sqrt{\frac{6}{n_\textrm{in} + n_\textrm{out}}}, \sqrt{\frac{6}{n_\textrm{in} + n_\textrm{out}}}\right).$$

Bár a fenti matematikai érvelésben szereplő
nemlinearitás hiányára vonatkozó feltételezés
könnyen megsérthető neurális hálózatokban,
az Xavier inicializálási módszer
a gyakorlatban jól működik.


### Továbbiakban

A fenti érvelés csupán felszínesen érinti
a paraméter-inicializálás modern megközelítéseit.
Egy mély tanulási keretrendszer többtucat különböző heurisztikát implementál.
Ráadásul a paraméter-inicializálás továbbra is
a mély tanulás alapkutatásának aktív területe.
Ezek között vannak heurisztikák
kapcsolt (megosztott) paraméterekre, szuperfelbontásra,
sorozatmodellekre és egyéb helyzetekre.
Például :citet:`Xiao.Bahri.Sohl-Dickstein.ea.2018` bemutatta
10 000 rétegű neurális hálózatok tanításának lehetőségét
architekturális trükkök nélkül,
gondosan megtervezett inicializálási módszer alkalmazásával.

Ha a téma érdekel, ajánljuk
egy elmélyülést a modul lehetőségeiben,
az egyes heurisztikákat javasoló és elemző cikkek elolvasásában,
majd a legújabb publikációk felfedezésében erről a témáról.
Talán rábuksz vagy akár fel is találsz
egy okos ötletet, és implementációt nyújtasz be mély tanulási keretrendszerekbe.


## Összefoglalás

Az elhaló és felrobbanó gradiensek általánosan előforduló problémák a mély hálózatokban. A paraméter-inicializálásnál nagy gondosság szükséges annak biztosítása érdekében, hogy a gradiensek és paraméterek jól kézben tarthatók legyenek.
Inicializálási heurisztikákra van szükség annak biztosítására, hogy a kezdeti gradiensek se ne legyenek túl nagyok, se túl kicsik.
A véletlenszerű inicializálás kulcsfontosságú a szimmetria megtörésének biztosításához az optimalizálás megkezdése előtt.
Az Xavier inicializálás azt javasolja, hogy minden rétegnél egy kimenet varianciáját ne befolyásolja a bemenetek száma, és egy gradiens varianciáját ne befolyásolja a kimenetek száma.
A ReLU aktivációs függvények enyhítik az elhaló gradiens problémát. Ez gyorsíthatja a konvergenciát.

## Feladatok

1. Tudsz-e más eseteket tervezni, ahol egy neurális hálózat megtörést igénylő szimmetriát mutathat, az MLP rétegeinek permutációs szimmetriáján túl?
1. Inicializálhatjuk-e az összes súlyparamétert a lineáris regresszióban vagy a softmax regresszióban ugyanolyan értékre?
1. Nézd meg a két mátrix szorzatának sajátértékeire vonatkozó analitikus korlátokat. Mit mond ez a gradiensek jó kondicionáltságának biztosításáról?
1. Ha tudjuk, hogy bizonyos tagok divergálnak, meg tudunk-e ezt utólag javítani? Nézd meg a réteges adaptív sebesség skálázásról szóló cikket inspirációként :cite:`You.Gitman.Ginsburg.2017`.


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/103)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/104)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/235)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17986)
:end_tab:
