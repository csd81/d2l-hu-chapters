# Figyelempooling hasonlóság alapján

:label:`sec_attention-pooling`

Most, hogy bevezettük a figyelemmechanizmus elsődleges összetevőit, alkalmazzuk őket egy meglehetősen klasszikus kontextusban, nevezetesen a regresszióban és az osztályozásban kernel-sűrűségbecslés segítségével :cite:`Nadaraya.1964,Watson.1964`. Ez a kitérő csupán további háttérinformációt nyújt: teljesen opcionális, és szükség esetén kihagyható.
A Nadaraya–Watson-becslők lényegükben valamely $\alpha(\mathbf{q}, \mathbf{k})$ hasonlósági kernelre támaszkodnak, amely a $\mathbf{q}$ lekérdezéseket a $\mathbf{k}$ kulcsokhoz kapcsolja. Néhány közös kernel:

$$\begin{aligned}
\alpha(\mathbf{q}, \mathbf{k}) & = \exp\left(-\frac{1}{2} \|\mathbf{q} - \mathbf{k}\|^2 \right) && \textrm{Gauss;} \\
\alpha(\mathbf{q}, \mathbf{k}) & = 1 \textrm{ if } \|\mathbf{q} - \mathbf{k}\| \leq 1 && \textrm{Boxcar;} \\
\alpha(\mathbf{q}, \mathbf{k}) & = \mathop{\mathrm{max}}\left(0, 1 - \|\mathbf{q} - \mathbf{k}\|\right) && \textrm{Epanechikov.}
\end{aligned}
$$

Számos más lehetőség is létezik. Bővebb áttekintésért és a kernelek választásának kernel-sűrűségbecsléssel való kapcsolatáért lásd a [Wikipédia cikket](https://en.wikipedia.org/wiki/Kernel_(statistics)), amelyet néha *Parzen-ablakoknak* is neveznek :cite:`parzen1957consistent`. Minden kernel heurisztikus és hangolható. Például a szélességüket állíthatjuk, nem csak globálisan, hanem koordinátánként is. Ettől függetlenül mindegyik az alábbi egyenlethez vezet, mind regresszió, mind osztályozás esetén:

$$f(\mathbf{q}) = \sum_i \mathbf{v}_i \frac{\alpha(\mathbf{q}, \mathbf{k}_i)}{\sum_j \alpha(\mathbf{q}, \mathbf{k}_j)}.$$

(Skaláris) regresszió esetén, ahol az $(\mathbf{x}_i, y_i)$ megfigyelések jellemzőket és címkéket jelölnek, $\mathbf{v}_i = y_i$ skalárok, $\mathbf{k}_i = \mathbf{x}_i$ vektorok, és a $\mathbf{q}$ lekérdezés azt az új helyet jelöli, ahol $f$-et ki kell értékelni. (Többosztályos) osztályozás esetén az $y_i$ one-hot kódolását használjuk a $\mathbf{v}_i$ előállításához. Ennek a becslőnek az egyik kényelmes tulajdonsága, hogy nem igényel tanítást. Sőt, ha a kernelt a növekvő adatmennyiséggel megfelelően szűkítjük, a megközelítés konzisztens :cite:`mack1982weak`, azaz valamely statisztikailag optimális megoldáshoz fog konvergálni. Kezdjük néhány kernel vizsgálatával.

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
npx.set_np()
d2l.use_svg_display()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

d2l.use_svg_display()
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import numpy as np

d2l.use_svg_display()
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
import jax
from jax import numpy as jnp
from flax import linen as nn
```

## [**Kernelek és adatok**]

Az ebben a szakaszban definiált összes $\alpha(\mathbf{k}, \mathbf{q})$ kernel *eltolás- és rotációinvariáns*; azaz, ha $\mathbf{k}$-t és $\mathbf{q}$-t azonos módon eltoljuk és elforgatjuk, az $\alpha$ értéke változatlan marad. Az egyszerűség kedvéért skaláris $k, q \in \mathbb{R}$ argumentumokat választunk, és a $k = 0$ kulcsot vesszük origónak. Ez az alábbit adja:

```{.python .input}
%%tab all
# Definiáljunk néhány kernelt
def gaussian(x):
    return d2l.exp(-x**2 / 2)

def boxcar(x):
    return d2l.abs(x) < 1.0

def constant(x):
    return 1.0 + 0 * x
 
if tab.selected('pytorch'):
    def epanechikov(x):
        return torch.max(1 - d2l.abs(x), torch.zeros_like(x))
if tab.selected('mxnet'):
    def epanechikov(x):
        return np.maximum(1 - d2l.abs(x), 0)
if tab.selected('tensorflow'):
    def epanechikov(x):
        return tf.maximum(1 - d2l.abs(x), 0)
if tab.selected('jax'):
    def epanechikov(x):
        return jnp.maximum(1 - d2l.abs(x), 0)
```

```{.python .input}
%%tab all
fig, axes = d2l.plt.subplots(1, 4, sharey=True, figsize=(12, 3))

kernels = (gaussian, boxcar, constant, epanechikov)
names = ('Gauss', 'Boxcar', 'Konstans', 'Epanechikov')
x = d2l.arange(-2.5, 2.5, 0.1)
for kernel, name, ax in zip(kernels, names, axes):
    if tab.selected('pytorch', 'mxnet', 'tensorflow'):
        ax.plot(d2l.numpy(x), d2l.numpy(kernel(x)))
    if tab.selected('jax'):
        ax.plot(x, kernel(x))
    ax.set_xlabel(name)

d2l.plt.show()
```

A különböző kernelek a hatótávolság és simítás különböző fogalmainak felelnek meg. Például a boxcar kernel csak az $1$-es (vagy más módon meghatározott hiperparaméter) távolságon belüli megfigyelésekre figyel, és ezt megkülönböztetés nélkül teszi.

A Nadaraya–Watson-becslés működésének szemléltetéséhez definiáljunk néhány tanítási adatot. A következőkben a

$$y_i = 2\sin(x_i) + x_i + \epsilon,$$

függőséget használjuk, ahol $\epsilon$ nulla átlagú és egységnyi szórású normál eloszlásból van húzva. 40 tanítási példát rajzolunk.

```{.python .input}
%%tab all
def f(x):
    return 2 * d2l.sin(x) + x

n = 40
if tab.selected('pytorch'):
    x_train, _ = torch.sort(d2l.rand(n) * 5)
    y_train = f(x_train) + d2l.randn(n)
if tab.selected('mxnet'):
    x_train = np.sort(d2l.rand(n) * 5, axis=None)
    y_train = f(x_train) + d2l.randn(n)
if tab.selected('tensorflow'):
    x_train = tf.sort(d2l.rand((n,1)) * 5, 0)
    y_train = f(x_train) + d2l.normal((n, 1))
if tab.selected('jax'):
    x_train = jnp.sort(jax.random.uniform(d2l.get_key(), (n,)) * 5)
    y_train = f(x_train) + jax.random.normal(d2l.get_key(), (n,))
x_val = d2l.arange(0, 5, 0.1)
y_val = f(x_val)
```

## [**Figyelempooling Nadaraya–Watson Regresszión Keresztül**]

Most, hogy megvannak az adataink és a kerneleink, csak egy függvényre van szükségünk, amely kiszámítja a kernel regressziós becsléseket. Vegyük észre, hogy a relatív kernel-súlyokat is meg akarjuk kapni néhány kisebb diagnosztika elvégzéséhez. Ezért először kiszámítjuk a kernelt az összes tanítási jellemző (kovariáns) `x_train` és az összes validációs jellemző `x_val` között. Ez egy mátrixot eredményez, amelyet ezután normalizálunk. A `y_train` tanítási címkékkel szorozva megkapjuk a becsléseket.

Idézzük fel a figyelempooling-ot a :eqref:`eq_attention_pooling`-ban. Legyen minden validációs jellemző lekérdezés, és minden tanítási jellemző–címke pár kulcs–érték pár. Ennek eredményeként a normalizált relatív kernel-súlyok (alább `attention_w`) a *figyelemsúlyok*.

```{.python .input}
%%tab all
def nadaraya_watson(x_train, y_train, x_val, kernel):
    dists = d2l.reshape(x_train, (-1, 1)) - d2l.reshape(x_val, (1, -1))
    # Minden oszlop/sor egy-egy lekérdezésnek/kulcsnak felel meg
    k = d2l.astype(kernel(dists), d2l.float32)
    # Kulcsonkénti normalizálás minden lekérdezéshez
    attention_w = k / d2l.reduce_sum(k, 0)
    if tab.selected('pytorch'):
        y_hat = y_train@attention_w
    if tab.selected('mxnet'):
        y_hat = np.dot(y_train, attention_w)
    if tab.selected('tensorflow'):
        y_hat = d2l.transpose(d2l.transpose(y_train)@attention_w)
    if tab.selected('jax'):
        y_hat = y_train@attention_w
    return y_hat, attention_w
```

Nézzük meg, milyen becsléseket állítanak elő a különböző kernelek.

```{.python .input}
%%tab all
def plot(x_train, y_train, x_val, y_val, kernels, names, attention=False):
    fig, axes = d2l.plt.subplots(1, 4, sharey=True, figsize=(12, 3))
    for kernel, name, ax in zip(kernels, names, axes):
        y_hat, attention_w = nadaraya_watson(x_train, y_train, x_val, kernel)
        if attention:
            if tab.selected('pytorch', 'mxnet', 'tensorflow'):
                pcm = ax.imshow(d2l.numpy(attention_w), cmap='Reds')
            if tab.selected('jax'):
                pcm = ax.imshow(attention_w, cmap='Reds')
        else:
            ax.plot(x_val, y_hat)
            ax.plot(x_val, y_val, 'm--')
            ax.plot(x_train, y_train, 'o', alpha=0.5);
        ax.set_xlabel(name)
        if not attention:
            ax.legend(['y_hat', 'y'])
    if attention:
        fig.colorbar(pcm, ax=axes, shrink=0.7)
```

```{.python .input}
%%tab all
plot(x_train, y_train, x_val, y_val, kernels, names)
```

Az első szembetűnő dolog az, hogy mindhárom nemtriviális kernel (Gauss, Boxcar és Epanechikov) meglehetősen használható becsléseket produkál, amelyek nem esnek messze a valódi függvénytől. Csak a konstans kernel, amely a triviális $f(x) = \frac{1}{n} \sum_i y_i$ becsléshez vezet, ad egy inkább irreális eredményt. Vizsgáljuk meg közelebbről a figyelmi súlyozást:

```{.python .input}
%%tab all
plot(x_train, y_train, x_val, y_val, kernels, names, attention=True)
```

A vizualizáció egyértelműen megmutatja, miért nagyon hasonlóak a Gauss, a Boxcar és az Epanechikov becslései: végül is nagyon hasonló figyelemsúlyokból nyerhetők, annak ellenére, hogy a kernel funkcionális formája eltérő. Ez felveti a kérdést, hogy ez mindig így van-e.

## [**Figyelempooling adaptálása**]

Lecserélhetnénk a Gauss-kernelt egy eltérő szélességűre. Azaz használhatnánk
$\alpha(\mathbf{q}, \mathbf{k}) = \exp\left(-\frac{1}{2 \sigma^2} \|\mathbf{q} - \mathbf{k}\|^2 \right)$-t, ahol $\sigma^2$ a kernel szélességét határozza meg. Lássuk, hogy ez befolyásolja-e az eredményeket.

```{.python .input}
%%tab all
sigmas = (0.1, 0.2, 0.5, 1)
names = ['Szigma ' + str(sigma) for sigma in sigmas]

def gaussian_with_width(sigma): 
    return (lambda x: d2l.exp(-x**2 / (2*sigma**2)))

kernels = [gaussian_with_width(sigma) for sigma in sigmas]
plot(x_train, y_train, x_val, y_val, kernels, names)
```

Egyértelműen minél szűkebb a kernel, annál kevésbé sima a becslés. Ugyanakkor jobban alkalmazkodik a helyi változásokhoz. Nézzük meg a megfelelő figyelemsúlyokat.

```{.python .input}
%%tab all
plot(x_train, y_train, x_val, y_val, kernels, names, attention=True)
```

Ahogyan várható, minél szűkebb a kernel, annál szűkebb a nagy figyelemsúlyok tartománya. Az is egyértelmű, hogy azonos szélesség választása nem biztos, hogy ideális. Valójában :citet:`Silverman86` egy heurisztikát javasolt, amely a helyi sűrűségtől függ. Számos ilyen „trükköt" javasoltak. Például :citet:`norelli2022asif` egy hasonló legközelebbi szomszéd interpolációs technikát használt képek és szöveg keresztmodális reprezentációinak tervezéséhez.

Az éles szemű olvasó talán kíváncsian tűnődik, miért nyújtunk ilyen részletes ismertetőt egy több mint fél évszázados módszerről. Először is, ez az egyik legkorábbi előfutára a modern figyelemmechanizmusoknak. Másodszor, kiválóan alkalmas vizualizációra. Harmadszor, és legalább ugyanolyan fontos, bemutatja a kézzel tervezett figyelemmechanizmusok korlátait. Sokkal jobb stratégia a mechanizmust *megtanulni*, a lekérdezések és kulcsok reprezentációit elsajátítva. Ezt fogjuk megtenni a következő szakaszokban.


## Összefoglalás

A Nadaraya–Watson kernel regresszió a jelenlegi figyelemmechanizmusok korai előfutára.
Közvetlenül használható kevés vagy egyáltalán nem igényelt tanítással vagy hangolással, mind osztályozáshoz, mind regresszióhoz.
A figyelemsúly a lekérdezés és a kulcs közötti hasonlóság (vagy távolság) alapján kerül hozzárendelésre, és attól függően, hogy hány hasonló megfigyelés áll rendelkezésre.

## Feladatok

1. A Parzen-ablak sűrűségbecslések $\hat{p}(\mathbf{x}) = \frac{1}{n} \sum_i k(\mathbf{x}, \mathbf{x}_i)$-vel adhatók meg. Bizonyítsd be, hogy bináris osztályozás esetén a $\hat{p}(\mathbf{x}, y=1) - \hat{p}(\mathbf{x}, y=-1)$ függvény, amelyet Parzen-ablakok adnak, ekvivalens a Nadaraya–Watson-osztályozással.
1. Implementálj sztochasztikus gradienscsökkenést a kernel-szélességek jó értékének megtanulásához Nadaraya–Watson regresszióban.
    1. Mi történik, ha egyszerűen a fenti becsléseket használod a $(f(\mathbf{x_i}) - y_i)^2$ közvetlen minimalizálásához? Tipp: $y_i$ része a $f$ kiszámításához használt tagoknak.
    1. Távolítsd el az $(\mathbf{x}_i, y_i)$-t az $f(\mathbf{x}_i)$ becsléséből, és optimalizálj a kernel-szélességeken. Még mindig megfigyelsz túltanulást?
1. Tételezd fel, hogy az összes $\mathbf{x}$ az egységgömbön van, azaz mindegyikre teljesül, hogy $\|\mathbf{x}\| = 1$. Egyszerűsítheted-e az $\|\mathbf{x} - \mathbf{x}_i\|^2$ tagot az exponensben? Tipp: később látni fogjuk, hogy ez szorosan összefügg a skalárisszorzat-alapú figyelemmel.
1. Emlékezz arra, hogy :citet:`mack1982weak` bizonyította, hogy a Nadaraya–Watson-becslés konzisztens. Milyen gyorsan kell csökkentened a figyelemmechanizmus skáláját, ahogy több adatot kapsz? Adj némi intuíciót a válaszodhoz. Függ az adatok dimenziójától? Hogyan?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1598)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1599)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/3866)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18026)
:end_tab:
