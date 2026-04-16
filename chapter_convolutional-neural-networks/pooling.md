```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Pooling
:label:`sec_pooling`

Sok esetben végső feladatunk a képpel kapcsolatos globális kérdéseket vet fel, például *tartalmaz-e macskát?* Következésképpen a végső réteg egységeinek érzékenynek kell lenniük a teljes bemenetre. Az információt fokozatosan összesítve, egyre durvább térképeket hozva létre, elérjük ezt a globális reprezentáció megtanulásának célját, miközben megőrizzük a konvolúciós rétegek összes előnyét a feldolgozás közbülső rétegeiben. Minél mélyebbre haladunk a hálózatban, annál nagyobb a receptív mező (a bemenethez viszonyítva), amelyre minden rejtett csomópont érzékeny. A térbeli felbontás csökkentése felgyorsítja ezt a folyamatot, mivel a konvolúciós kernelek nagyobb tényleges területet fednek le.

Ráadásul az alacsonyabb szintű jellemzők, például az élek detektálásakor (amint a :numref:`sec_conv_layer`-ben tárgyaltuk), gyakran szeretnénk, ha reprezentációink némileg invariánsak lennének az eltolásra. Például ha az `X` képet, amelyen éles határ van a fekete és fehér között, egy pixellel jobbra toljuk, azaz `Z[i, j] = X[i, j + 1]`, akkor az új `Z` kép kimenete alapvetően eltérhet. Az él egy pixellel eltolódott. A valóságban az objektumok szinte sohasem jelennek meg pontosan ugyanazon a helyen. Valójában még állványon rögzített kamerával és álló objektummal is az exponálás mozgása miatti kamerarezdülés egy pixellel vagy úgy eltolhat mindent (a csúcskategóriás kamerákat speciális funkciókkal látják el ennek a problémának a kezelésére).

Ez a rész a *pooling rétegeket* mutatja be, amelyek kettős célt szolgálnak: csökkentik a konvolúciós rétegek helyre való érzékenységét, és térbeli lecsökkentést végeznek a reprezentációkon.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

## Maximum pooling és átlagos pooling

A konvolúciós rétegekhez hasonlóan a *pooling* operátorok rögzített alakú ablakból állnak, amely a bemenet összes régiójában végigcsúszik a lépésköz szerint, minden egyes, a rögzített alakú ablak (néha *pooling ablaknak* is nevezett) által bejárt helyen egyetlen kimenetet számítva. Azonban a konvolúciós réteg bemenetei és kernelei közötti keresztkorreláció számítástól eltérően a pooling réteg nem tartalmaz paramétereket (nincs *kernel*). Ehelyett a pooling operátorok determinisztikusak, általában a pooling ablakban lévő elemek maximális vagy átlagos értékét számítják. Ezeket a műveleteket rendre *maximum poolingnak* (*max-poolingnak*) és *átlagos poolingnak* nevezzük.

Az *átlagos pooling* lényegében olyan régi, mint a CNN-ek. Az ötlet hasonló a kép lecsökkentéséhez. Ahelyett, hogy egyszerűen minden második (vagy harmadik) pixelértéket vennénk az alacsonyabb felbontású képhez, szomszédos pixelek átlagát vehetjük, hogy jobb jel-zaj arányú képet kapjunk, mivel több szomszédos pixel információját kombináljuk. A *max-poolinget* :citet:`Riesenhuber.Poggio.1999` vezette be a kognitív idegtudományok kontextusában, leírva, hogyan összesíthető hierarchikusan az információ az objektumdetektálás céljából; korábban már volt egy előzmény a beszédfelismerésben :cite:`Yamaguchi.Sakamoto.Akabane.ea.1990`. Szinte minden esetben a max-pooling előnyösebb az átlagos poolingnál.

Mindkét esetben, csakúgy mint a keresztkorreláció operátornál, a pooling ablakra gondolhatunk úgy, mint amely a bemeneti tenzor bal felső sarkától indul, és balról jobbra, illetve felülről lefelé csúszik végig. A pooling ablak által elért minden egyes helyen a bemeneti altenzor ablakbeli maximális vagy átlagos értékét számítja ki, attól függően, hogy max vagy átlagos poolingot alkalmazunk.


![Max-pooling $2\times 2$ méretű pooling ablakkal. Az árnyékolt részek az első kimeneti elem, valamint a kimeneti számításhoz használt bemeneti tenzorelemek: $\max(0, 1, 3, 4)=4$.](../img/pooling.svg)
:label:`fig_pooling`

A :numref:`fig_pooling`-ban a kimeneti tenzor magassága 2, szélessége 2. A négy elem az egyes pooling ablakokban lévő maximális értékből adódik:

$$
\max(0, 1, 3, 4)=4,\\
\max(1, 2, 4, 5)=5,\\
\max(3, 4, 6, 7)=7,\\
\max(4, 5, 7, 8)=8.\\
$$

Általánosabban, definiálhatunk egy $p \times q$-s pooling réteget az adott méretű régióban való összesítéssel. Visszatérve az éldetektálás problémájához, a konvolúciós réteg kimenetét $2\times 2$-es max-pooling bemenetként használjuk. Jelöljük `X`-szel a konvolúciós réteg bemenetét és `Y`-nal a pooling réteg kimenetét. Függetlenül attól, hogy az `X[i, j]`, `X[i, j + 1]`, `X[i+1, j]` és `X[i+1, j + 1]` értékek különbözők-e, a pooling réteg mindig `Y[i, j] = 1` értéket ad ki. Vagyis a $2\times 2$-es max-pooling réteg segítségével még mindig detektálhatjuk, ha a konvolúciós réteg által felismert minta legfeljebb egy elemmel mozdul el magasságban vagy szélességben.

Az alábbi kódban (**implementáljuk a pooling réteg előreterjesztését**) a `pool2d` függvényben. Ez a függvény hasonló a `corr2d` függvényhez a :numref:`sec_conv_layer`-ben. Azonban nem szükséges kernel, a kimenetet a bemenet minden régiójának maximumaként vagy átlagaként számítjuk.

```{.python .input}
%%tab mxnet, pytorch
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = d2l.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y
```

```{.python .input}
%%tab jax
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = jnp.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y = Y.at[i, j].set(X[i: i + p_h, j: j + p_w].max())
            elif mode == 'avg':
                Y = Y.at[i, j].set(X[i: i + p_h, j: j + p_w].mean())
    return Y
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf

def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = tf.Variable(tf.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w +1)))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j].assign(tf.reduce_max(X[i: i + p_h, j: j + p_w]))
            elif mode =='avg':
                Y[i, j].assign(tf.reduce_mean(X[i: i + p_h, j: j + p_w]))
    return Y
```

A :numref:`fig_pooling`-beli `X` bemeneti tenzort felépíthetjük a [**kétdimenziós max-pooling réteg kimenetének érvényesítéséhez**].

```{.python .input}
%%tab all
X = d2l.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
pool2d(X, (2, 2))
```

Ezenkívül kísérletezhetünk az (**átlagos pooling réteggel**) is.

```{.python .input}
%%tab all
pool2d(X, (2, 2), 'avg')
```

## [**Párnázás és lépésköz**]

A konvolúciós rétegekhez hasonlóan a pooling rétegek is megváltoztatják a kimenet alakját. És csakúgy mint korábban, a bemeneti párnázással és a lépésköz módosításával beállíthatjuk a műveletet, hogy elérjük a kívánt kimeneti alakot. A párnázás és a lépésköz pooling rétegekben való alkalmazását a deep learning keretrendszer beépített kétdimenziós max-pooling rétege segítségével szemléltethetjük. Először létrehozzuk az `X` bemeneti tenzort, amelynek alakja négy dimenzióval rendelkezik, ahol a példák száma (batch méret) és a csatornák száma egyaránt 1.

:begin_tab:`tensorflow`
Vegyük figyelembe, hogy más keretrendszerektől eltérően a TensorFlow preferálja és optimalizálja a *channels-last* bemeneti formátumot.
:end_tab:

```{.python .input}
%%tab mxnet, pytorch
X = d2l.reshape(d2l.arange(16, dtype=d2l.float32), (1, 1, 4, 4))
X
```

```{.python .input}
%%tab tensorflow, jax
X = d2l.reshape(d2l.arange(16, dtype=d2l.float32), (1, 4, 4, 1))
X
```

Mivel a pooling egy területen összesíti az információkat, (**a deep learning keretrendszerek alapértelmezés szerint a pooling ablak méretét és lépésközét egyeztetik.**) Például ha `(3, 3)` alakú pooling ablakot használunk, alapértelmezés szerint `(3, 3)` lépésközt kapunk.

```{.python .input}
%%tab mxnet
pool2d = nn.MaxPool2D(3)
# A poolingnak nincsenek modellparaméterei, így inicializálás sem szükséges
pool2d(X)
```

```{.python .input}
%%tab pytorch
pool2d = nn.MaxPool2d(3)
# A poolingnak nincsenek modellparaméterei, így inicializálás sem szükséges
pool2d(X)
```

```{.python .input}
%%tab tensorflow
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3])
# A poolingnak nincsenek modellparaméterei, így inicializálás sem szükséges
pool2d(X)
```

```{.python .input}
%%tab jax
# A poolingnak nincsenek modellparaméterei, így inicializálás sem szükséges
nn.max_pool(X, window_shape=(3, 3), strides=(3, 3))
```

Természetesen [**a lépésköz és párnázás manuálisan is megadható**] a keretrendszer alapértelmezéseinek felülbírálásához, ha szükséges.

```{.python .input}
%%tab mxnet
pool2d = nn.MaxPool2D(3, padding=1, strides=2)
pool2d(X)
```

```{.python .input}
%%tab pytorch
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
```

```{.python .input}
%%tab tensorflow
paddings = tf.constant([[0, 0], [1,0], [1,0], [0,0]])
X_padded = tf.pad(X, paddings, "CONSTANT")
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3], padding='valid',
                                   strides=2)
pool2d(X_padded)
```

```{.python .input}
%%tab jax
X_padded = jnp.pad(X, ((0, 0), (1, 0), (1, 0), (0, 0)), mode='constant')
nn.max_pool(X_padded, window_shape=(3, 3), padding='VALID', strides=(2, 2))
```

Természetesen tetszőleges téglalap alakú pooling ablakot is megadhatunk tetszőleges magassággal és szélességgel, ahogy az alábbi példa mutatja.

```{.python .input}
%%tab mxnet
pool2d = nn.MaxPool2D((2, 3), padding=(0, 1), strides=(2, 3))
pool2d(X)
```

```{.python .input}
%%tab pytorch
pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
pool2d(X)
```

```{.python .input}
%%tab tensorflow
paddings = tf.constant([[0, 0], [0, 0], [1, 1], [0, 0]])
X_padded = tf.pad(X, paddings, "CONSTANT")

pool2d = tf.keras.layers.MaxPool2D(pool_size=[2, 3], padding='valid',
                                   strides=(2, 3))
pool2d(X_padded)
```

```{.python .input}
%%tab jax

X_padded = jnp.pad(X, ((0, 0), (0, 0), (1, 1), (0, 0)), mode='constant')
nn.max_pool(X_padded, window_shape=(2, 3), strides=(2, 3), padding='VALID')
```

## Több csatorna

Többcsatornás bemeneti adatok feldolgozásakor [**a pooling réteg minden bemeneti csatornát külön-külön pooloz**], nem a konvolúciós réteghez hasonlóan összesíti a bemeneteket a csatornák felett. Ez azt jelenti, hogy a pooling réteg kimeneti csatornáinak száma megegyezik a bemeneti csatornák számával. Az alábbiakban összefűzzük az `X` és `X + 1` tenzorokat a csatorna dimenzióban, hogy két csatornájú bemenetet hozzunk létre.

:begin_tab:`tensorflow`
Vegyük figyelembe, hogy ez a TensorFlow esetén az utolsó dimenzió mentén való összefűzést igényli a channels-last szintaxis miatt.
:end_tab:

```{.python .input}
%%tab mxnet, pytorch
X = d2l.concat((X, X + 1), 1)
X
```

```{.python .input}
%%tab tensorflow, jax
# Összefűzés a `dim=3` dimenzió mentén a channels-last szintaxis miatt
X = d2l.concat([X, X + 1], 3)
X
```

Ahogy látható, a kimeneti csatornák száma pooling után még mindig kettő.

```{.python .input}
%%tab mxnet
pool2d = nn.MaxPool2D(3, padding=1, strides=2)
pool2d(X)
```

```{.python .input}
%%tab pytorch
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
```

```{.python .input}
%%tab tensorflow
paddings = tf.constant([[0, 0], [1,0], [1,0], [0,0]])
X_padded = tf.pad(X, paddings, "CONSTANT")
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3], padding='valid',
                                   strides=2)
pool2d(X_padded)

```

```{.python .input}
%%tab jax
X_padded = jnp.pad(X, ((0, 0), (1, 0), (1, 0), (0, 0)), mode='constant')
nn.max_pool(X_padded, window_shape=(3, 3), padding='VALID', strides=(2, 2))
```

:begin_tab:`tensorflow`
Vegyük figyelembe, hogy a TensorFlow pooling kimenete első pillantásra eltérőnek tűnhet, azonban numerikusan ugyanazok az eredmények jelennek meg, mint az MXNet és PyTorch esetén. A különbség a dimenzionalitásban rejlik, és a kimenet függőleges olvasása ugyanazt a kimenetet adja, mint a többi implementáció.
:end_tab:

## Összefoglalás

A pooling rendkívül egyszerű művelet. Pontosan azt csinálja, amit a neve jelez: összesíti az eredményeket egy értékablak felett. Az összes konvolúciós szemantika, mint a lépésköz és a párnázás, ugyanúgy alkalmazandó, mint korábban. Vegyük figyelembe, hogy a pooling közömbös a csatornákra, azaz változatlanul hagyja a csatornák számát, és minden csatornára külön alkalmazza. Végül a két népszerű pooling választás közül a max-pooling előnyösebb az átlagos poolingnál, mivel bizonyos mértékű invarianciát biztosít a kimenetnek. Népszerű választás a $2 \times 2$-es pooling ablak méret, amely negyedeli a kimenet térbeli felbontását.

Megjegyezzük, hogy a felbontáscsökkentésnek a poolingon kívül sok más módja is van. Például a sztochasztikus poolingban :cite:`Zeiler.Fergus.2013` és a töredékes max-poolingban :cite:`Graham.2014` az összesítés véletlenszerűséggel kombinálódik. Ez bizonyos esetekben kissé javíthatja a pontosságot. Végül, ahogyan majd az figyelemmechanizmusnál látjuk, finomabb módszerek is léteznek a kimenetek összesítésére, például egy lekérdezés és a reprezentációs vektorok közötti illeszkedés alkalmazásával.


## Feladatok

1. Implementáljunk átlagos poolingot konvolúció segítségével.
1. Bizonyítsuk be, hogy a max-pooling nem valósítható meg pusztán konvolúcióval.
1. A max-pooling elvégezhető ReLU műveletekkel, azaz $\textrm{ReLU}(x) = \max(0, x)$.
    1. Fejezzük ki $\max (a, b)$ értékét kizárólag ReLU műveletek segítségével.
    1. Használjuk ezt a max-pooling implementálásához konvolúciók és ReLU rétegek segítségével.
    1. Hány csatornára és rétegre van szükségünk egy $2 \times 2$-es konvolúcióhoz? Hányra egy $3 \times 3$-ashoz?
1. Mi a pooling réteg számítási költsége? Tegyük fel, hogy a pooling réteg bemenete $c\times h\times w$ méretű, a pooling ablak alakja $p_\textrm{h}\times p_\textrm{w}$, $(p_\textrm{h}, p_\textrm{w})$ párnázással és $(s_\textrm{h}, s_\textrm{w})$ lépésközzel.
1. Miért várható, hogy a max-pooling és az átlagos pooling eltérően működik?
1. Szükségünk van-e különálló minimum pooling rétegre? Helyettesíthető-e más művelettel?
1. Használhatnánk a softmax műveletet poolinghoz. Miért nem lehet annyira népszerű?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/71)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/72)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/274)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17999)
:end_tab:
