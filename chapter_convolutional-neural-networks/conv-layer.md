```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Konvolúciók képekhez
:label:`sec_conv_layer`

Most, hogy elméletileg megértettük, hogyan működnek a konvolúciós rétegek, készen állunk arra, hogy lássuk, hogyan működnek a gyakorlatban. A konvolúciós neurális hálózatok képadatokban lévő struktúrák feltárásának hatékony architektúrájaként való motivációjára építve, továbbra is a képeket tartjuk fő példánknak.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
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

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

## A keresztkorreláció művelet

Emlékeztető: szigorúan véve a konvolúciós rétegek elnevezése nem pontos, mivel az általuk kifejezett műveletek pontosabban keresztkorrelációként írhatók le. A konvolúciós rétegekről szóló leírásaink alapján (:numref:`sec_why-conv`), egy ilyen rétegben egy bemeneti tenzor és egy kernel tenzor kombinálódik egy kimeneti tenzor előállításához **keresztkorreláció műveleten** keresztül.

Hagyjuk egyelőre figyelmen kívül a csatornákat, és lássuk, hogyan működik ez kétdimenziós adatokkal és rejtett reprezentációkkal. A :numref:`fig_correlation`-ban a bemenet egy kétdimenziós tenzor, amelynek magassága 3 és szélessége 3. A tenzor alakját $3 \times 3$ vagy ($3$, $3$) jelöli. A kernel magassága és szélessége egyaránt 2. A *kernel-ablak* (vagy *konvolúciós ablak*) alakját a kernel magassága és szélessége adja (itt $2 \times 2$).

![Kétdimenziós keresztkorreláció művelet. Az árnyékolt részek az első kimeneti elem, valamint a kimeneti számításhoz használt bemeneti és kernel tenzorelemek: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)
:label:`fig_correlation`

A kétdimenziós keresztkorreláció műveletben a konvolúciós ablakkal a bemeneti tenzor bal felső sarkától kezdjük, és végigsiklunk a bemeneti tenzoron, balról jobbra és felülről lefelé egyaránt. Amikor a konvolúciós ablak egy bizonyos pozícióba csúszik, az abban az ablakban lévő bemeneti alterenzort és a kernel tenzort elemenként megszorozzuk, és az eredő tenzort összegezzük, egyetlen skaláris értéket kapva. Ez az eredmény adja a kimeneti tenzor értékét a megfelelő helyen. Az $2 \times 2$ kimeneti tenzor négy eleme a kétdimenziós keresztkorreláció műveletből adódik:

$$
0\times0+1\times1+3\times2+4\times3=19,\\
1\times0+2\times1+4\times2+5\times3=25,\\
3\times0+4\times1+6\times2+7\times3=37,\\
4\times0+5\times1+7\times2+8\times3=43.
$$

Vegyük figyelembe, hogy minden tengely mentén a kimeneti méret kissé kisebb, mint a bemeneti méret. Mivel a kernelnek 1-nél nagyobb szélessége és magassága van, a keresztkorrelációt csak olyan helyeken tudjuk megfelelően kiszámítani, ahol a kernel teljesen belefér a képbe; a kimeneti méret a bemeneti méret $n_\textrm{h} \times n_\textrm{w}$ mínusz a konvolúciós kernel mérete $k_\textrm{h} \times k_\textrm{w}$:

$$(n_\textrm{h}-k_\textrm{h}+1) \times (n_\textrm{w}-k_\textrm{w}+1).$$

Ez azért van, mert elegendő helyre van szükségünk a konvolúciós kernel képen való "eltolásához". Később látni fogjuk, hogyan tartható változatlan a méret, ha nullákkal párnázzuk a képet a határain, hogy elegendő hely legyen a kernel eltolásához. Ezután ezt a folyamatot a `corr2d` függvényben implementáljuk, amely egy `X` bemeneti tenzort és egy `K` kernel tenzort fogad el, és egy `Y` kimeneti tenzort ad vissza.

```{.python .input}
%%tab mxnet
def corr2d(X, K):  #@save
    """Kétdimenziós keresztkorreláció kiszámítása."""
    h, w = K.shape
    Y = d2l.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = d2l.reduce_sum((X[i: i + h, j: j + w] * K))
    return Y
```

```{.python .input}
%%tab pytorch
def corr2d(X, K):  #@save
    """Kétdimenziós keresztkorreláció kiszámítása."""
    h, w = K.shape
    Y = d2l.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = d2l.reduce_sum((X[i: i + h, j: j + w] * K))
    return Y
```

```{.python .input}
%%tab jax
def corr2d(X, K):  #@save
    """Kétdimenziós keresztkorreláció kiszámítása."""
    h, w = K.shape
    Y = jnp.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y = Y.at[i, j].set((X[i:i + h, j:j + w] * K).sum())
    return Y
```

```{.python .input}
%%tab tensorflow
def corr2d(X, K):  #@save
    """Kétdimenziós keresztkorreláció kiszámítása."""
    h, w = K.shape
    Y = tf.Variable(tf.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1)))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j].assign(tf.reduce_sum(
                X[i: i + h, j: j + w] * K))
    return Y
```

A :numref:`fig_correlation`-ból felépíthetjük az `X` bemeneti tenzort és a `K` kernel tenzort, hogy [**érvényesítsük a kétdimenziós keresztkorreláció művelet fenti implementációjának kimenetét**].

```{.python .input}
%%tab all
X = d2l.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = d2l.tensor([[0.0, 1.0], [2.0, 3.0]])
corr2d(X, K)
```

## Konvolúciós rétegek

Egy konvolúciós réteg keresztkorrelációt számít a bemenet és a kernel között, és skaláris torzítást ad hozzá a kimenet előállításához. A konvolúciós réteg két paramétere a kernel és a skaláris torzítás. Konvolúciós rétegeken alapuló modellek tanításakor általában véletlenszerűen inicializáljuk a kerneleket, ahogyan egy teljesen összekötött réteggel is tennénk.

Most készen állunk egy [**kétdimenziós konvolúciós réteg implementálására**] a fent definiált `corr2d` függvény alapján. A `__init__` konstruktor metódusban a `weight`-et és a `bias`-t két modell paraméterként deklaráljuk. Az előreterjesztési metódus meghívja a `corr2d` függvényt és hozzáadja a torzítást.

```{.python .input}
%%tab mxnet
class Conv2D(nn.Block):
    def __init__(self, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.weight = self.params.get('weight', shape=kernel_size)
        self.bias = self.params.get('bias', shape=(1,))

    def forward(self, x):
        return corr2d(x, self.weight.data()) + self.bias.data()
```

```{.python .input}
%%tab pytorch
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
```

```{.python .input}
%%tab tensorflow
class Conv2D(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, kernel_size):
        initializer = tf.random_normal_initializer()
        self.weight = self.add_weight(name='w', shape=kernel_size,
                                      initializer=initializer)
        self.bias = self.add_weight(name='b', shape=(1, ),
                                    initializer=initializer)

    def call(self, inputs):
        return corr2d(inputs, self.weight) + self.bias
```

```{.python .input}
%%tab jax
class Conv2D(nn.Module):
    kernel_size: int

    def setup(self):
        self.weight = nn.param('w', nn.initializers.uniform, self.kernel_size)
        self.bias = nn.param('b', nn.initializers.zeros, 1)

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
```

$h \times w$ konvolúcióban vagy $h \times w$ konvolúciós kernelben a konvolúciós kernel magassága és szélessége rendre $h$ és $w$. Egy $h \times w$ konvolúciós kernelt tartalmazó konvolúciós réteget egyszerűen $h \times w$-es konvolúciós rétegnek is nevezünk.


## Objektumél-detektálás képekben

Szánjunk egy pillanatot egy [**konvolúciós réteg egyszerű alkalmazásának elemzésére: egy kép objektumélének detektálása**] a pixelváltozás helyének megtalálásával. Először felépítünk egy $6\times 8$ pixeles "képet". A középső négy oszlop fekete ($0$), a többi fehér ($1$).

```{.python .input}
%%tab mxnet, pytorch
X = d2l.ones((6, 8))
X[:, 2:6] = 0
X
```

```{.python .input}
%%tab tensorflow
X = tf.Variable(tf.ones((6, 8)))
X[:, 2:6].assign(tf.zeros(X[:, 2:6].shape))
X
```

```{.python .input}
%%tab jax
X = jnp.ones((6, 8))
X = X.at[:, 2:6].set(0)
X
```

Ezután felépítünk egy `K` kernelt 1-es magassággal és 2-es szélességgel. Amikor a keresztkorreláció műveletet elvégezzük a bemenettel, ha a vízszintesen szomszédos elemek azonosak, a kimenet 0. Ellenkező esetben a kimenet nem nulla. Megjegyezzük, hogy ez a kernel egy véges differencia operátor speciális esete. Az $(i,j)$ helyen a $x_{i,j} - x_{(i+1),j}$ értéket számítja, azaz a vízszintesen szomszédos pixelek értékei közötti különbséget. Ez a vízszintes irányban vett első derivált diszkrét közelítése. Végül is, egy $f(i,j)$ függvény esetén a deriváltja $-\partial_i f(i,j) = \lim_{\epsilon \to 0} \frac{f(i,j) - f(i+\epsilon,j)}{\epsilon}$. Lássuk, hogyan működik ez a gyakorlatban.

```{.python .input}
%%tab all
K = d2l.tensor([[1.0, -1.0]])
```

Most készen állunk a keresztkorreláció művelet elvégzésére az `X` (bemenetünk) és `K` (kernelünk) argumentumokkal. Ahogyan látható, [**$1$-et detektálunk a fehérről feketére való élen, és $-1$-et a feketéről fehérre való élen.**] Minden más kimenet $0$ értéket vesz fel.

```{.python .input}
%%tab all
Y = corr2d(X, K)
Y
```

Most alkalmazhatjuk a kernelt a transzponált képre. Ahogy várható, eltűnik. [**A `K` kernel csak függőleges éleket detektál.**]

```{.python .input}
%%tab all
corr2d(d2l.transpose(X), K)
```

## Kernel tanítása

Egy éldetektort véges differenciák `[1, -1]` segítségével tervezni elegáns megoldás, ha tudjuk, hogy pontosan ezt keressük. Azonban, ha nagyobb kerneleket vizsgálunk, és konvolúciókat egymásra következő rétegeit vesszük figyelembe, lehetetlen lehet pontosan meghatározni, hogy mit csináljon manuálisan minden szűrő.

Most lássuk, meg tudjuk-e [**tanulni azt a kernelt, amely `Y`-t generálta `X`-ből**], csak a bemenet-kimenet párok vizsgálatával. Először felépítünk egy konvolúciós réteget, és kernelét véletlenszerű tenzorként inicializáljuk. Ezután minden iterációban a négyzethibát fogjuk használni a `Y` és a konvolúciós réteg kimenetének összehasonlításához. Majd kiszámíthatjuk a gradienst a kernel frissítéséhez. Az egyszerűség kedvéért a következőkben a kétdimenziós konvolúciós rétegek beépített osztályát használjuk, és figyelmen kívül hagyjuk a torzítást.

```{.python .input}
%%tab mxnet
# Kétdimenziós konvolúciós réteget építünk 1 kimeneti csatornával és
# (1, 2) alakú kernellel. Az egyszerűség kedvéért a torzítást figyelmen kívül hagyjuk
conv2d = nn.Conv2D(1, kernel_size=(1, 2), use_bias=False)
conv2d.initialize()

# A kétdimenziós konvolúciós réteg négydimenzós be- és kimenetet használ
# (példa, csatorna, magasság, szélesség) formátumban, ahol a batch mérete
# (a batchben lévő példák száma) és a csatornák száma egyaránt 1
X = X.reshape(1, 1, 6, 8)
Y = Y.reshape(1, 1, 6, 7)
lr = 3e-2  # Tanulási ráta

for i in range(10):
    with autograd.record():
        Y_hat = conv2d(X)
        l = (Y_hat - Y) ** 2
    l.backward()
    # A kernel frissítése
    conv2d.weight.data()[:] -= lr * conv2d.weight.grad()
    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {float(l.sum()):.3f}')
```

```{.python .input}
%%tab pytorch
# Kétdimenziós konvolúciós réteget építünk 1 kimeneti csatornával és
# (1, 2) alakú kernellel. Az egyszerűség kedvéért a torzítást figyelmen kívül hagyjuk
conv2d = nn.LazyConv2d(1, kernel_size=(1, 2), bias=False)

# A kétdimenziós konvolúciós réteg négydimenzós be- és kimenetet használ
# (példa, csatorna, magasság, szélesség) formátumban, ahol a batch mérete
# (a batchben lévő példák száma) és a csatornák száma egyaránt 1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2  # Tanulási ráta

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # A kernel frissítése
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {l.sum():.3f}')
```

```{.python .input}
%%tab tensorflow
# Kétdimenziós konvolúciós réteget építünk 1 kimeneti csatornával és
# (1, 2) alakú kernellel. Az egyszerűség kedvéért a torzítást figyelmen kívül hagyjuk
conv2d = tf.keras.layers.Conv2D(1, (1, 2), use_bias=False)

# A kétdimenziós konvolúciós réteg négydimenzós be- és kimenetet használ
# (példa, magasság, szélesség, csatorna) formátumban, ahol a batch mérete
# (a batchben lévő példák száma) és a csatornák száma egyaránt 1
X = tf.reshape(X, (1, 6, 8, 1))
Y = tf.reshape(Y, (1, 6, 7, 1))
lr = 3e-2  # Tanulási ráta

Y_hat = conv2d(X)
for i in range(10):
    with tf.GradientTape(watch_accessed_variables=False) as g:
        g.watch(conv2d.weights[0])
        Y_hat = conv2d(X)
        l = (abs(Y_hat - Y)) ** 2
        # A kernel frissítése
        update = tf.multiply(lr, g.gradient(l, conv2d.weights[0]))
        weights = conv2d.get_weights()
        weights[0] = conv2d.weights[0] - update
        conv2d.set_weights(weights)
        if (i + 1) % 2 == 0:
            print(f'epoch {i + 1}, loss {tf.reduce_sum(l):.3f}')
```

```{.python .input}
%%tab jax
# Kétdimenziós konvolúciós réteget építünk 1 kimeneti csatornával és
# (1, 2) alakú kernellel. Az egyszerűség kedvéért a torzítást figyelmen kívül hagyjuk
conv2d = nn.Conv(1, kernel_size=(1, 2), use_bias=False, padding='VALID')

# A kétdimenziós konvolúciós réteg négydimenzós be- és kimenetet használ
# (példa, magasság, szélesség, csatorna) formátumban, ahol a batch mérete
# (a batchben lévő példák száma) és a csatornák száma egyaránt 1
X = X.reshape((1, 6, 8, 1))
Y = Y.reshape((1, 6, 7, 1))
lr = 3e-2  # Tanulási ráta

params = conv2d.init(jax.random.PRNGKey(d2l.get_seed()), X)

def loss(params, X, Y):
    Y_hat = conv2d.apply(params, X)
    return ((Y_hat - Y) ** 2).sum()

for i in range(10):
    l, grads = jax.value_and_grad(loss)(params, X, Y)
    # A kernel frissítése
    params = jax.tree_map(lambda p, g: p - lr * g, params, grads)
    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {l:.3f}')
```

Vegyük figyelembe, hogy a hiba 10 iteráció után kis értékre csökkent. Most [**megnézzük a tanult kernel tenzort.**]

```{.python .input}
%%tab mxnet
d2l.reshape(conv2d.weight.data(), (1, 2))
```

```{.python .input}
%%tab pytorch
d2l.reshape(conv2d.weight.data, (1, 2))
```

```{.python .input}
%%tab tensorflow
d2l.reshape(conv2d.get_weights()[0], (1, 2))
```

```{.python .input}
%%tab jax
params['params']['kernel'].reshape((1, 2))
```

Valóban, a tanult kernel tenzor figyelemreméltóan közel van a korábban definiált `K` kernel tenzorhoz.

## Keresztkorreláció és konvolúció

Emlékeztessük magunkat a :numref:`sec_why-conv`-ban tett megfigyelésünkre a keresztkorreláció és konvolúció műveletek közötti megfelelésről. Folytassuk a kétdimenziós konvolúciós rétegek vizsgálatát. Mi lenne, ha ezek a rétegek az :eqref:`eq_2d-conv-discrete`-ban definiált szigorú konvolúciós műveleteket végeznék a keresztkorrelációk helyett? A szigorú *konvolúció* művelet kimenetének megszerzéséhez csak vízszintesen és függőlegesen is meg kell tükröznünk a kétdimenziós kernel tenzort, majd a bemeneti tenzorral elvégeznünk a *keresztkorreláció* műveletet.

Figyelemre méltó, hogy mivel a kerneleket adatokból tanítják mélytanulásban, a konvolúciós rétegek kimenetei változatlanok maradnak, függetlenül attól, hogy ezek a rétegek szigorú konvolúciós műveleteket vagy keresztkorreláció műveleteket végeznek.

Ennek szemléltetéséhez tegyük fel, hogy egy konvolúciós réteg *keresztkorrelációt* végez, és megtanulja a :numref:`fig_correlation`-ban szereplő kernelt, amelyet itt $\mathbf{K}$ mátrixként jelölünk. Feltételezve, hogy más feltételek változatlanok maradnak, ha ez a réteg ehelyett szigorú *konvolúciót* végez, a tanult $\mathbf{K}'$ kernel ugyanolyan lesz, mint a $\mathbf{K}$, miután $\mathbf{K}'$-t vízszintesen és függőlegesen is megfordítottuk. Vagyis, amikor a konvolúciós réteg szigorú *konvolúciót* végez a :numref:`fig_correlation`-beli bemenetre és $\mathbf{K}'$-re, ugyanolyan kimenetet kapunk, mint a :numref:`fig_correlation`-ban (a bemenet és $\mathbf{K}$ keresztkorrelációja).

A mélytanulási irodalom bevett terminológiájával összhangban a keresztkorreláció műveletet tovább konvolúciónak nevezzük, bár szigorúan véve kissé különbözik. Ezenkívül az *elem* kifejezést egy réteget megjelenítő tenzor vagy konvolúciós kernel bármely bejegyzésének (vagy összetevőjének) jelölésére használjuk.


## Jellemzőtérkép és receptív mező

Ahogy a :numref:`subsec_why-conv-channels`-ban leírták, a :numref:`fig_correlation`-beli konvolúciós réteg kimenetét néha *jellemzőtérképnek* is nevezik, mivel a következő réteg térbeli dimenzióiban (pl. szélesség és magasság) tanult reprezentációknak (jellemzőknek) tekinthető. A CNN-ekben bármely réteg bármely $x$ elemére vonatkozóan a *receptív mező* az összes (az összes előző rétegből vett) olyan elemet jelöli, amely befolyásolhatja $x$ kiszámítását az előreterjesztés során. Vegyük figyelembe, hogy a receptív mező nagyobb lehet a tényleges bemeneti méretnél.

Folytassuk a :numref:`fig_correlation` használatát a receptív mező magyarázatához. Az adott $2 \times 2$-es konvolúciós kernel esetén az árnyékolt kimeneti elem receptív mezeje (értéke $19$) a bemenet árnyékolt részének négy eleme. Most jelöljük a $2 \times 2$-es kimenetet $\mathbf{Y}$-ként, és vizsgáljunk egy mélyebb CNN-t egy további $2 \times 2$-es konvolúciós réteggel, amely $\mathbf{Y}$-t veszi bemenetként, és egyetlen $z$ elemet ad ki. Ebben az esetben $z$ receptív mezeje $\mathbf{Y}$-n $\mathbf{Y}$ összes négy elemét tartalmazza, míg a bemeneten a receptív mező az összes kilenc bemeneti elemet tartalmazza. Így, amikor egy jellemzőtérkép bármely eleméhez nagyobb receptív mező szükséges a bemeneti jellemzők szélesebb területen való detektálásához, mélyebb hálózatot építhetünk.

A receptív mezők nevüket a neurofiziológiából kapták. Különböző ingereket alkalmazó, különböző állatokon végzett kísérletek sorozata :cite:`Hubel.Wiesel.1959,Hubel.Wiesel.1962,Hubel.Wiesel.1968` megvizsgálta a vizuális kéreg nevű struktúra válaszát ezekre az ingerekre. Általánosságban azt találták, hogy az alacsonyabb szintek élekre és kapcsolódó alakzatokra reagálnak. Később :citet:`Field.1987` természetes képeken szemléltette ezt a hatást, ami csak konvolúciós kerneleknek nevezhető eszközökkel. Egy kulcsfontosságú ábrát újranyomtatunk a :numref:`field_visual`-ban, hogy szemléltessük a megdöbbentő hasonlóságokat.

![Ábra és felirat :citet:`Field.1987`-ből: Kódolás példája hat különböző csatornával. (Bal) Példák az egyes csatornákhoz tartozó hat érzékelőtípusra. (Jobb) A (Középső) képnek a (Bal oldalon) látható hat érzékelővel való konvolúciója. Az egyes érzékelők válaszát ezeknek a szűrt képeknek az érzékelő méretével arányos távolságon (pontokkal jelölve) való mintavételével határozzuk meg. Ez az ábra csak a páros szimmetrikus érzékelők válaszát mutatja.](../img/field-visual.png)
:label:`field_visual`

Mint kiderül, ez az összefüggés még a képosztályozási feladatokon tanított hálózatok mélyebb rétegei által kiszámított jellemzőkre is érvényes, ahogyan azt például :citet:`Kuzovkin.Vicente.Petton.ea.2018` bemutatta. Elég annyit mondani, hogy a konvolúciók hihetetlenül hatékony eszköznek bizonyultak a számítógépes látásban, mind a biológiában, mind a kódban. Mint ilyen, nem meglepő (visszatekintve), hogy előhírnökei voltak a mélytanulás közelmúltbeli sikerének.

## Összefoglalás

A konvolúciós réteghez szükséges alapvető számítás egy keresztkorreláció művelet. Láttuk, hogy az értékének kiszámításához csak egy egyszerű beágyazott for-ciklus szükséges. Ha több bemeneti és több kimeneti csatornánk van, mátrix-mátrix műveletet végzünk a csatornák között. Ahogy látható, a számítás egyszerű, és ami a legfontosabb, erősen *lokális*. Ez jelentős hardveres optimalizálást tesz lehetővé, és sok közelmúltbeli számítógépes látási eredmény csak ennek köszönhetően lehetséges. Végül is ez azt jelenti, hogy a chipek tervezői gyors számításokba fektethetnek be a memória helyett, amikor konvolúciókra optimalizálnak. Bár ez más alkalmazások esetén nem feltétlenül vezet optimális megoldásokhoz, megnyitja az utat az általánosan elérhető és megfizethető számítógépes látás előtt.

Maguk a konvolúciók sok célra felhasználhatók, például élek és vonalak detektálására, képek elmosódítására vagy élesítésére. A legfontosabb, hogy a statisztikusnak (vagy mérnöknek) nem szükséges kézzel megalkotni a megfelelő szűrőket. Ehelyett egyszerűen *megtanulhatjuk* azokat az adatokból. Ez a jellemzőmérnöklési heurisztikákat bizonyítékalapú statisztikával váltja fel. Végül, és igen örömtelien, ezek a szűrők nem csupán a mély hálózatok építéséhez előnyösek, hanem az agyban lévő receptív mezőknek és jellemzőtérképeknek is megfelelnek. Ez megerősíti bennünk, hogy jó úton haladunk.

## Feladatok

1. Építsünk egy átlós éleket tartalmazó `X` képet.
    1. Mi történik, ha erre alkalmazzuk ebben a részben szereplő `K` kernelt?
    1. Mi történik, ha transzponáljuk `X`-et?
    1. Mi történik, ha transzponáljuk `K`-t?
1. Tervezzünk néhány kernelt manuálisan.
    1. Egy adott $\mathbf{v} = (v_1, v_2)$ irányvektorból kiindulva vezessük le az éldetektáló kernelt, amely a $\mathbf{v}$-re merőleges éleket detektálja, azaz a $(v_2, -v_1)$ irányú éleket.
    1. Vezessük le a második derivált véges differencia operátorát. Mi az ehhez kapcsolódó konvolúciós kernel minimális mérete? Mely képi struktúrák reagálnak rá a legerőteljesebben?
    1. Hogyan terveznénk egy elmosódítási kernelt? Miért érdemes ilyen kernelt használni?
    1. Mi a minimum mérete egy kernelnek egy $d$-edrendű derivált eléréséhez?
1. Amikor megpróbáljuk automatikusan megtalálni a gradienst az általunk létrehozott `Conv2D` osztályhoz, milyen hibaüzenetet kapunk?
1. Hogyan lehet a keresztkorreláció műveletet mátrixszorzásként ábrázolni a bemeneti és kernel tenzorok megváltoztatásával?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/65)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/66)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/271)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17996)
:end_tab:
