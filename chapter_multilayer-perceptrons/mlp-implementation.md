```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Többrétegű perceptronok implementálása
:label:`sec_mlp-implementation`

A többrétegű perceptronok (MLP-k) implementálása nem sokkal összetettebb, mint az egyszerű lineáris modelleké. A fő fogalmi különbség az, hogy most több réteget fűzünk össze egymás után.

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

## Implementálás nulláról

Kezdjük ismét azzal, hogy ilyen hálózatot nulláról implementálunk.

### A modell paramétereinek inicializálása

Emlékeztetőül: a Fashion-MNIST 10 osztályt tartalmaz,
és minden kép egy $28 \times 28 = 784$-es
szürkeárnyalatos pixelértéket tartalmazó rácsból áll.
Mint korábban, most is figyelmen kívül hagyjuk
a pixelek közötti térbeli struktúrát,
így ezt egy 784 bemeneti jellemzővel és 10 osztállyal rendelkező
osztályozási adathalmazként kezelhetjük.
Kezdetnek [**implementálunk egy MLP-t
egy rejtett réteggel és 256 rejtett egységgel.**]
A rétegek száma és szélességük egyaránt állítható
(ezeket hiperparamétereknek tekintjük).
Általában a rétegszélességeket a kettő nagyobb hatványaival osztható értékekre választjuk.
Ez számítási szempontból hatékony a memória
hardverben való kiosztásának és kezelésének módja miatt.

A paramétereinket ismét több tenzorral fogjuk jelölni.
Megjegyezzük, hogy *minden rétegnél* nyomon kell követnünk
egy súlymátrixot és egy torzításvektort.
Mint mindig, lefoglalunk memóriát
a veszteség ezen paraméterekre vonatkozó gradienseinek számára.

:begin_tab:`mxnet`
Az alábbi kódban először definiáljuk és inicializáljuk a paramétereket,
majd engedélyezzük a gradiens nyomon követését.
:end_tab:

:begin_tab:`pytorch`
Az alábbi kódban `nn.Parameter`-t használunk
egy osztályattribútum automatikus regisztrálásához
olyan paraméterként, amelyet az `autograd` követ nyomon (:numref:`sec_autograd`).
:end_tab:

:begin_tab:`tensorflow`
Az alábbi kódban `tf.Variable`-t használunk
a modell paramétereinek definiálásához.
:end_tab:

:begin_tab:`jax`
Az alábbi kódban `flax.linen.Module.param`-ot használunk
a modell paramétereinek definiálásához.
:end_tab:

```{.python .input}
%%tab mxnet
class MLPScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = np.random.randn(num_inputs, num_hiddens) * sigma
        self.b1 = np.zeros(num_hiddens)
        self.W2 = np.random.randn(num_hiddens, num_outputs) * sigma
        self.b2 = np.zeros(num_outputs)
        for param in self.get_scratch_params():
            param.attach_grad()
```

```{.python .input}
%%tab pytorch
class MLPScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) * sigma)
        self.b1 = nn.Parameter(torch.zeros(num_hiddens))
        self.W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs) * sigma)
        self.b2 = nn.Parameter(torch.zeros(num_outputs))
```

```{.python .input}
%%tab tensorflow
class MLPScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = tf.Variable(
            tf.random.normal((num_inputs, num_hiddens)) * sigma)
        self.b1 = tf.Variable(tf.zeros(num_hiddens))
        self.W2 = tf.Variable(
            tf.random.normal((num_hiddens, num_outputs)) * sigma)
        self.b2 = tf.Variable(tf.zeros(num_outputs))
```

```{.python .input}
%%tab jax
class MLPScratch(d2l.Classifier):
    num_inputs: int
    num_outputs: int
    num_hiddens: int
    lr: float
    sigma: float = 0.01

    def setup(self):
        self.W1 = self.param('W1', nn.initializers.normal(self.sigma),
                             (self.num_inputs, self.num_hiddens))
        self.b1 = self.param('b1', nn.initializers.zeros, self.num_hiddens)
        self.W2 = self.param('W2', nn.initializers.normal(self.sigma),
                             (self.num_hiddens, self.num_outputs))
        self.b2 = self.param('b2', nn.initializers.zeros, self.num_outputs)
```

### A modell

Annak érdekében, hogy megbizonyosodjunk arról, hogyan működik minden,
[**saját magunk implementáljuk a ReLU aktivációt**],
ahelyett hogy közvetlenül a beépített `relu` függvényt hívnánk meg.

```{.python .input}
%%tab mxnet
def relu(X):
    return np.maximum(X, 0)
```

```{.python .input}
%%tab pytorch
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)
```

```{.python .input}
%%tab tensorflow
def relu(X):
    return tf.math.maximum(X, 0)
```

```{.python .input}
%%tab jax
def relu(X):
    return jnp.maximum(X, 0)
```

Mivel figyelmen kívül hagyjuk a térbeli struktúrát,
minden kétdimenziós képet `reshape`-pel
`num_inputs` hosszúságú lapos vektorrá alakítunk.
Végül (**implementáljuk a modellünket**)
csupán néhány sornyi kóddal. Mivel a keretrendszer beépített autogradját használjuk, ennyi is elegendő.

```{.python .input}
%%tab all
@d2l.add_to_class(MLPScratch)
def forward(self, X):
    X = d2l.reshape(X, (-1, self.num_inputs))
    H = relu(d2l.matmul(X, self.W1) + self.b1)
    return d2l.matmul(H, self.W2) + self.b2
```

### Tanítás

Szerencsére [**az MLP-k tanítási ciklusa
pontosan ugyanolyan, mint a softmax regresszióé.**] Definiáljuk a modellt, az adatokat és a trénert, majd meghívjuk a `fit` metódust a modellen és az adatokon.

```{.python .input}
%%tab all
model = MLPScratch(num_inputs=784, num_outputs=10, num_hiddens=256, lr=0.1)
data = d2l.FashionMNIST(batch_size=256)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

## Tömör implementáció

Ahogy várható, a magas szintű API-kra támaszkodva még tömörebben implementálhatjuk az MLP-ket.

### A modell

A softmax regresszió tömör implementációjával összehasonlítva
(:numref:`sec_softmax_concise`),
az egyetlen különbség az, hogy most *két*
teljesen összekötött réteget adunk hozzá, ahol korábban csak *egyet* adtunk.
Az első [**a rejtett réteg**],
a második a kimeneti réteg.

```{.python .input}
%%tab mxnet
class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential()
        self.net.add(nn.Dense(num_hiddens, activation='relu'),
                     nn.Dense(num_outputs))
        self.net.initialize()
```

```{.python .input}
%%tab pytorch
class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(), nn.LazyLinear(num_hiddens),
                                 nn.ReLU(), nn.LazyLinear(num_outputs))
```

```{.python .input}
%%tab tensorflow
class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_hiddens, activation='relu'),
            tf.keras.layers.Dense(num_outputs)])
```

```{.python .input}
%%tab jax
class MLP(d2l.Classifier):
    num_outputs: int
    num_hiddens: int
    lr: float

    @nn.compact
    def __call__(self, X):
        X = X.reshape((X.shape[0], -1))  # Lapítás
        X = nn.Dense(self.num_hiddens)(X)
        X = nn.relu(X)
        X = nn.Dense(self.num_outputs)(X)
        return X
```

Korábban `forward` metódusokat definiáltunk a modellekhez, hogy a modell paramétereivel transzformálják a bemenetet. Ezek a műveletek lényegében egy csővezetéket alkotnak: fogunk egy bemenetet, és alkalmazunk egy transzformációt (pl. mátrixszorzás súlyokkal, majd torzítás hozzáadása), majd ismételten az aktuális transzformáció kimenetét használjuk a következő transzformáció bemeneteként. Vegyük észre azonban, hogy itt nincs `forward` metódus definiálva. Valójában az `MLP` a `Module` osztályból örökli a `forward` metódust (:numref:`subsec_oo-design-models`), amely egyszerűen meghívja a `self.net(X)`-et (ahol `X` a bemenet), amely most transzformációk sorozataként van definiálva a `Sequential` osztályon keresztül. A `Sequential` osztály absztrahálja az előre irányú folyamatot, lehetővé téve számunkra, hogy a transzformációkra összpontosítsunk. A `Sequential` osztály működéséről bővebben a :numref:`subsec_model-construction-sequential` részben tárgyalunk.


### Tanítás

[**A tanítási ciklus**] pontosan ugyanolyan,
mint a softmax regresszió implementálásakor.
Ez a modularitás lehetővé teszi számunkra, hogy elválasszuk
a modellarchitektúrával kapcsolatos kérdéseket az ortogonális szempontoktól.

```{.python .input}
%%tab all
model = MLP(num_outputs=10, num_hiddens=256, lr=0.1)
trainer.fit(model, data)
```

## Összefoglalás

Most, hogy több tapasztalatunk van mély hálózatok tervezésében, az egyetlen rétegtől a mély hálózatok több rétegéig való átlépés már nem jelent akkora kihívást. Különösen fontos, hogy a tanítási algoritmust és az adatbetöltőt újra fel tudjuk használni. Megjegyezzük azonban, hogy az MLP-k nulláról való implementálása mégis zavaros: a modellparaméterek elnevezése és nyomon követése megnehezíti a modellek bővítését. Képzeljük el például, hogy egy réteget kell beilleszteni a 42. és 43. réteg közé. Ez most a 42b. réteg lehet, hacsak nem vagyunk hajlandók szekvenciális átnevezést végezni. Ráadásul, ha a hálózatot nulláról implementáljuk, a keretrendszer számára sokkal nehezebb értelmes teljesítményoptimalizációkat végezni.

Mindazonáltal most elértük az 1980-as évek végi fejlett állapotot, amikor a teljesen összekötött mély hálózatok voltak a neurális hálózati modellezés választott módszerei. Következő fogalmi lépésünk a képek vizsgálata lesz. Mielőtt ezt tennénk, számos statisztikai alapfogalmat és a modellek hatékony kiszámításának részleteit kell áttekinteni.


## Feladatok

1. Módosítsd a rejtett egységek számát `num_hiddens`, és ábrázold, hogyan befolyásolja a száma a modell pontosságát. Mi ennek a hiperparaméternek a legjobb értéke?
1. Próbálj meg egy rejtett réteget hozzáadni, és nézd meg, hogyan befolyásolja az eredményeket.
1. Miért rossz ötlet egyetlen neuronból álló rejtett réteget beilleszteni? Mi mehet rosszul?
1. Hogyan változtatja meg a tanulási sebesség megváltoztatása az eredményeket? Az összes többi paraméter rögzítése mellett melyik tanulási sebesség adja a legjobb eredményt? Hogyan függ ez össze az epochok számával?
1. Optimalizáljunk az összes hiperparaméter felett együttesen, azaz a tanulási sebesség, az epochok száma, a rejtett rétegek száma és a rétegenként lévő rejtett egységek száma felett.
    1. Mi a legjobb eredmény, amelyet az összes paraméter optimalizálásával elérhetsz?
    1. Miért sokkal kihívóbb több hiperparaméterrel foglalkozni?
    1. Írj le egy hatékony stratégiát több paraméter együttes optimalizálásához.
1. Hasonlítsd össze a keretrendszer és a nulláról való implementáció sebességét egy kihívó problémán. Hogyan változik ez a hálózat összetettségével?
1. Mérd meg a tenzor-mátrix szorzások sebességét jól illeszkedő és nem illeszkedő mátrixok esetén. Például teszteld 1024, 1025, 1026, 1028 és 1032 dimenziójú mátrixok esetén.
    1. Hogyan változik ez GPU-k és CPU-k között?
    1. Határozd meg a CPU és GPU memóriabusz szélességét.
1. Próbálj ki különböző aktivációs függvényeket. Melyik működik a legjobban?
1. Van-e különbség a hálózat súlyinicializálásai között? Számít-e ez?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/92)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/93)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/227)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17985)
:end_tab:
