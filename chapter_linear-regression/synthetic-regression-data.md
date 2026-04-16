```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Szintetikus regressziós adatok
:label:`sec_synthetic-regression-data`


A gépi tanulás lényege az adatokból való információkinyerés.
Így talán felmerül a kérdés: mit tanulhatunk egyáltalán szintetikus adatokból?
Bár önmagában nem feltétlenül érdekel minket az a mintázat,
amelyet mi magunk vittünk be egy mesterségesen generált adatmodellbe,
az ilyen adathalmazok mégis hasznosak didaktikai célokra:
segítenek kiértékelni a tanulási algoritmusaink tulajdonságait,
és megerősíteni, hogy az implementációink elvárásszerűen működnek.
Például, ha olyan adatokat hozunk létre, amelyeknél a helyes paraméterek előre ismertek (*a priori*),
akkor ellenőrizhetjük, hogy a modellünk valóban vissza tudja-e nyerni őket.

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx, gluon
import random
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import random
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import random
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import numpy as np
import random
import tensorflow as tf
import tensorflow_datasets as tfds
```

## Az adathalmaz generálása

Ebben a példában tömörség kedvéért alacsony dimenziókkal dolgozunk.
A következő kódrészlet 1000 példát generál,
amelyek 2-dimenziós jellemzőit standard normális eloszlásból vesszük.
A kapott tervezési mátrix $\mathbf{X}$
a $\mathbb{R}^{1000 \times 2}$ térbe esik.
Minden egyes címkét egy *igaz* lineáris függvény alkalmazásával generálunk,
amelyet egy $\boldsymbol{\epsilon}$ additív zajjal rontunk meg,
amelyet minden példánál egymástól független, azonos eloszlású módon húzunk:

(**$$\mathbf{y}= \mathbf{X} \mathbf{w} + b + \boldsymbol{\epsilon}.$$**)

A kényelem kedvéért feltételezzük, hogy $\boldsymbol{\epsilon}$ 
$\mu= 0$ várható értékű és $\sigma = 0.01$ szórású normális eloszlásból való.
Megjegyezzük, hogy az objektumorientált tervezés érdekében
a kódot a `d2l.DataModule` alosztályának `__init__` metódusába adjuk hozzá (amelyet a :numref:`oo-design-data` részben mutattunk be).
Jó gyakorlat, ha bármely további hiperparaméter beállítása megengedett.
Ezt a `save_hyperparameters()` segítségével érjük el.
A `batch_size` értékét később határozzuk meg.

```{.python .input}
%%tab all
class SyntheticRegressionData(d2l.DataModule):  #@save
    """Szintetikus adatok lineáris regresszióhoz."""
    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000, 
                 batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val
        if tab.selected('pytorch') or tab.selected('mxnet'):                
            self.X = d2l.randn(n, len(w))
            noise = d2l.randn(n, 1) * noise
        if tab.selected('tensorflow'):
            self.X = tf.random.normal((n, w.shape[0]))
            noise = tf.random.normal((n, 1)) * noise
        if tab.selected('jax'):
            key = jax.random.PRNGKey(0)
            key1, key2 = jax.random.split(key)
            self.X = jax.random.normal(key1, (n, w.shape[0]))
            noise = jax.random.normal(key2, (n, 1)) * noise
        self.y = d2l.matmul(self.X, d2l.reshape(w, (-1, 1))) + b + noise
```

Az alábbiakban a valódi paramétereket $\mathbf{w} = [2, -3.4]^\top$ és $b = 4.2$ értékekre állítjuk be.
Később ellenőrizhetjük a becsült paramétereinket ezen *igaz* értékekkel szemben.

```{.python .input}
%%tab all
data = SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
```

[**A `features` minden sora egy vektort tartalmaz $\mathbb{R}^2$-ből, és a `labels` minden sora egy skalár.**] Nézzük meg az első bejegyzést.

```{.python .input}
%%tab all
print('features:', data.X[0],'\nlabel:', data.y[0])
```

## Az adathalmaz beolvasása

A gépi tanulási modellek tanítása gyakran megköveteli az adathalmazon való többszöri áthaladást,
egyszerre egy mini-batch példányt megragadva.
Ezeket az adatokat aztán a modell frissítéséhez használjuk.
Ennek illusztrálásához
[**implementáljuk a `get_dataloader` metódust,**]
amelyet a `SyntheticRegressionData` osztályba regisztrálunk az `add_to_class` segítségével (amelyet a :numref:`oo-design-utilities` részben mutattunk be).
(**Bemenetként egy kötegméretet, egy jellemzőmátrixot
és egy címkevektort vesz, és `batch_size` méretű mini-batch-eket generál.**)
Így minden mini-batch egy jellemzők és címkék párból áll.
Ügyelni kell arra, hogy tanítási vagy validációs módban vagyunk-e:
az előbbiben véletlenszerű sorrendben szeretnénk beolvasni az adatokat,
míg az utóbbinál az adatok előre meghatározott sorrendben való olvasása
hibakeresési szempontból fontos lehet.

```{.python .input}
%%tab all
@d2l.add_to_class(SyntheticRegressionData)
def get_dataloader(self, train):
    if train:
        indices = list(range(0, self.num_train))
        # A példányokat véletlenszerű sorrendben olvassuk be
        random.shuffle(indices)
    else:
        indices = list(range(self.num_train, self.num_train+self.num_val))
    for i in range(0, len(indices), self.batch_size):
        if tab.selected('mxnet', 'pytorch', 'jax'):
            batch_indices = d2l.tensor(indices[i: i+self.batch_size])
            yield self.X[batch_indices], self.y[batch_indices]
        if tab.selected('tensorflow'):
            j = tf.constant(indices[i : i+self.batch_size])
            yield tf.gather(self.X, j), tf.gather(self.y, j)
```

Némi intuíció felépítéséhez vizsgáljuk meg az adatok első mini-batch-ét.
Minden mini-batch jellemzői megmutatják méretét és a bemeneti jellemzők dimenzionalitását.
Hasonlóképpen, a mini-batch címkéi `batch_size` által megadott illeszkedő alakkal rendelkeznek.

```{.python .input}
%%tab all
X, y = next(iter(data.train_dataloader()))
print('X shape:', X.shape, '\ny shape:', y.shape)
```

Bár látszólag ártalmatlan, az
`iter(data.train_dataloader())`
meghívása bemutatja a Python objektumorientált tervezésének erejét.
Vegyük észre, hogy egy metódust adtunk a `SyntheticRegressionData` osztályhoz
*az* `data` objektum *létrehozása után*.
Ennek ellenére az objektum profitál az osztályhoz
utólagosan hozzáadott funkcionalitásból.

Az iteráció során különböző mini-batch-eket kapunk,
amíg az egész adathalmaz ki nem merül (próbáld ki).
Bár a fent implementált iteráció didaktikai szempontból megfelelő,
valós problémáknál nehézségeket okozó hatékonysági hiányosságai vannak.
Például az összes adat memóriába töltését igényli,
és sok véletlenszerű memória-hozzáférést végez.
A mélytanulás keretrendszerek beépített iterátorai
lényegesen hatékonyabbak, és kezelni tudják
az olyan forrásokat is, mint a fájlokban tárolt adatok,
adatok adatfolyamon keresztül fogadva,
és menet közben generált vagy feldolgozott adatok.
Ezután próbáljuk ugyanezt a metódust beépített iterátorok segítségével implementálni.

## Az adatbetöltő tömör implementációja

Saját iterátor megírása helyett
[**meghívhatjuk a keretrendszer meglévő API-ját az adatok betöltéséhez.**]
Mint korábban, szükségünk van egy adathalmazra `X` jellemzőkkel és `y` címkékkel.
Ezenkívül a `batch_size`-t a beépített adatbetöltőben állítjuk be,
és hagyjuk, hogy hatékonyan gondoskodjon a példányok keveréséről.

:begin_tab:`jax`
A JAX lényege a NumPy-szerű API eszközgyorsítással és funkcionális
transzformációkkal, ezért legalább a jelenlegi verzió nem tartalmaz adatbetöltési
metódusokat. Más könyvtárakban már kiváló adatbetöltők állnak rendelkezésre,
és a JAX azok használatát javasolja. Itt a TensorFlow adatbetöltőjét fogjuk felhasználni,
és kissé módosítani, hogy JAX-szal is működjön.
:end_tab:

```{.python .input}
%%tab all
@d2l.add_to_class(d2l.DataModule)  #@save
def get_tensorloader(self, tensors, train, indices=slice(0, None)):
    tensors = tuple(a[indices] for a in tensors)
    if tab.selected('mxnet'):
        dataset = gluon.data.ArrayDataset(*tensors)
        return gluon.data.DataLoader(dataset, self.batch_size,
                                     shuffle=train)
    if tab.selected('pytorch'):
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size,
                                           shuffle=train)
    if tab.selected('jax'):
        # TensorFlow Datasets és Dataloader használata. A JAX vagy Flax nem biztosít
        # adatbetöltési funkciókat
        shuffle_buffer = tensors[0].shape[0] if train else 1
        return tfds.as_numpy(
            tf.data.Dataset.from_tensor_slices(tensors).shuffle(
                buffer_size=shuffle_buffer).batch(self.batch_size))

    if tab.selected('tensorflow'):
        shuffle_buffer = tensors[0].shape[0] if train else 1
        return tf.data.Dataset.from_tensor_slices(tensors).shuffle(
            buffer_size=shuffle_buffer).batch(self.batch_size)
```

```{.python .input}
%%tab all
@d2l.add_to_class(SyntheticRegressionData)  #@save
def get_dataloader(self, train):
    i = slice(0, self.num_train) if train else slice(self.num_train, None)
    return self.get_tensorloader((self.X, self.y), train, i)
```

Az új adatbetöltő pontosan ugyanúgy viselkedik, mint az előző, azzal a különbséggel, hogy hatékonyabb és néhány extra funkcióval rendelkezik.

```{.python .input  n=4}
%%tab all
X, y = next(iter(data.train_dataloader()))
print('X shape:', X.shape, '\ny shape:', y.shape)
```

Például a keretrendszer API által biztosított adatbetöltő
támogatja a beépített `__len__` metódust,
így lekérdezhetjük a hosszát,
azaz a kötegek számát.

```{.python .input}
%%tab all
len(data.train_dataloader())
```

## Összefoglalás

Az adatbetöltők kényelmes módot biztosítanak az adatok betöltési
és manipulációs folyamatának absztrahálására.
Így ugyanaz a gépi tanulási *algoritmus*
képes sokféle adattípust és adatforrást feldolgozni
módosítás nélkül.
Az adatbetöltők egyik szép tulajdonsága,
hogy összeilleszthetők.
Például betölthetünk képeket,
majd egy utófeldolgozó szűrővel
levághatjuk vagy más módon módosíthatjuk őket.
Az adatbetöltők tehát felhasználhatók
egy teljes adatfeldolgozási csővezeték leírására.

Ami magát a modellt illeti, a kétdimenziós lineáris modell
az egyik legegyszerűbb, amellyel találkozhatunk.
Lehetővé teszi a regressziós modellek pontosságának tesztelését
anélkül, hogy aggódnánk az adatok elégtelensége
vagy az alulhatározott egyenletrendszer miatt.
Ezt a következő részben jól hasznosítjuk.


## Feladatok

1. Mi történik, ha a példányok száma nem osztható a kötegmérettel? Hogyan változtatnád meg ezt a viselkedést a keretrendszer API-jának egy másik argumentumával?
1. Tegyük fel, hogy hatalmas adathalmazt szeretnénk generálni, ahol mind a `w` paramétervektora mérete, mind a `num_examples` példányok száma nagy.
    1. Mi történik, ha az összes adat nem fér be a memóriába?
    1. Hogyan kevernéd össze az adatokat, ha azok lemezen vannak tárolva? A feladatod egy *hatékony* algoritmus tervezése, amely nem igényel túl sok véletlenszerű olvasást vagy írást. Tipp: a [pszeudovéletlen permutációgenerátorok](https://en.wikipedia.org/wiki/Pseudorandom_permutation) lehetővé teszik egy újrakeverés tervezését anélkül, hogy explicit módon tárolni kellene a permutációs táblát :cite:`Naor.Reingold.1999`.
1. Implementálj egy adatgenerátort, amely menet közben generál új adatokat minden alkalommal, amikor az iterátort meghívják.
1. Hogyan terveznél egy véletlenszerű adatgenerátort, amely minden híváskor *ugyanazokat* az adatokat generálja?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/6662)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/6663)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/6664)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17975)
:end_tab:
