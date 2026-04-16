```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Dokumentáció
:begin_tab:`mxnet`
Bár nem tudjuk bemutatni az összes MXNet függvényt és osztályt
(és az információk gyorsan elavulhatnak),
az [API dokumentáció](https://mxnet.apache.org/versions/1.8.0/api)
és a kiegészítő [oktatóanyagok](https://mxnet.apache.org/versions/1.8.0/api/python/docs/tutorials/) és példák
biztosítják ezt.
Ez a szakasz útmutatást nyújt az MXNet API feltérképezéséhez.
:end_tab:

:begin_tab:`pytorch`
Bár nem tudjuk bemutatni az összes PyTorch függvényt és osztályt
(és az információk gyorsan elavulhatnak),
az [API dokumentáció](https://pytorch.org/docs/stable/index.html) és a kiegészítő [oktatóanyagok](https://pytorch.org/tutorials/beginner/basics/intro.html) és példák
biztosítják ezt.
Ez a szakasz útmutatást nyújt a PyTorch API feltérképezéséhez.
:end_tab:

:begin_tab:`tensorflow`
Bár nem tudjuk bemutatni az összes TensorFlow függvényt és osztályt
(és az információk gyorsan elavulhatnak),
az [API dokumentáció](https://www.tensorflow.org/api_docs) és a kiegészítő [oktatóanyagok](https://www.tensorflow.org/tutorials) és példák
biztosítják ezt.
Ez a szakasz útmutatást nyújt a TensorFlow API feltérképezéséhez.
:end_tab:

```{.python .input}
%%tab mxnet
from mxnet import np
```

```{.python .input}
%%tab pytorch
import torch
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
```

```{.python .input}
%%tab jax
import jax
```

## Függvények és osztályok egy modulban

Annak megismeréséhez, hogy egy modulban milyen függvények és osztályok hívhatók,
a `dir` függvényt alkalmazzuk. Például
(**lekérdezhetjük a véletlenszám-generáláshoz szükséges modul összes tulajdonságát**):

```{.python .input  n=1}
%%tab mxnet
print(dir(np.random))
```

```{.python .input  n=1}
%%tab pytorch
print(dir(torch.distributions))
```

```{.python .input  n=1}
%%tab tensorflow
print(dir(tf.random))
```

```{.python .input}
%%tab jax
print(dir(jax.random))
```

Általában figyelmen kívül hagyhatjuk azokat a függvényeket, amelyek `__` karakterekkel kezdődnek és végződnek (ezek a Python speciális objektumai),
illetve a `_` karakterrel kezdődőket (amelyek általában belső függvények).
A maradék függvény- és attribútumnevekből következtethetünk arra,
hogy a modul különféle módszereket kínál véletlenszámok generálásához,
beleértve az egyenletes eloszlásból (`uniform`),
a normális eloszlásból (`normal`) és a multinomiális eloszlásból (`multinomial`) való mintavételt.

## Konkrét függvények és osztályok

Egy adott függvény vagy osztály használatának részletes leírásához
a `help` függvényt hívjuk meg. Például
[**nézzük meg a tenzorok `ones` függvényének használati útmutatóját**].

```{.python .input}
%%tab mxnet
help(np.ones)
```

```{.python .input}
%%tab pytorch
help(torch.ones)
```

```{.python .input}
%%tab tensorflow
help(tf.ones)
```

```{.python .input}
%%tab jax
help(jax.numpy.ones)
```

A dokumentációból láthatjuk, hogy az `ones` függvény
egy új, a megadott alakú tenzort hoz létre,
amelynek minden elemét 1-re állítja.
Valahányszor lehetséges, végezz (**gyors tesztet**)
az értelmezésed megerősítéséhez:

```{.python .input}
%%tab mxnet
np.ones(4)
```

```{.python .input}
%%tab pytorch
torch.ones(4)
```

```{.python .input}
%%tab tensorflow
tf.ones(4)
```

```{.python .input}
%%tab jax
jax.numpy.ones(4)
```

A Jupyter notebookban a `?` karakterrel egy másik ablakban jeleníthetjük meg a dokumentációt.
Például a `list?` a `help(list)` tartalmával majdnem azonos szöveget hoz létre,
amelyet egy új böngészőablakban jelenít meg.
Emellett ha két kérdőjelet használunk, például `list??`,
megjelenik a függvényt megvalósító Python kód is.

A hivatalos dokumentáció számos leírást és példát tartalmaz,
amelyek meghaladják e könyv kereteit.
Mi a fontos felhasználási eseteket emeljük ki,
amelyek segítenek gyorsan elindulni a gyakorlati problémák megoldásában,
nem a teljes lefedettségre törekszünk.
Arra is bátorítunk, hogy tanulmányozd a könyvtárak forráskódját,
hogy magas minőségű, termelési kódra vonatkozó megvalósítási példákat láss.
Ezzel nemcsak jobb tudóssá, hanem jobb mérnökké is válhatsz.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/38)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/39)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/199)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17972)
:end_tab:
