```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Adatelőfeldolgozás
:label:`sec_pandas`

Eddig szintetikus adatokkal dolgoztunk,
amelyek kész tenzorok formájában érkeztek.
Azonban a deep learning valós alkalmazásához
rendezetlen, tetszőleges formátumban tárolt adatokat
kell kinyernünk és igényeinknek megfelelően
előfeldolgoznunk.
Szerencsére a *pandas* [könyvtár](https://pandas.pydata.org/)
elvégzi a munka nagy részét.
Ez a szakasz – bár nem pótol egy alapos *pandas*
[oktatóanyagot](https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html) –
gyors áttekintést nyújt a leggyakoribb műveletekről.

## Az adathalmaz beolvasása

A vesszővel elválasztott értékeket tartalmazó (CSV) fájlok széles körben elterjedtek
táblázatos (táblázatszerű) adatok tárolására.
Bennük minden sor egy rekordnak felel meg,
és több (vesszővel elválasztott) mezőből áll, pl.:
„Albert Einstein,1879. március 14.,Ulm,Szövetségi műszaki főiskola,gravitációs fizika".
Annak bemutatásához, hogyan tölthetünk be CSV fájlokat a `pandas` segítségével,
(**az alábbiakban létrehozzuk a**) `../data/house_tiny.csv` **CSV fájlt**.
Ez a fájl egy lakásokból álló adathalmazt ábrázol,
ahol minden sor egy különálló lakásnak felel meg,
az oszlopok pedig a szobák számát (`NumRooms`),
a tető típusát (`RoofType`) és az árat (`Price`) tartalmazzák.

```{.python .input}
%%tab all
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('''NumRooms,RoofType,Price
NA,NA,127500
2,NA,106000
4,Slate,178100
NA,NA,140000''')
```

Most importáljuk a `pandas`-t, és töltsük be az adathalmazt a `read_csv` segítségével.

```{.python .input}
%%tab all
import pandas as pd

data = pd.read_csv(data_file)
print(data)
```

## Adatelőkészítés

A felügyelt tanulásban modelleket tanítunk,
amelyek adott *bemeneti* értékek alapján
egy kijelölt *célértéket* jósolnak meg.
Az adathalmaz feldolgozásának első lépése
a bemeneti és célértékeknek megfelelő oszlopok szétválasztása.
Az oszlopokat névvel vagy egész-alapú indexeléssel (`iloc`) is kijelölhetjük.

Észreveheted, hogy a `pandas` az összes `NA` értékű CSV bejegyzést
egy speciális `NaN` (*not a number*, nem szám) értékre cserélte.
Ez akkor is előfordulhat, ha egy bejegyzés üres,
pl. „3,,,270000".
Ezeket *hiányzó értékeknek* nevezzük;
az adattudomány „poloskái",
amelyekkel pályád során folyamatosan szembesülni fogsz.
A kontextustól függően a hiányzó értékeket
*pótlással* (*imputation*) vagy *törléssel* (*deletion*) lehet kezelni.
A pótlás a hiányzó értékeket becsült értékekkel helyettesíti,
míg a törlés egyszerűen elveti a hiányzó értékeket tartalmazó sorokat vagy oszlopokat.

Az alábbiakban néhány általános pótlási heurisztika látható.
[**Kategorikus bemeneti mezők esetén a `NaN`-t kategóriaként kezelhetjük.**]
Mivel a `RoofType` oszlop `Slate` és `NaN` értékeket vesz fel,
a `pandas` ezt az oszlopot két oszlopra tudja bontani: `RoofType_Slate` és `RoofType_nan`.
Egy `Slate` tetőtípusú sor esetén a `RoofType_Slate` és `RoofType_nan` értéke 1, illetve 0 lesz.
Hiányzó `RoofType` érték esetén ennek fordítottja igaz.

```{.python .input}
%%tab all
inputs, targets = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
```

Numerikus hiányzó értékek esetén az egyik általános heurisztika,
hogy [**a `NaN` bejegyzéseket a megfelelő oszlop átlagértékével helyettesítjük**].

```{.python .input}
%%tab all
inputs = inputs.fillna(inputs.mean())
print(inputs)
```

## Konverzió tenzorformátumba

Most, hogy [**az `inputs` és `targets` összes bejegyzése numerikus,
be tudjuk tölteni őket egy tenzorba**] (lásd: :numref:`sec_ndarray`).

```{.python .input}
%%tab mxnet
from mxnet import np

X, y = np.array(inputs.to_numpy(dtype=float)), np.array(targets.to_numpy(dtype=float))
X, y
```

```{.python .input}
%%tab pytorch
import torch

X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(targets.to_numpy(dtype=float))
X, y
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf

X = tf.constant(inputs.to_numpy(dtype=float))
y = tf.constant(targets.to_numpy(dtype=float))
X, y
```

```{.python .input}
%%tab jax
from jax import numpy as jnp

X = jnp.array(inputs.to_numpy(dtype=float))
y = jnp.array(targets.to_numpy(dtype=float))
X, y
```

## Összefoglalás

Most már tudod, hogyan kell adatoszlopokat szétválasztani,
hiányzó változókat pótolni,
és `pandas` adatokat tenzorba tölteni.
A :numref:`sec_kaggle_house` fejezetben
további adatfeldolgozási technikákat sajátíthatsz el.
Bár ez a gyorstalpaló egyszerű maradt,
az adatfeldolgozás bonyolulttá válhat.
Például az adathalmazunk nem egyetlen CSV fájlban érkezhet,
hanem több, relációs adatbázisból kinyert fájlban szétszórva.
Egy e-kereskedelmi alkalmazásban például
a vásárlói címek egy táblában,
a vásárlási adatok egy másikban lehetnek.
Ezenkívül a szakemberek számtalan, kategorikus és numerikus típuson túlmutató adattípussal találkoznak,
mint például szöveges karakterláncok, képek,
hangadatok és pontfelhők.
Sokszor fejlett eszközökre és hatékony algoritmusokra van szükség ahhoz,
hogy az adatfeldolgozás ne váljon a gépi tanulási folyamat
legnagyobb szűk keresztmetszetévé.
Ezek a problémák a gépi látás és a természetes nyelvfeldolgozás témaköreiben merülnek fel.
Végül figyelmet kell fordítani az adatminőségre is.
A valós adathalmazokat sok esetben kiugró értékek, hibás szenzormérések
és rögzítési hibák terhelik,
amelyeket az adatok modellbe való betáplálása előtt kezelni kell.
Az olyan adatvizualizációs eszközök, mint a [seaborn](https://seaborn.pydata.org/),
a [Bokeh](https://docs.bokeh.org/) vagy a [matplotlib](https://matplotlib.org/),
segítenek manuálisan megvizsgálni az adatokat
és megérteni, milyen problémákkal kell szembenézni.


## Feladatok

1. Próbálj meg adathalmazokat betölteni, pl. az Abalone adathalmazt az [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets)-ból, és vizsgáld meg a tulajdonságaikat. Mekkora hányadukban vannak hiányzó értékek? A változók mekkora hányada numerikus, kategorikus, illetve szöveges?
1. Próbálj adatoszlopokat neve alapján, ne oszlopszám szerint indexelni és kijelölni. A pandas [indexelésről szóló dokumentációja](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html) részletes útmutatást nyújt.
1. Mekkora adathalmazt gondolsz, hogy így be lehet tölteni? Mik lehetnek a korlátok? Tipp: vedd figyelembe az adatolvasás idejét, a reprezentációt, a feldolgozást és a memóriaigényt. Próbáld ki a laptopodon! Mi történik, ha szerveren próbálod?
1. Hogyan kezelnél olyan adatokat, amelyekben nagyon sok kategória van? Mi a helyzet, ha az összes kategóriacímke egyedi? Szerepeljenek-e ezek az adatban?
1. Milyen alternatívái vannak a pandasnak? Mi a helyzet a [NumPy tenzorok fájlból való betöltésével](https://numpy.org/doc/stable/reference/generated/numpy.load.html)? Nézd meg a [Pillow](https://python-pillow.org/), a Python képfeldolgozó könyvtárát.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/28)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/29)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/195)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17967)
:end_tab:
