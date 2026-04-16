# A MovieLens adathalmaz

Számos adathalmaz áll rendelkezésre ajánlási kutatásokhoz. Ezek közül a [MovieLens](https://movielens.org/) adathalmaz valószínűleg az egyik legnépszerűbb. A MovieLens egy nem kereskedelmi, webalapú filmajánló rendszer. 1997-ben hozták létre, és a Minnesotai Egyetem kutatólaborja, a GroupLens üzemelteti azzal a céllal, hogy kutatási felhasználásra filmértékelési adatokat gyűjtsön. A MovieLens-adatok számos kutatási vizsgálatban kulcsszerepet játszottak, többek között a személyre szabott ajánlás és a szociálpszichológia területén.


## Az adatok letöltése


A MovieLens adathalmaz a [GroupLens](https://grouplens.org/datasets/movielens/) webhelyen érhető el. Több változat közül választhatunk. Mi a MovieLens 100K adathalmazt használjuk :cite:`Herlocker.Konstan.Borchers.ea.1999`. Ez az adathalmaz 100000 értékelést tartalmaz, 1 és 5 csillag közötti skálán, 943 felhasználótól 1682 filmre. Az adatokat megtisztították úgy, hogy minden felhasználó legalább 20 filmet értékelt. Egyszerű demográfiai információk, például a felhasználók és az elemek életkora, neme és műfajai is elérhetők. Letölthetjük az [ml-100k.zip](http://files.grouplens.org/datasets/movielens/ml-100k.zip) fájlt, majd kicsomagolhatjuk belőle a `u.data` fájlt, amely a 100000 értékelést csv formátumban tartalmazza. A mappában sok más fájl is található, és az egyes fájlok részletes leírása az adathalmaz [README](http://files.grouplens.org/datasets/movielens/ml-100k-README.txt) fájljában olvasható.

Kezdésként importáljuk a szakasz kísérleteihez szükséges csomagokat.

```{.python .input  n=1}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, np
import os
import pandas as pd
```

Ezután letöltjük a MovieLens 100k adathalmazt, és az interakciókat `DataFrame`-be töltjük.

```{.python .input  n=2}
#@tab mxnet
#@save
d2l.DATA_HUB['ml-100k'] = (
    'https://files.grouplens.org/datasets/movielens/ml-100k.zip',
    'cd4dcac4241c8a4ad7badc7ca635da8a69dddb83')

#@save
def read_data_ml100k():
    data_dir = d2l.download_extract('ml-100k')
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv(os.path.join(data_dir, 'u.data'), sep='\t',
                       names=names, engine='python')
    num_users = data.user_id.unique().shape[0]
    num_items = data.item_id.unique().shape[0]
    return data, num_users, num_items
```

## Az adathalmaz statisztikái

Töltsük be az adatokat, és nézzük meg kézzel az első öt rekordot. Ez hatékony módja az adatszerkezet megismerésének és annak ellenőrzésére, hogy az adatok helyesen töltődtek-e be.

```{.python .input  n=3}
#@tab mxnet
data, num_users, num_items = read_data_ml100k()
sparsity = 1 - len(data) / (num_users * num_items)
print(f'number of users: {num_users}, number of items: {num_items}')
print(f'matrix sparsity: {sparsity:f}')
print(data.head(5))
```

Láthatjuk, hogy minden sor négy oszlopból áll: "user id" 1-943, "item id" 1-1682, "rating" 1-5 és "timestamp". Készíthetünk egy $n \times m$ méretű interakciós mátrixot, ahol $n$ és $m$ a felhasználók, illetve az elemek száma. Ez az adathalmaz csak a meglévő értékeléseket rögzíti, ezért értékbecslési mátrixnak is nevezhetjük, és az interakciós mátrix és az értékbecslési mátrix kifejezéseket felváltva használjuk, ha a mátrix elemei pontos értékeléseket jelentenek. Az értékbecslési mátrix értékeinek nagy része ismeretlen, mivel a felhasználók a filmek többségét nem értékelték. Az adathalmaz ritkaságát is megmutatjuk. A ritkaság definíciója: `1 - nemnulla elemek száma / ( felhasználók száma * elemek száma)`. Egyértelmű, hogy az interakciós mátrix rendkívül ritka (vagyis a ritkaság = 93.695%). A valós adathalmazok még ennél is ritkábbak lehetnek, és ez régóta fennálló kihívás az ajánlórendszerek építésében. Egy lehetséges megoldás további mellékinformációk, például felhasználó- és elemjellemzők használata a ritkaság enyhítésére.

Ezután ábrázoljuk a különböző értékelések gyakoriságának eloszlását. Ahogy várható, ez nagyjából normális eloszlást követ, és az értékelések többsége 3 és 4 körül koncentrálódik.

```{.python .input  n=4}
#@tab mxnet
d2l.plt.hist(data['rating'], bins=5, ec='black')
d2l.plt.xlabel('Rating')
d2l.plt.ylabel('Count')
d2l.plt.title('Distribution of Ratings in MovieLens 100K')
d2l.plt.show()
```

## Az adathalmaz felosztása

Az adathalmazt tanító- és teszthalmazra osztjuk. A következő függvény két felosztási módot kínál: `random` és `seq-aware`. A `random` módban a függvény időbélyeg figyelembevétele nélkül véletlenszerűen osztja fel a 100k interakciót, és alapértelmezés szerint az adatok 90%-át tanító-, 10%-át teszthalmaznak használja. A `seq-aware` módban minden felhasználónál a legutóbb értékelt elemet hagyjuk ki tesztelésre, a korábbi interakciókat pedig tanítóhalmazként használjuk. A felhasználói történeti interakciókat időbélyeg alapján a legrégebbitől a legújabbig rendezzük. Ezt a módot fogjuk használni a sorrendtudatos ajánlási részben.

```{.python .input  n=5}
#@tab mxnet
#@save
def split_data_ml100k(data, num_users, num_items,
                      split_mode='random', test_ratio=0.1):
    """Az adathalmazt véletlenszerű vagy sorrendtudatos módban osztja fel."""
    if split_mode == 'seq-aware':
        train_items, test_items, train_list = {}, {}, []
        for line in data.itertuples():
            u, i, rating, time = line[1], line[2], line[3], line[4]
            train_items.setdefault(u, []).append((u, i, rating, time))
            if u not in test_items or test_items[u][-1] < time:
                test_items[u] = (i, rating, time)
        for u in range(1, num_users + 1):
            train_list.extend(sorted(train_items[u], key=lambda k: k[3]))
        test_data = [(key, *value) for key, value in test_items.items()]
        train_data = [item for item in train_list if item not in test_data]
        train_data = pd.DataFrame(train_data)
        test_data = pd.DataFrame(test_data)
    else:
        mask = [True if x == 1 else False for x in np.random.uniform(
            0, 1, (len(data))) < 1 - test_ratio]
        neg_mask = [not x for x in mask]
        train_data, test_data = data[mask], data[neg_mask]
    return train_data, test_data
```

Megjegyzendő, hogy a gyakorlatban a teszthalmazon túl érdemes egy validációs halmazt is használni. Ezt azonban a rövidség kedvéért itt kihagyjuk. Ebben az esetben a teszthalmazunk tekinthető megtartott validációs halmaznak.

## Az adatok betöltése

Az adathalmaz felosztása után a tanító- és teszthalmazt kényelmi okokból listákká, illetve szótárakká/mátrixokká alakítjuk. A következő függvény soronként olvassa be a `DataFrame`-et, és a felhasználók/elemek indexelését nullától kezdi. Ezután visszaadja a felhasználók, elemek és értékelések listáit, valamint egy szótárat/mátrixot, amely az interakciókat tárolja. A visszajelzés típusát megadhatjuk `explicit` vagy `implicit` értékként.

```{.python .input  n=6}
#@tab mxnet
#@save
def load_data_ml100k(data, num_users, num_items, feedback='explicit'):
    users, items, scores = [], [], []
    inter = np.zeros((num_items, num_users)) if feedback == 'explicit' else {}
    for line in data.itertuples():
        user_index, item_index = int(line[1] - 1), int(line[2] - 1)
        score = int(line[3]) if feedback == 'explicit' else 1
        users.append(user_index)
        items.append(item_index)
        scores.append(score)
        if feedback == 'implicit':
            inter.setdefault(user_index, []).append(item_index)
        else:
            inter[item_index, user_index] = score
    return users, items, scores, inter
```

Ezután az előző lépéseket összevonjuk, és a következő szakaszban ezt fogjuk használni. Az eredményeket `Dataset` és `DataLoader` objektumokba csomagoljuk. Megjegyzendő, hogy a tanítóadatokhoz használt `DataLoader` `last_batch` beállítása `rollover` módra van állítva (a maradék minták átcsúsznak a következő epochba), és az elemek sorrendje keverve van.

```{.python .input  n=7}
#@tab mxnet
#@save
def split_and_load_ml100k(split_mode='seq-aware', feedback='explicit',
                          test_ratio=0.1, batch_size=256):
    data, num_users, num_items = read_data_ml100k()
    train_data, test_data = split_data_ml100k(
        data, num_users, num_items, split_mode, test_ratio)
    train_u, train_i, train_r, _ = load_data_ml100k(
        train_data, num_users, num_items, feedback)
    test_u, test_i, test_r, _ = load_data_ml100k(
        test_data, num_users, num_items, feedback)
    train_set = gluon.data.ArrayDataset(
        np.array(train_u), np.array(train_i), np.array(train_r))
    test_set = gluon.data.ArrayDataset(
        np.array(test_u), np.array(test_i), np.array(test_r))
    train_iter = gluon.data.DataLoader(
        train_set, shuffle=True, last_batch='rollover',
        batch_size=batch_size)
    test_iter = gluon.data.DataLoader(
        test_set, batch_size=batch_size)
    return num_users, num_items, train_iter, test_iter
```

## Összefoglalás

* A MovieLens adathalmazokat széles körben használják ajánlási kutatásokban. Nyilvánosan elérhetők és szabadon felhasználhatók.
* Olyan függvényeket definiálunk, amelyek letöltik és előfeldolgozzák a MovieLens 100k adathalmazt, hogy későbbi szakaszokban is használhassuk.


## Gyakorlatok

* Milyen más, hasonló ajánlási adathalmazokat tudsz találni?
* Nézz körül a [https://movielens.org/](https://movielens.org/) oldalon további MovieLens-információkért.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/399)
:end_tab:
