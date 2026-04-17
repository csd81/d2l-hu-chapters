# Naiv Bayes
:label:`sec_naive_bayes`

Az előző szakaszokban megismerkedtünk a valószínűségelmélet és a véletlen változók elméletével. Hogy ezt a tudást a gyakorlatban is alkalmazzuk, vezessük be a *naiv Bayes*-osztályozót. Ez kizárólag valószínűségszámítási alapokon teszi lehetővé számjegyek osztályozását.

A tanulás lényege a feltételezések megtétele. Ha egy soha nem látott új adatpéldát szeretnénk osztályozni, feltételezéseket kell tennünk arról, hogy mely adatpéldák hasonlítanak egymáshoz. A naiv Bayes-osztályozó — egy népszerű és figyelemreméltóan egyszerű algoritmus — azt feltételezi, hogy az összes jellemző egymástól független, ezzel egyszerűsítve a számítást. Ebben a részben ezt a modellt alkalmazzuk képeken lévő karakterek felismerésére.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import gluon, np, npx
npx.set_np()
d2l.use_svg_display()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
import torchvision
d2l.use_svg_display()
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
d2l.use_svg_display()
```

## Optikai Karakterfelismerés

Az MNIST :cite:`LeCun.Bottou.Bengio.ea.1998` az egyik legszélesebb körben használt adathalmaz. $60\,000$ tanítási képet és $10\,000$ tesztképet tartalmaz. Minden kép egy kézzel írt számjegyet ábrázol $0$-tól $9$-ig. A feladat minden képet a megfelelő számjegybe sorolni.

A Gluon a `data.vision` modulban egy `MNIST` osztályt biztosít, amely automatikusan letölti az adathalmazt az internetről. Ezt követően a Gluon a már letöltött helyi másolatot használja. A `train` paraméter `True` vagy `False` értékre állításával adhatjuk meg, hogy a tanítási vagy a teszt adathalmazt kérjük. Minden kép $28 \times 28 \times 1$ alakú szürkeárnyalatos kép. Egyedi átalakítással eltávolítjuk az utolsó csatorna-dimenziót. Az adathalmaz minden pixelt előjel nélküli $8$-bites egész számként tárol; ezeket bináris jellemzőkké kvantáljuk a feladat egyszerűsítése érdekében.

```{.python .input}
#@tab mxnet
def transform(data, label):
    return np.floor(data.astype('float32') / 128).squeeze(axis=-1), label

mnist_train = gluon.data.vision.MNIST(train=True, transform=transform)
mnist_test = gluon.data.vision.MNIST(train=False, transform=transform)
```

```{.python .input}
#@tab pytorch
data_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    lambda x: torch.floor(x * 255 / 128).squeeze(dim=0)
])

mnist_train = torchvision.datasets.MNIST(
    root='./temp', train=True, transform=data_transform, download=True)
mnist_test = torchvision.datasets.MNIST(
    root='./temp', train=False, transform=data_transform, download=True)
```

```{.python .input}
#@tab tensorflow
((train_images, train_labels), (
    test_images, test_labels)) = tf.keras.datasets.mnist.load_data()

# Az MNIST eredeti pixelei 0 és 255 közé esnek (mivel a számjegyek uint8
# formában vannak tárolva). Ebben a részben az eredeti képen 128-nál nagyobb
# pixeleket 1-re, a 128-nál kisebbeket 0-ra alakítjuk. Hogy miért, azt lásd a
# 18.9.2 és 18.9.3 szakaszban.
train_images = tf.floor(tf.constant(train_images / 128, dtype = tf.float32))
test_images = tf.floor(tf.constant(test_images / 128, dtype = tf.float32))

train_labels = tf.constant(train_labels, dtype = tf.int32)
test_labels = tf.constant(test_labels, dtype = tf.int32)
```

Hozzáférhetünk egy adott példához, amely a képet és a hozzá tartozó címkét tartalmazza.

```{.python .input}
#@tab mxnet
image, label = mnist_train[2]
image.shape, label
```

```{.python .input}
#@tab pytorch
image, label = mnist_train[2]
image.shape, label
```

```{.python .input}
#@tab tensorflow
image, label = train_images[2], train_labels[2]
image.shape, label.numpy()
```

Az `image` változóban tárolt képpéldánk magassága és szélessége egyaránt $28$ pixel.

```{.python .input}
#@tab all
image.shape, image.dtype
```

A kódunk minden kép címkéjét skaláris értékként tárolja. Típusa $32$-bites egész szám.

```{.python .input}
#@tab mxnet
label, type(label), label.dtype
```

```{.python .input}
#@tab pytorch
label, type(label)
```

```{.python .input}
#@tab tensorflow
label.numpy(), label.dtype
```

Egyszerre több példát is elérhetünk.

```{.python .input}
#@tab mxnet
images, labels = mnist_train[10:38]
images.shape, labels.shape
```

```{.python .input}
#@tab pytorch
images = torch.stack([mnist_train[i][0] for i in range(10, 38)], dim=0)
labels = torch.tensor([mnist_train[i][1] for i in range(10, 38)])
images.shape, labels.shape
```

```{.python .input}
#@tab tensorflow
images = tf.stack([train_images[i] for i in range(10, 38)], axis=0)
labels = tf.constant([train_labels[i].numpy() for i in range(10, 38)])
images.shape, labels.shape
```

Jelenítsük meg ezeket a példákat.

```{.python .input}
#@tab all
d2l.show_images(images, 2, 9);
```

## Az Osztályozás Valószínűségi Modellje

Az osztályozási feladatban egy példát egy kategóriába sorolunk. Itt a példa egy szürkeárnyalatos $28\times 28$-as kép, a kategória pedig egy számjegy. (Részletesebb magyarázatért lásd: :numref:`sec_softmax`.)
Az osztályozási feladat természetes megfogalmazása egy valószínűségi kérdés: mi a legvalószínűbb címke a jellemzők (azaz a képpontok) alapján? Jelöljük a példa jellemzőit $\mathbf x\in\mathbb R^d$-vel, a címkét pedig $y\in\mathbb R$-rel. A jellemzők a képpontok, ahol egy $2$-dimenziós képet vektorrá alakíthatunk, így $d=28^2=784$, a címkék pedig számjegyek.
A jellemzők alapján számított címkevalószínűség $p(y  \mid  \mathbf{x})$. Ha ki tudjuk számítani ezeket a valószínűségeket — amelyek példánkban $p(y  \mid  \mathbf{x})$, ahol $y=0, \ldots,9$ — az osztályozó a következő predikciót adja:

$$\hat{y} = \mathrm{argmax} \> p(y  \mid  \mathbf{x}).$$

Sajnos ehhez meg kell becsülnünk $p(y  \mid  \mathbf{x})$ értékét $\mathbf{x} = x_1, ..., x_d$ minden lehetséges értékére. Képzeljük el, hogy minden jellemző $2$ értéket vehet fel. Például $x_1 = 1$ azt jelenti, hogy az „alma" szó szerepel egy adott dokumentumban, $x_1 = 0$ pedig azt, hogy nem. Ha $30$ ilyen bináris jellemzőnk lenne, fel kellene készülnünk a bemeneti $\mathbf{x}$ vektor $2^{30}$ (több mint egymilliárd!) lehetséges értékének osztályozására.

Ezenkívül hol is van itt a tanulás? Ha az összes lehetséges példát látni kell a megfelelő címke előrejelzéséhez, valójában nem tanulunk mintát, hanem csupán memorizáljuk az adathalmazt.

## A Naiv Bayes-Osztályozó

Szerencsére feltételes függetlenségi feltételezések megtételével induktív torzítást (inductive bias-t) vihetünk a modellbe, és képessé tehetjük azt arra, hogy viszonylag kevés tanítási példából általánosítson. Kezdjük azzal, hogy a Bayes-tétel segítségével az osztályozót a következőképpen írjuk fel:

$$\hat{y} = \mathrm{argmax}_y \> p(y  \mid  \mathbf{x}) = \mathrm{argmax}_y \> \frac{p( \mathbf{x}  \mid  y) p(y)}{p(\mathbf{x})}.$$

Vegyük észre, hogy a nevező a $p(\mathbf{x})$ normalizáló tag, amely nem függ a $y$ címke értékétől. Ezért csupán a számláló különböző $y$ értékeken való összehasonlítására kell figyelnünk. Még ha a nevező kiszámítása nehézkes is lenne, elhanyagolhatnánk, amíg a számlálót ki tudjuk értékelni. A normalizáló konstanst egyébként mindig visszaállíthatjuk, hiszen $\sum_y p(y  \mid  \mathbf{x}) = 1$.

Most összpontosítsunk a $p( \mathbf{x}  \mid  y)$ tagra. A valószínűség láncolási szabálya alapján $p( \mathbf{x}  \mid  y)$ a következőképpen írható fel:

$$p(x_1  \mid y) \cdot p(x_2  \mid  x_1, y) \cdot ... \cdot p( x_d  \mid  x_1, ..., x_{d-1}, y).$$

Önmagában ez a kifejezés nem visz minket előrébb — még mindig hozzávetőleg $2^d$ paramétert kellene megbecsülni. Ha azonban feltételezzük, hogy *a jellemzők feltételesen egymástól függetlenek, adott címke esetén*, akkor a helyzet lényegesen javul, mivel ez a tag $\prod_i p(x_i  \mid  y)$-ra egyszerűsödik, és az előrejelző a következő lesz:

$$\hat{y} = \mathrm{argmax}_y \> \prod_{i=1}^d p(x_i  \mid  y) p(y).$$

Ha minden $i$ és $y$ esetén meg tudjuk becsülni $p(x_i=1  \mid  y)$ értékét, és azt $P_{xy}[i, y]$-ban tároljuk — ahol $P_{xy}$ egy $d\times n$ mátrix, $n$ az osztályok száma, $y\in\{1, \ldots, n\}$ — akkor ebből $p(x_i = 0 \mid y)$ is becsülhető:

$$
p(x_i = t_i \mid y) =
\begin{cases}
    P_{xy}[i, y] & \textrm{for } t_i=1 ;\\
    1 - P_{xy}[i, y] & \textrm{for } t_i = 0 .
\end{cases}
$$

Emellett minden $y$-ra megbecsüljük $p(y)$ értékét, amelyet $P_y[y]$-ban tárolunk, ahol $P_y$ egy $n$ hosszú vektor. Ekkor bármely új $\mathbf t = (t_1, t_2, \ldots, t_d)$ példára kiszámíthatjuk:

$$\begin{aligned}\hat{y} &= \mathrm{argmax}_ y \ p(y)\prod_{i=1}^d   p(x_t = t_i \mid y) \\ &= \mathrm{argmax}_y \ P_y[y]\prod_{i=1}^d \ P_{xy}[i, y]^{t_i}\, \left(1 - P_{xy}[i, y]\right)^{1-t_i}\end{aligned}$$
:eqlabel:`eq_naive_bayes_estimation`

minden $y$-ra. A feltételes függetlenségi feltételezés tehát a modellünk komplexitását a jellemzők számától exponenciálisan függő $\mathcal{O}(2^dn)$-ről lineárisan függő $\mathcal{O}(dn)$-re csökkentette.


## Tanítás

A probléma most az, hogy nem ismerjük $P_{xy}$-t és $P_y$-t. Ezért értékeiket tanítási adatok alapján kell megbecsülni. Ez a modell *tanítása*. $P_y$ becslése nem túl bonyolult: mivel csak $10$ osztállyal dolgozunk, megszámolhatjuk az egyes számjegyek $n_y$ előfordulásait, és elosztjuk az adatok teljes $n$ számával. Például ha a 8-as számjegy $n_8 = 5\,800$-szor fordul elő és összesen $n = 60\,000$ képünk van, a valószínűség becslése $p(y=8) = 0{,}0967$.

```{.python .input}
#@tab mxnet
X, Y = mnist_train[:]  # All training examples

n_y = np.zeros((10))
for y in range(10):
    n_y[y] = (Y == y).sum()
P_y = n_y / n_y.sum()
P_y
```

```{.python .input}
#@tab pytorch
X = torch.stack([mnist_train[i][0] for i in range(len(mnist_train))], dim=0)
Y = torch.tensor([mnist_train[i][1] for i in range(len(mnist_train))])

n_y = torch.zeros(10)
for y in range(10):
    n_y[y] = (Y == y).sum()
P_y = n_y / n_y.sum()
P_y
```

```{.python .input}
#@tab tensorflow
X = train_images
Y = train_labels

n_y = tf.Variable(tf.zeros(10))
for y in range(10):
    n_y[y].assign(tf.reduce_sum(tf.cast(Y == y, tf.float32)))
P_y = n_y / tf.reduce_sum(n_y)
P_y
```

Most térjünk rá a valamivel nehezebb $P_{xy}$ becslésre. Mivel fekete-fehér képeket választottunk, $p(x_i  \mid  y)$ annak valószínűségét jelöli, hogy az $i$-edik pixel be van kapcsolva a $y$ osztályban. Ugyanúgy, mint korábban, megszámlálhatjuk, hányszor fordul elő az $n_{iy}$ esemény, és elosztjuk $y$ összes előfordulásával, azaz $n_y$-nal. Van azonban egy kis probléma: bizonyos pixelek soha nem lehetnek feketék (például jól kivágott képeknél a sarokpixelek mindig fehérek lehetnek). Erre a problémára a statisztikusok bevett megoldása az álszámlálók hozzáadása minden előforduláshoz. Ezért $n_{iy}$ helyett $n_{iy}+1$-et, $n_y$ helyett $n_{y}+2$-t használunk (mivel az $i$-edik pixel két lehetséges értéket vehet fel: fekete vagy fehér). Ezt *Laplace-simítás*nak is nevezik. Bár ad-hoc módszernek tűnhet, egy Beta-binomiális modell alapján bayesi szempontból is motiválható.

```{.python .input}
#@tab mxnet
n_x = np.zeros((10, 28, 28))
for y in range(10):
    n_x[y] = np.array(X.asnumpy()[Y.asnumpy() == y].sum(axis=0))
P_xy = (n_x + 1) / (n_y + 2).reshape(10, 1, 1)

d2l.show_images(P_xy, 2, 5);
```

```{.python .input}
#@tab pytorch
n_x = torch.zeros((10, 28, 28))
for y in range(10):
    n_x[y] = torch.tensor(X.numpy()[Y.numpy() == y].sum(axis=0))
P_xy = (n_x + 1) / (n_y + 2).reshape(10, 1, 1)

d2l.show_images(P_xy, 2, 5);
```

```{.python .input}
#@tab tensorflow
n_x = tf.Variable(tf.zeros((10, 28, 28)))
for y in range(10):
    n_x[y].assign(tf.cast(tf.reduce_sum(
        X.numpy()[Y.numpy() == y], axis=0), tf.float32))
P_xy = (n_x + 1) / tf.reshape((n_y + 2), (10, 1, 1))

d2l.show_images(P_xy, 2, 5);
```

Ezen $10\times 28\times 28$ valószínűség (minden osztályhoz minden pixel) megjelenítésével néhány átlagos kinézetű számjegyet kaphatunk.

Most a :eqref:`eq_naive_bayes_estimation` képlettel megjósolhatunk egy új képet. Adott $\mathbf x$ esetén a következő függvények minden $y$-ra kiszámítják $p(\mathbf x \mid y)p(y)$ értékét.

```{.python .input}
#@tab mxnet
def bayes_pred(x):
    x = np.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = P_xy * x + (1 - P_xy)*(1 - x)
    p_xy = p_xy.reshape(10, -1).prod(axis=1)  # p(x|y)
    return np.array(p_xy) * P_y

image, label = mnist_test[0]
bayes_pred(image)
```

```{.python .input}
#@tab pytorch
def bayes_pred(x):
    x = x.unsqueeze(0)  # (28, 28) -> (1, 28, 28)
    p_xy = P_xy * x + (1 - P_xy)*(1 - x)
    p_xy = p_xy.reshape(10, -1).prod(dim=1)  # p(x|y)
    return p_xy * P_y

image, label = mnist_test[0]
bayes_pred(image)
```

```{.python .input}
#@tab tensorflow
def bayes_pred(x):
    x = tf.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = P_xy * x + (1 - P_xy)*(1 - x)
    p_xy = tf.math.reduce_prod(tf.reshape(p_xy, (10, -1)), axis=1)  # p(x|y)
    return p_xy * P_y

image, label = train_images[0], train_labels[0]
bayes_pred(image)
```

Ez szörnyen rosszul sült el! Hogy megértsük, miért, nézzük meg a pixelenkénti valószínűségeket. Ezek jellemzően $0{,}001$ és $1$ közötti értékek. Ebből $784$ értéket szorzunk össze. Érdemes megemlíteni, hogy ezeket a számokat egy számítógépen számítjuk, amelynek véges a kitevő-tartománya. Az történik, hogy *numerikus alulcsordulással* találkozunk: az összes kis szám szorzata egyre kisebb lesz, majd nullára kerekítődik. Ezt elméleti problémaként tárgyaltuk a :numref:`sec_maximum_likelihood` részben, de itt a gyakorlatban is egyértelműen láthatjuk.

Ahogy abban a részben tárgyaltuk, a megoldás a $\log a b = \log a + \log b$ azonosság kihasználása, azaz a logaritmusok összegére való áttérés. Még ha mind $a$, mind $b$ kis szám is, a logaritmikus értékek egy megfelelő tartományban lesznek.

```{.python .input}
#@tab mxnet
a = 0.1
print('underflow:', a**784)
print('logarithm is normal:', 784*math.log(a))
```

```{.python .input}
#@tab pytorch
a = 0.1
print('underflow:', a**784)
print('logarithm is normal:', 784*math.log(a))
```

```{.python .input}
#@tab tensorflow
a = 0.1
print('underflow:', a**784)
print('logarithm is normal:', 784*tf.math.log(a).numpy())
```

Mivel a logaritmus monoton növekvő függvény, a :eqref:`eq_naive_bayes_estimation` képletet a következőképpen írhatjuk át:

$$ \hat{y} = \mathrm{argmax}_y \ \log P_y[y] + \sum_{i=1}^d \Big[t_i\log P_{xy}[x_i, y] + (1-t_i) \log (1 - P_{xy}[x_i, y]) \Big].$$

Megvalósítható a következő numerikusan stabil változat:

```{.python .input}
#@tab mxnet
log_P_xy = np.log(P_xy)
log_P_xy_neg = np.log(1 - P_xy)
log_P_y = np.log(P_y)

def bayes_pred_stable(x):
    x = np.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = log_P_xy * x + log_P_xy_neg * (1 - x)
    p_xy = p_xy.reshape(10, -1).sum(axis=1)  # p(x|y)
    return p_xy + log_P_y

py = bayes_pred_stable(image)
py
```

```{.python .input}
#@tab pytorch
log_P_xy = torch.log(P_xy)
log_P_xy_neg = torch.log(1 - P_xy)
log_P_y = torch.log(P_y)

def bayes_pred_stable(x):
    x = x.unsqueeze(0)  # (28, 28) -> (1, 28, 28)
    p_xy = log_P_xy * x + log_P_xy_neg * (1 - x)
    p_xy = p_xy.reshape(10, -1).sum(axis=1)  # p(x|y)
    return p_xy + log_P_y

py = bayes_pred_stable(image)
py
```

```{.python .input}
#@tab tensorflow
log_P_xy = tf.math.log(P_xy)
log_P_xy_neg = tf.math.log(1 - P_xy)
log_P_y = tf.math.log(P_y)

def bayes_pred_stable(x):
    x = tf.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = log_P_xy * x + log_P_xy_neg * (1 - x)
    p_xy = tf.math.reduce_sum(tf.reshape(p_xy, (10, -1)), axis=1)  # p(x|y)
    return p_xy + log_P_y

py = bayes_pred_stable(image)
py
```

Most már ellenőrizhetjük, hogy helyes-e a jóslat.

```{.python .input}
#@tab mxnet
# Alakítsuk át a címkét, amely int32 típusú skalár tenzor, Python skalárrá az összehasonlításhoz
py.argmax(axis=0) == int(label)
```

```{.python .input}
#@tab pytorch
py.argmax(dim=0) == label
```

```{.python .input}
#@tab tensorflow
tf.argmax(py, axis=0, output_type = tf.int32) == label
```

Ha most néhány validációs példát jósolunk meg, láthatjuk, hogy a Bayes-osztályozó elég jól teljesít.

```{.python .input}
#@tab mxnet
def predict(X):
    return [bayes_pred_stable(x).argmax(axis=0).astype(np.int32) for x in X]

X, y = mnist_test[:18]
preds = predict(X)
d2l.show_images(X, 2, 9, titles=[str(d) for d in preds]);
```

```{.python .input}
#@tab pytorch
def predict(X):
    return [bayes_pred_stable(x).argmax(dim=0).type(torch.int32).item()
            for x in X]

X = torch.stack([mnist_test[i][0] for i in range(18)], dim=0)
y = torch.tensor([mnist_test[i][1] for i in range(18)])
preds = predict(X)
d2l.show_images(X, 2, 9, titles=[str(d) for d in preds]);
```

```{.python .input}
#@tab tensorflow
def predict(X):
    return [tf.argmax(
        bayes_pred_stable(x), axis=0, output_type = tf.int32).numpy()
            for x in X]

X = tf.stack([train_images[i] for i in range(10, 38)], axis=0)
y = tf.constant([train_labels[i].numpy() for i in range(10, 38)])
preds = predict(X)
d2l.show_images(X, 2, 9, titles=[str(d) for d in preds]);
```

Végül számítsuk ki az osztályozó összesített pontosságát.

```{.python .input}
#@tab mxnet
X, y = mnist_test[:]
preds = np.array(predict(X), dtype=np.int32)
float((preds == y).sum()) / len(y)  # Validációs pontosság
```

```{.python .input}
#@tab pytorch
X = torch.stack([mnist_test[i][0] for i in range(len(mnist_test))], dim=0)
y = torch.tensor([mnist_test[i][1] for i in range(len(mnist_test))])
preds = torch.tensor(predict(X), dtype=torch.int32)
float((preds == y).sum()) / len(y)  # Validációs pontosság
```

```{.python .input}
#@tab tensorflow
X = test_images
y = test_labels
preds = tf.constant(predict(X), dtype=tf.int32)
# Validációs pontosság
tf.reduce_sum(tf.cast(preds == y, tf.float32)).numpy() / len(y)
```

A modern mély hálózatok $0{,}01$-nél kisebb hibaarányt érnek el. A viszonylag gyenge teljesítmény a modellünkben tett helytelen statisztikai feltételezéseknek köszönhető: azt feltételeztük, hogy minden egyes pixel *egymástól függetlenül* generálódik, kizárólag a címkétől függően. Nyilvánvalóan nem így ír az ember számjegyeket, és ez a téves feltételezés okozta a túlságosan naiv (Bayes) osztályozónk bukását.

## Összefoglalás
* A Bayes-tétel segítségével egy osztályozó építhető azzal a feltételezéssel, hogy az összes megfigyelt jellemző független egymástól.
* Ezt az osztályozót egy adathalmazon a címkék és a pixelértékek kombinációinak előfordulásait megszámlálva lehet betanítani.
* Ez az osztályozó volt évtizedekig az iparági standard olyan feladatoknál, mint a spam-szűrés.

## Feladatok
1. Vegyük a $[[0,0], [0,1], [1,0], [1,1]]$ adathalmazt, ahol a két elem XOR-ja adja a $[0,1,1,0]$ címkéket. Milyen valószínűségeket rendel ehhez az adathalmazhoz a naiv Bayes-osztályozó? Sikeresen osztályozza-e a pontokat? Ha nem, milyen feltételezések sérülnek?
1. Tegyük fel, hogy Laplace-simítás nélkül becsültük a valószínűségeket, és tesztelés során érkezik egy adatpélda, amelynek értéke soha nem szerepelt a tanítás során. Mit adna ki a modell?
1. A naiv Bayes-osztályozó egy speciális példája a Bayes-hálónak, amelyben a véletlen változók közötti függőségeket gráfstruktúra kódolja. Bár a teljes elmélet meghaladja e fejezet kereteit (részletes leírásért lásd :citet:`Koller.Friedman.2009`), magyarázza meg, hogy a XOR-modellben miért teszi lehetővé a két bemeneti változó közötti explicit függőség megengedése egy sikeres osztályozó létrehozását.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/418)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1100)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1101)
:end_tab:
