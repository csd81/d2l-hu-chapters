```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Többrétegű perceptronok
:label:`sec_mlp`

A :numref:`sec_softmax` fejezetben bemutattuk
a softmax regressziót,
az algoritmust nulláról implementálva
(:numref:`sec_softmax_scratch`) és magas szintű API-k segítségével
(:numref:`sec_softmax_concise`). Ez lehetővé tette, hogy
olyan osztályozókat tanítsunk, amelyek képesek felismerni
10 kategóriájú ruházatot alacsony felbontású képekből.
Közben megtanultuk, hogyan kell adatokat kezelni,
kimeneteinket érvényes valószínűségi eloszlásba alakítani,
megfelelő veszteségfüggvényt alkalmazni,
és azt minimalizálni a modell paramétereivel szemben.
Most, hogy ezeket a mechanizmusokat
egyszerű lineáris modellek kontextusában elsajátítottuk,
elindulhatunk a mély neurális hálózatok felfedezésére,
amelyek összehasonlíthatatlanul gazdagabb modellosztályt alkotnak,
és amelyekkel ez a könyv elsősorban foglalkozik.

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

## Rejtett rétegek

Az affin transzformációkat a :numref:`subsec_linear_model` részben
lineáris transzformációkként definiáltuk hozzáadott torzítással.
Kezdetként idézzük fel a softmax regressziós példánknak megfelelő
modellarchitektúrát, amelyet a :numref:`fig_softmaxreg` ábra illusztrál.
Ez a modell egyetlen affin transzformáción keresztül képezi le
a bemeneteket a kimenetekre, majd egy softmax műveletet alkalmaz.
Ha a címkék valóban egyszerű affin transzformáció révén kapcsolódnának
a bemeneti adatokhoz, akkor ez a megközelítés elegendő lenne.
Azonban a linearitás (affin transzformációkban) egy *erős* feltételezés.

### A lineáris modellek korlátai

Például a linearitás egy *gyengébb*
feltételezést von maga után, a *monotonitást*, azaz,
hogy bármely jellemzőnk növekedése
mindig a modell kimenetének növekedéséhez kell, hogy vezessen
(ha a megfelelő súly pozitív),
vagy mindig csökkenéshez
(ha a megfelelő súly negatív).
Néha ez értelmes.
Például, ha azt próbálnánk megjósolni,
hogy egy személy visszafizet-e egy kölcsönt,
ésszerűen feltételezhetjük, hogy minden más feltétel azonossága esetén
a magasabb jövedelmű kérelmező
mindig valószínűbben fog visszafizetni,
mint az alacsonyabb jövedelmű.
Bár monoton, ez a kapcsolat valószínűleg
nem lineárisan függ össze a visszafizetés valószínűségével.
A jövedelem 0-ról 50 000 dollárra való növekedése
valószínűleg nagyobb növekedést jelent
a visszafizetés valószínűségében,
mint az 1 millióról 1,05 millió dollárra való növekedés.
Ennek kezelésére kimeneteinkre utófeldolgozást alkalmazhatunk,
hogy a linearitás valószínűbbé váljon,
például a logisztikus leképezés (és így a kimenet valószínűségének logaritmusa) segítségével.

Vegyük észre, hogy könnyen találhatunk a monotonitást sértő példákat.
Tegyük fel például, hogy az egészséget a testhőmérséklet függvényeként
szeretnénk megjósolni.
A normál testhőmérséklet 37°C (98,6°F) feletti értékű egyéneknél
a magasabb hőmérséklet nagyobb kockázatot jelez.
Azonban ha a testhőmérséklet 37°C alá csökken,
az alacsonyabb hőmérséklet jelent nagyobb kockázatot!
Ennél a problémánál is megoldható lehetne
némi okos előfeldolgozással, például a 37°C-tól való távolság
jellemzőként való felhasználásával.

De mi a helyzet macskák és kutyák képeinek osztályozásával?
A (13, 17) koordinátájú pixel intenzitásának növelése
mindig növelné (vagy mindig csökkentené)
annak valószínűségét, hogy a kép kutyát ábrázol?
A lineáris modellre való támaszkodás megfelel annak az implicit
feltételezésnek, hogy macskák és kutyák megkülönböztetéséhez
az egyéni pixelek fényességének értékelése elegendő.
Ez a megközelítés kudarcra van ítélve egy olyan világban,
ahol a kép invertálása megőrzi a kategóriát.

A linearitás nyilvánvaló abszurditása ellenére azonban,
összehasonlítva a korábbi példáinkkal,
nem annyira nyilvánvaló, hogy a problémát
egyszerű előfeldolgozással meg lehetne oldani.
Ez azért van, mert bármely pixel jelentősége
összetett módon függ a kontextusától
(a szomszédos pixelek értékeitől).
Bár lehet, hogy létezne egy adatreprezentáció,
amely figyelembe venné a jellemzőink közötti
releváns kölcsönhatásokat,
amelyen felül egy lineáris modell megfelelő lenne,
egyszerűen nem tudjuk, hogyan számítsuk ki ezt kézzel.
Mély neurális hálózatokkal megfigyelt adatokat használtunk
a rejtett rétegeken keresztüli reprezentáció
és a lineáris prediktor közös tanulására.

A nemlinearitás problémáját legalább egy évszázada tanulmányozzák :cite:`Fisher.1928`.
Például a döntési fák alapformájukban bináris döntések sorozatát alkalmazzák
az osztálytagság meghatározásához :cite:`quinlan2014c4`. Hasonlóképpen,
a kernel-módszereket évtizedek óta alkalmazzák nemlineáris függőségek modellezésére
:cite:`Aronszajn.1950`. Ez megjelent
a nemparametrikus spline-modellekben :cite:`Wahba.1990` és a kernel-módszerekben
:cite:`Scholkopf.Smola.2002`. Ez az agy számára is egészen természetesen megoldott probléma.
Végül is a neuronok más neuronokba táplálnak, amelyek viszont
ismét más neuronokba táplálnak :cite:`Cajal.Azoulay.1894`.
Ennek következtében viszonylag egyszerű transzformációk sorozatát kapjuk.

### Rejtett rétegek bevezetése

A lineáris modellek korlátait leküzdhetjük
egy vagy több rejtett réteg bevezetésével.
Ennek legegyszerűbb módja, hogy
több teljesen összekötött réteget helyezünk egymás fölé.
Minden réteg a felette lévő rétegbe táplál,
amíg el nem érjük a kimeneteket.
Az első $L-1$ réteget tekinthetjük
reprezentációnak, az utolsó réteget
lineáris prediktornak.
Ezt az architektúrát általában
*többrétegű perceptronnak* nevezik,
amelyet gyakran *MLP*-nek rövidítenek (:numref:`fig_mlp`).

![Egy MLP öt rejtett egységből álló rejtett réteggel.](../img/mlp.svg)
:label:`fig_mlp`

Ez az MLP négy bemenettel, három kimenettel rendelkezik,
és rejtett rétege öt rejtett egységet tartalmaz.
Mivel a bemeneti réteg nem végez semmilyen számítást,
a hálózat kimenetének előállításához
mind a rejtett, mind a kimeneti réteg számításait meg kell valósítani;
így az MLP rétegszáma kettő.
Vegyük észre, hogy mindkét réteg teljesen összekötött.
Minden bemenet hat a rejtett réteg minden neuronjára,
és ezek mindegyike viszont hat
a kimeneti réteg minden neuronjára. De még nem vagyunk teljesen készen.

### A lineáristól a nemlineárisig

Korábbihoz hasonlóan az $\mathbf{X} \in \mathbb{R}^{n \times d}$ mátrixszal jelölünk
egy $n$ példányból álló minibatchet, ahol minden példánynak $d$ bemenete (jellemzője) van.
Egy rejtett réteggel rendelkező MLP esetén, amelynek rejtett rétege $h$ rejtett egységből áll,
$\mathbf{H} \in \mathbb{R}^{n \times h}$ jelöli a rejtett réteg kimeneteit,
amelyeket *rejtett reprezentációknak* nevezzük.
Mivel mind a rejtett, mind a kimeneti réteg teljesen összekötött,
rejtett réteg súlyai $\mathbf{W}^{(1)} \in \mathbb{R}^{d \times h}$ és torzítása $\mathbf{b}^{(1)} \in \mathbb{R}^{1 \times h}$,
kimeneti réteg súlyai $\mathbf{W}^{(2)} \in \mathbb{R}^{h \times q}$ és torzítása $\mathbf{b}^{(2)} \in \mathbb{R}^{1 \times q}$.
Ez lehetővé teszi, hogy az egy rejtett rétegű MLP $\mathbf{O} \in \mathbb{R}^{n \times q}$ kimeneteit
a következőképpen számítsuk ki:

$$
\begin{aligned}
    \mathbf{H} & = \mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}, \\
    \mathbf{O} & = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}.
\end{aligned}
$$

Vegyük észre, hogy a rejtett réteg hozzáadásával
a modellünknek most további paraméterkészleteket kell nyomon követni és frissíteni.
Tehát mit nyertünk cserébe?
Meglepő lehet megtudni,
hogy — a fenti modellben — *semmit sem nyertünk a fáradozásunkért*!
Az ok egyszerű.
A fenti rejtett egységek
a bemenetek affin függvényeivel adottak,
és a kimenetek (softmax előtt) csupán
a rejtett egységek affin függvényei.
Egy affin függvény affin függvénye
maga is affin függvény.
Ráadásul a lineáris modellünk már korábban is
képes volt bármilyen affin függvény reprezentálására.

Ennek formális belátásához egyszerűen összevonhatjuk a rejtett réteget a fenti definícióban,
egy egyenértékű egyréteges modellt kapva, amelynek paraméterei:
$\mathbf{W} = \mathbf{W}^{(1)}\mathbf{W}^{(2)}$ és $\mathbf{b} = \mathbf{b}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)}$:

$$
\mathbf{O} = (\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)})\mathbf{W}^{(2)} + \mathbf{b}^{(2)} = \mathbf{X} \mathbf{W}^{(1)}\mathbf{W}^{(2)} + \mathbf{b}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)} = \mathbf{X} \mathbf{W} + \mathbf{b}.
$$

A többrétegű architektúrák potenciáljának kiaknázásához
szükségünk van még egy kulcselemre: egy
nemlineáris *aktivációs függvényre* $\sigma$,
amelyet minden rejtett egységre alkalmazunk
az affin transzformáció után. Például egy népszerű
választás a ReLU (rectified linear unit) aktivációs függvény :cite:`Nair.Hinton.2010`
$\sigma(x) = \mathrm{max}(0, x)$, amely elemenként hat az argumentumaira.
Az aktivációs függvények $\sigma(\cdot)$ kimeneteit *aktivációknak* nevezzük.
Általában, ha aktivációs függvényeket helyezünk el,
az MLP már nem redukálható lineáris modellre:

$$
\begin{aligned}
    \mathbf{H} & = \sigma(\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}), \\
    \mathbf{O} & = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}.\\
\end{aligned}
$$

Mivel az $\mathbf{X}$ minden sora a minibatch egy példányának felel meg,
némi jelölésvisszaéléssel a nemlinearitást
$\sigma$-t soronként alkalmazzuk a bemenetekre,
azaz egyszerre egy példányra.
Vegyük észre, hogy ugyanezt a jelölést használtuk a softmax esetén
is, amikor soronkénti műveletet jelöltünk a :numref:`subsec_softmax_vectorization` részben.
Az általunk használt aktivációs függvények nem csupán soronként, hanem
elemenként hatnak. Ez azt jelenti, hogy a réteg lineáris részének kiszámítása után
minden aktivációt ki tudunk számítani
a többi rejtett egység értékeinek vizsgálata nélkül.

Általánosabb MLP-k felépítéséhez folytathatjuk
ilyen rejtett rétegek egymásra helyezését,
pl. $\mathbf{H}^{(1)} = \sigma_1(\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)})$
és $\mathbf{H}^{(2)} = \sigma_2(\mathbf{H}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)})$,
egymás tetejére helyezve, egyre kifejezőbb modelleket eredményezve.

### Univerzális közelítők

Tudjuk, hogy az agy nagyon kifinomult statisztikai elemzésre képes. Ezért
érdemes megkérdezni, *milyen erős* lehet egy mély hálózat. Ezt a kérdést
több alkalommal is megválaszolták, pl. :citet:`Cybenko.1989` az
MLP-k kontextusában, és :citet:`micchelli1984interpolation` a reprodukáló kernel
Hilbert-terek kontextusában, ami radiális bázisfüggvény (RBF) hálózatoknak tekinthető egyetlen rejtett réteggel.
Ezek (és kapcsolódó eredmények) azt sugallják, hogy még egyetlen rejtett rétegű hálózattal is,
elegendő csomópont esetén (esetleg abszurd sok csomóponttal),
és a megfelelő súlyokkal,
bármilyen függvényt modellezhetünk.
Azonban ezt a függvényt valóban megtanulni a nehéz rész.
Gondolhat a neurális hálózatára
úgy, mint a C programozási nyelvre.
A nyelv, mint minden más modern nyelv,
képes bármilyen kiszámítható programot kifejezni.
De valóban egy programot kitalálni,
amely megfelel a specifikációknak, az a nehéz rész.

Ráadásul csak azért, mert egy rejtett rétegű hálózat
*képes* bármilyen függvényt megtanulni,
nem jelenti, hogy minden problémát
ezzel kellene megoldani. Valójában ebben az esetben a kernel módszerek
sokkal hatékonyabbak, mivel képesek a problémát *pontosan* megoldani
végtelen dimenziós terekben is :cite:`Kimeldorf.Wahba.1971,Scholkopf.Herbrich.Smola.2001`.
Valójában sok függvényt sokkal kompaktabban közelíthetünk
mélyebb (nem szélesebb) hálózatok használatával :cite:`Simonyan.Zisserman.2014`.
A következő fejezetekben szigorúbb érvekre is kitérünk.


## Aktivációs függvények
:label:`subsec_activation-functions`

Az aktivációs függvények döntik el, hogy egy neuront aktiválni kell-e vagy sem, a
súlyozott összeg kiszámításával és torzítás hozzáadásával.
Differenciálható operátorok, amelyek bemeneti jeleket kimenetekké alakítanak,
és legtöbbjük nemlinearitást ad hozzá.
Mivel az aktivációs függvények alapvetők a mély tanulásban,
(**tekintsünk röviden néhány közismert egyet**).

### ReLU függvény

A legnépszerűbb választás,
mind az implementálás egyszerűsége,
mind a sokféle prediktív feladaton mutatott jó teljesítménye miatt,
a *rectified linear unit* (*ReLU*) :cite:`Nair.Hinton.2010`.
[**A ReLU egy nagyon egyszerű nemlineáris transzformációt biztosít**].
Egy $x$ elem esetén a függvény
az elem és $0$ maximumaként van definiálva:

$$\operatorname{ReLU}(x) = \max(x, 0).$$

Informálisan a ReLU függvény csak a pozitív
elemeket tartja meg, és elveti az összes negatív elemet
azáltal, hogy a megfelelő aktivációkat 0-ra állítja.
A függvény szemléltetéséhez ábrázoljuk azt.
Mint látható, az aktivációs függvény darabonként lineáris.

```{.python .input}
%%tab mxnet
x = np.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    y = npx.relu(x)
d2l.plot(x, y, 'x', 'relu(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab pytorch
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab tensorflow
x = tf.Variable(tf.range(-8.0, 8.0, 0.1), dtype=tf.float32)
y = tf.nn.relu(x)
d2l.plot(x.numpy(), y.numpy(), 'x', 'relu(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab jax
x = jnp.arange(-8.0, 8.0, 0.1)
y = jax.nn.relu(x)
d2l.plot(x, y, 'x', 'relu(x)', figsize=(5, 2.5))
```

Amikor a bemenet negatív,
a ReLU függvény deriváltja 0,
és amikor a bemenet pozitív,
a ReLU függvény deriváltja 1.
Vegyük észre, hogy a ReLU függvény nem differenciálható
akkor, amikor a bemenet értéke pontosan 0.
Ezekben az esetekben alapértelmezetten a bal oldali
deriváltat használjuk, és azt mondjuk, hogy a derivált 0, amikor a bemenet 0.
Megtehetjük ezt, mert
a bemenet a valóságban soha nem lesz pontosan nulla (a matematikusok azt mondanák,
hogy nulla mértékű halmazon nem differenciálható).
Van egy régi mondás, hogy ha a finom határfeltételek számítanak,
valószínűleg (*valódi*) matematikát végzünk, nem mérnöki munkát.
Ez a konvencionális bölcsesség itt is érvényes lehet, vagy legalábbis az a tény,
hogy nem végzünk korlátos optimalizálást :cite:`Mangasarian.1965,Rockafellar.1970`.
Az alábbiakban ábrázoljuk a ReLU függvény deriváltját.

```{.python .input}
%%tab mxnet
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
```

```{.python .input}
%%tab pytorch
y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
```

```{.python .input}
%%tab tensorflow
with tf.GradientTape() as t:
    y = tf.nn.relu(x)
d2l.plot(x.numpy(), t.gradient(y, x).numpy(), 'x', 'grad of relu',
         figsize=(5, 2.5))
```

```{.python .input}
%%tab jax
grad_relu = vmap(grad(jax.nn.relu))
d2l.plot(x, grad_relu(x), 'x', 'grad of relu', figsize=(5, 2.5))
```

A ReLU használatának oka az,
hogy deriváltjai különösen jól viselkednek:
vagy eltűnnek, vagy egyszerűen átengedik az argumentumot.
Ez az optimalizálást jobban kezelhetővé teszi,
és enyhíti az elhaló gradiensek jól dokumentált problémáját,
amely a neurális hálózatok korábbi verzióit gyötörte (erről bővebben később).

Vegyük észre, hogy a ReLU függvénynek számos változata létezik,
beleértve a *parametrizált ReLU* (*pReLU*) függvényt :cite:`He.Zhang.Ren.ea.2015`.
Ez a változat egy lineáris tagot ad a ReLU-hoz,
így néhány információ még akkor is átjut,
amikor az argumentum negatív:

$$\operatorname{pReLU}(x) = \max(0, x) + \alpha \min(0, x).$$

### Sigmoid függvény

[**A *sigmoid függvény* azokat a bemeneteket alakítja át**],
amelyek értékei a $\mathbb{R}$ tartományban vannak,
(**a (0, 1) intervallumon lévő kimenetekké.**)
Ezért a sigmoidt
gyakran *összenyomó függvénynek* nevezik:
az (−inf, inf) tartomány bármely bemenetét
a (0, 1) tartomány valamelyik értékére nyomja össze:

$$\operatorname{sigmoid}(x) = \frac{1}{1 + \exp(-x)}.$$

A korai neurális hálózatokban a kutatók
biológiai neuronok modellezésével foglalkoztak,
amelyek vagy *tüzelnek*, vagy *nem tüzelnek*.
Ezért ennek a területnek az úttörői,
egészen McCulloch és Pittsig,
a mesterséges neuron feltalálóiig visszamenőleg,
a küszöbegységekre összpontosítottak :cite:`McCulloch.Pitts.1943`.
Egy küszöb-aktiváció értéke 0,
ha a bemenete egy bizonyos küszöb alatt van,
és 1, ha a bemenet meghaladja a küszöböt.

Amikor a figyelem a gradiens alapú tanulás felé fordult,
a sigmoid függvény természetes választásnak tűnt,
mert egy sima, differenciálható
közelítése a küszöbegységnek.
A sigmoiddot ma is széles körben alkalmazzák
aktivációs függvényként a kimeneti egységekben,
amikor a kimeneteket bináris osztályozási feladatok valószínűségeként
szeretnénk értelmezni: a sigmoid a softmax speciális eseteként fogható fel.
Azonban a sigmoidt nagymértékben felváltotta
az egyszerűbb és könnyebben tanítható ReLU
a rejtett rétegek többségében. Ez nagyrészt azzal függ össze,
hogy a sigmoid kihívásokat jelent az optimalizálás során
:cite:`LeCun.Bottou.Orr.ea.1998`, mivel gradiensei nagy pozitív *és* negatív argumentumok esetén eltűnnek.
Ez olyan fennsíkokhoz vezethet, amelyekből nehéz kiszabadulni.
Mindazonáltal a sigmoidok fontosak. A visszatérő neurális hálózatokról szóló
(pl. :numref:`sec_lstm`) későbbi fejezetekben olyan architektúrákat írunk le,
amelyek sigmoid egységeket használnak az időbeli információáramlás szabályozására.

Az alábbiakban a sigmoid függvényt ábrázoljuk.
Vegyük észre, hogy amikor a bemenet közel van a 0-hoz,
a sigmoid függvény megközelíti
a lineáris transzformációt.

```{.python .input}
%%tab mxnet
with autograd.record():
    y = npx.sigmoid(x)
d2l.plot(x, y, 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab pytorch
y = torch.sigmoid(x)
d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab tensorflow
y = tf.nn.sigmoid(x)
d2l.plot(x.numpy(), y.numpy(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab jax
y = jax.nn.sigmoid(x)
d2l.plot(x, y, 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

A sigmoid függvény deriváltját a következő egyenlet adja meg:

$$\frac{d}{dx} \operatorname{sigmoid}(x) = \frac{\exp(-x)}{(1 + \exp(-x))^2} = \operatorname{sigmoid}(x)\left(1-\operatorname{sigmoid}(x)\right).$$


A sigmoid függvény deriváltját az alábbiakban ábrázoljuk.
Vegyük észre, hogy amikor a bemenet 0,
a sigmoid függvény deriváltja
0,25-ös maximumot ér el.
Ahogy a bemenet 0-tól bármelyik irányban eltávolodik,
a derivált 0-hoz közelít.

```{.python .input}
%%tab mxnet
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
```

```{.python .input}
%%tab pytorch
# Előző gradiensek törlése
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
```

```{.python .input}
%%tab tensorflow
with tf.GradientTape() as t:
    y = tf.nn.sigmoid(x)
d2l.plot(x.numpy(), t.gradient(y, x).numpy(), 'x', 'grad of sigmoid',
         figsize=(5, 2.5))
```

```{.python .input}
%%tab jax
grad_sigmoid = vmap(grad(jax.nn.sigmoid))
d2l.plot(x, grad_sigmoid(x), 'x', 'grad of sigmoid', figsize=(5, 2.5))
```

### Tanh függvény
:label:`subsec_tanh`

A sigmoid függvényhez hasonlóan [**a tanh (hiperbolikus tangens)
függvény is leképezi bemeneteit**],
az elemeket a (**$-1$ és $1$ közötti**) intervallumra korlátozva:

$$\operatorname{tanh}(x) = \frac{1 - \exp(-2x)}{1 + \exp(-2x)}.$$

Az alábbiakban a tanh függvényt ábrázoljuk. Vegyük észre, hogy amint a bemenet közelít a 0-hoz, a tanh függvény megközelíti a lineáris transzformációt. Bár a függvény alakja hasonló a sigmoid függvényéhez, a tanh függvény pontszimmetriát mutat a koordinátarendszer origójához képest :cite:`Kalman.Kwasny.1992`.

```{.python .input}
%%tab mxnet
with autograd.record():
    y = np.tanh(x)
d2l.plot(x, y, 'x', 'tanh(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab pytorch
y = torch.tanh(x)
d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab tensorflow
y = tf.nn.tanh(x)
d2l.plot(x.numpy(), y.numpy(), 'x', 'tanh(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab jax
y = jax.nn.tanh(x)
d2l.plot(x, y, 'x', 'tanh(x)', figsize=(5, 2.5))
```

A tanh függvény deriváltja:

$$\frac{d}{dx} \operatorname{tanh}(x) = 1 - \operatorname{tanh}^2(x).$$

Az alábbiakban ábrázolva látható.
Ahogy a bemenet közelít a 0-hoz,
a tanh függvény deriváltja közelít az 1-es maximumhoz.
Ahogy azt a sigmoid függvénynél is tapasztaltuk,
ahogy a bemenet 0-tól bármelyik irányban eltávolodik,
a tanh függvény deriváltja 0-hoz közelít.

```{.python .input}
%%tab mxnet
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
```

```{.python .input}
%%tab pytorch
# Előző gradiensek törlése
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
```

```{.python .input}
%%tab tensorflow
with tf.GradientTape() as t:
    y = tf.nn.tanh(x)
d2l.plot(x.numpy(), t.gradient(y, x).numpy(), 'x', 'grad of tanh',
         figsize=(5, 2.5))
```

```{.python .input}
%%tab jax
grad_tanh = vmap(grad(jax.nn.tanh))
d2l.plot(x, grad_tanh(x), 'x', 'grad of tanh', figsize=(5, 2.5))
```

## Összefoglalás és vita

Most már tudjuk, hogyan kell nemlinearitásokat bevezetni
kifejező többrétegű neurális hálózati architektúrák felépítéséhez.
Mellékesen megjegyezzük, hogy a tudásod már
nagyjából az 1990-es évek körüli szakemberek eszközkészletének szintjén van.
Bizonyos szempontokból előnyben vagy
az akkori szakemberekkel szemben,
mert kihasználhatod a hatékony
nyílt forráskódú mély tanulási keretrendszereket
a modellek gyors felépítéséhez, csupán néhány sornyi kóddal.
Korábban ezeknek a hálózatoknak a tanításához
a kutatóknak explicit módon kellett kódolniuk a rétegeket és deriváltakat
C-ben, Fortranban vagy akár Lispben (a LeNet esetében).

Másodlagos előny, hogy a ReLU lényegesen könnyebben optimalizálható,
mint a sigmoid vagy a tanh függvény. Mondhatjuk,
hogy ez volt az egyik kulcsinnovációk egyike, amely hozzájárult
a mély tanulás elmúlt évtizedes újjászületéséhez. Megjegyezzük azonban, hogy az
aktivációs függvényekkel kapcsolatos kutatás nem állt le.
Például a GELU (Gaussian error linear unit)
aktivációs függvény $x \Phi(x)$ :citet:`Hendrycks.Gimpel.2016` ($\Phi(x)$
a standard Gauss-féle kumulatív eloszlásfüggvény)
és a Swish aktivációs
függvény $\sigma(x) = x \operatorname{sigmoid}(\beta x)$, amelyet :citet:`Ramachandran.Zoph.Le.2017` javasolt,
sok esetben jobb pontosságot adhat.

## Feladatok

1. Mutasd meg, hogy egy *lineáris* mély hálózathoz rétegek hozzáadása, azaz egy nemlinearitás $\sigma$ nélküli hálózat esetén soha nem növeli a hálózat kifejezőerejét. Adj példát arra, ahol ez aktívan csökkenti azt.
1. Számítsd ki a pReLU aktivációs függvény deriváltját.
1. Számítsd ki a Swish aktivációs függvény $x \operatorname{sigmoid}(\beta x)$ deriváltját.
1. Mutasd meg, hogy a csak ReLU-t (vagy pReLU-t) használó MLP egy folytonos darabonként lineáris függvényt konstruál.
1. A sigmoid és a tanh nagyon hasonlók.
    1. Mutasd meg, hogy $\operatorname{tanh}(x) + 1 = 2 \operatorname{sigmoid}(2x)$.
    1. Bizonyítsd be, hogy mindkét nemlinearitás által parametrizált függvényosztályok azonosak. Tipp: az affin rétegeknek is vannak torzítás tagjaik.
1. Tegyük fel, hogy van egy nemlinearitásunk, amelyet egyszerre egy minibatchre alkalmazunk, például a batch normalizáció :cite:`Ioffe.Szegedy.2015`. Milyen problémákat okozhat ez?
1. Adj példát arra, ahol a gradiensek eltűnnek a sigmoid aktivációs függvény esetén.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/90)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/91)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/226)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17984)
:end_tab:
