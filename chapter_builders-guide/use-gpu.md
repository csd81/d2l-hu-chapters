```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# GPU-k
:label:`sec_use_gpu`

A :numref:`tab_intro_decade` táblázatban szemléltettük
a számítási teljesítmény rohamos növekedését az elmúlt két évtizedben.
Röviden, a GPU-k teljesítménye 2000 óta évtizedenként 1000-szeres mértékben nőtt.
Ez nagy lehetőségeket kínál, de azt is jelzi,
hogy ilyen teljesítmény iránt jelentős igény mutatkozott.


Ebben a részben tárgyalni kezdjük, hogyan lehet felhasználni
ezt a számítási teljesítményt a kutatáshoz.
Először egyetlen GPU segítségével, majd egy későbbi ponton,
hogyan lehet több GPU-t és több szervert (több GPU-val) használni.

Pontosabban, tárgyalni fogjuk, hogyan
kell egyetlen NVIDIA GPU-t használni a számításokhoz.
Először győzödjünk meg arról, hogy legalább egy NVIDIA GPU van telepítve.
Ezután töltsd le az [NVIDIA illesztőprogramot és CUDA-t](https://developer.nvidia.com/cuda-downloads)
és kövesd az utasításokat a megfelelő elérési út beállításához.
Ha ezek az előkészületek befejeződtek,
az `nvidia-smi` parancs használható
(**a grafikus kártya információinak megtekintéséhez**).

:begin_tab:`mxnet`
Talán észrevetted, hogy egy MXNet tenzor
szinte azonos egy NumPy `ndarray`-jel.
De néhány döntő különbség van.
Az egyik kulcsfontosságú jellemző, amely megkülönbözteti az MXNet-et
a NumPy-tól, az a különféle hardvereszközök támogatása.

Az MXNet-ben minden tömbnek van egy kontextusa.
Eddig alapértelmezés szerint minden változó
és a kapcsolódó számítás
a CPU-hoz volt rendelve.
Általában más kontextusok különféle GPU-k lehetnek.
A dolgok még bonyolultabbá válhatnak, ha
munkákat telepítünk több szerverre.
A tömbök intelligens kontextusokhoz való rendelésével
minimalizálhatjuk az eszközök közötti
adatátvitelre fordított időt.
Például, amikor neurális hálózatokat tanítunk GPU-val rendelkező szerveren,
általában inkább azt szeretnénk, hogy a modell paraméterei a GPU-n legyenek.

Ezután meg kell győzödnünk arról, hogy
az MXNet GPU-s verziója telepítve van.
Ha a CPU-s verziója már telepítve van,
előbb el kell távolítanunk.
Például a `pip uninstall mxnet` paranccsal,
majd a CUDA verziódnak megfelelő MXNet verziót telepíthetjük.
Feltéve, hogy CUDA 10.0 telepítve van,
a CUDA 10.0-t támogató MXNet verziót
a `pip install mxnet-cu100` paranccsal telepítheted.
:end_tab:

:begin_tab:`pytorch`
A PyTorch-ban minden tömbnek van egy eszköze; sokszor *kontextusnak* nevezzük.
Eddig alapértelmezés szerint minden változó
és a kapcsolódó számítás
a CPU-hoz volt rendelve.
Általában más kontextusok különféle GPU-k lehetnek.
A dolgok még bonyolultabbá válhatnak, ha
munkákat telepítünk több szerverre.
A tömbök intelligens kontextusokhoz való rendelésével
minimalizálhatjuk az eszközök közötti
adatátvitelre fordított időt.
Például, amikor neurális hálózatokat tanítunk GPU-val rendelkező szerveren,
általában inkább azt szeretnénk, hogy a modell paraméterei a GPU-n legyenek.
:end_tab:

Az ebben a részben szereplő programok futtatásához
legalább két GPU szükséges.
Vegyük észre, hogy ez a legtöbb asztali számítógép számára pazarlónak tűnhet,
de könnyen elérhető a felhőben, pl.
az AWS EC2 több GPU-s példányainak használatával.
Szinte az összes többi rész *nem* igényel több GPU-t, de itt csupán a különböző eszközök közötti adatáramlást szeretnénk szemléltetni.

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

## [**Számítási eszközök**]

Megadhatjuk az eszközöket, például CPU-kat és GPU-kat,
a tároláshoz és számításhoz.
Alapértelmezés szerint a tenzorok a főmemóriában jönnek létre,
majd a CPU-t használjuk a számításokhoz.

:begin_tab:`mxnet`
Az MXNet-ben a CPU és GPU a `cpu()` és `gpu()` segítségével jelölhető.
Fontos megjegyezni, hogy a `cpu()`
(vagy bármely egész szám a zárójelben)
minden fizikai CPU-t és memóriát jelent.
Ez azt jelenti, hogy az MXNet számításai
megpróbálják az összes CPU-magot használni.
Azonban a `gpu()` csak egy kártyát
és a megfelelő memóriát jelenti.
Ha több GPU van, a `gpu(i)` segítségével jelöljük
az $i$-edik GPU-t ($i$ nullától kezdődik).
Emellett a `gpu(0)` és `gpu()` egyenértékűek.
:end_tab:

:begin_tab:`pytorch`
A PyTorch-ban a CPU és GPU a `torch.device('cpu')` és `torch.device('cuda')` segítségével jelölhető.
Fontos megjegyezni, hogy a `cpu` eszköz
minden fizikai CPU-t és memóriát jelent.
Ez azt jelenti, hogy a PyTorch számításai
megpróbálják az összes CPU-magot használni.
Azonban egy `gpu` eszköz csak egy kártyát
és a megfelelő memóriát jelenti.
Ha több GPU van, a `torch.device(f'cuda:{i}')` segítségével jelöljük
az $i$-edik GPU-t ($i$ nullától kezdődik).
Emellett a `gpu:0` és `gpu` egyenértékűek.
:end_tab:

```{.python .input}
%%tab pytorch
def cpu():  #@save
    """A CPU eszköz lekérése."""
    return torch.device('cpu')

def gpu(i=0):  #@save
    """Egy GPU eszköz lekérése."""
    return torch.device(f'cuda:{i}')

cpu(), gpu(), gpu(1)
```

```{.python .input}
%%tab mxnet, tensorflow, jax
def cpu():  #@save
    """A CPU eszköz lekérése."""
    if tab.selected('mxnet'):
        return npx.cpu()
    if tab.selected('tensorflow'):
        return tf.device('/CPU:0')
    if tab.selected('jax'):
        return jax.devices('cpu')[0]

def gpu(i=0):  #@save
    """Egy GPU eszköz lekérése."""
    if tab.selected('mxnet'):
        return npx.gpu(i)
    if tab.selected('tensorflow'):
        return tf.device(f'/GPU:{i}')
    if tab.selected('jax'):
        return jax.devices('gpu')[i]

cpu(), gpu(), gpu(1)
```

(**Lekérdezhetjük az elérhető GPU-k számát.**)

```{.python .input}
%%tab pytorch
def num_gpus():  #@save
    """Az elérhető GPU-k számának lekérése."""
    return torch.cuda.device_count()

num_gpus()
```

```{.python .input}
%%tab mxnet, tensorflow, jax
def num_gpus():  #@save
    """Az elérhető GPU-k számának lekérése."""
    if tab.selected('mxnet'):
        return npx.num_gpus()
    if tab.selected('tensorflow'):
        return len(tf.config.experimental.list_physical_devices('GPU'))
    if tab.selected('jax'):
        try:
            return jax.device_count('gpu')
        except:
            return 0  # Nem található GPU háttérrendszer

num_gpus()
```

Most [**definiálunk két kényelmes függvényt, amelyek lehetővé teszik,
hogy kódot futtassunk akkor is, ha a kért GPU-k nem léteznek.**]

```{.python .input}
%%tab all
def try_gpu(i=0):  #@save
    """Ha létezik, gpu(i)-t ad vissza, különben cpu()-t."""
    if num_gpus() >= i + 1:
        return gpu(i)
    return cpu()

def try_all_gpus():  #@save
    """Az összes elérhető GPU-t adja vissza, vagy [cpu(),]-t, ha nincs GPU."""
    return [gpu(i) for i in range(num_gpus())]

try_gpu(), try_gpu(10), try_all_gpus()
```

## Tenzorok és GPU-k

:begin_tab:`pytorch`
Alapértelmezés szerint a tenzorok a CPU-n jönnek létre.
[**Lekérdezhetjük azt az eszközt, ahol a tenzor található.**]
:end_tab:

:begin_tab:`mxnet`
Alapértelmezés szerint a tenzorok a CPU-n jönnek létre.
[**Lekérdezhetjük azt az eszközt, ahol a tenzor található.**]
:end_tab:

:begin_tab:`tensorflow, jax`
Alapértelmezés szerint a tenzorok GPU-n/TPU-n jönnek létre, ha elérhetők,
egyébként CPU-t használnak, ha nem elérhetők.
[**Lekérdezhetjük azt az eszközt, ahol a tenzor található.**]
:end_tab:

```{.python .input}
%%tab mxnet
x = np.array([1, 2, 3])
x.ctx
```

```{.python .input}
%%tab pytorch
x = torch.tensor([1, 2, 3])
x.device
```

```{.python .input}
%%tab tensorflow
x = tf.constant([1, 2, 3])
x.device
```

```{.python .input}
%%tab jax
x = jnp.array([1, 2, 3])
x.device()
```

Fontos megjegyezni, hogy ha több tagot szeretnénk kezelni,
azoknak ugyanazon az eszközön kell lenniük.
Például, ha összeadunk két tenzort,
meg kell győzödnünk arról, hogy mindkét argumentum
ugyanazon az eszközön él – ellenkező esetben a keretrendszer
nem tudná, hol tárolja az eredményt,
sőt azt sem, hol végezze el a számítást.

### GPU-s tárolás

[**Tenzor GPU-n való tárolásának**] több módja is van.
Megadhatunk például egy tárolóeszközt a tenzor létrehozásakor.
Ezután létrehozzuk az `X` tenzorváltozót az első `gpu`-n.
A GPU-n létrehozott tenzor csak ennek a GPU-nak a memóriáját fogyasztja.
Az `nvidia-smi` paranccsal megtekinthetjük a GPU memóriakihasználást.
Általában ügyelnünk kell arra, hogy ne hozzunk létre a GPU memóriát meghaladó adatot.

```{.python .input}
%%tab mxnet
X = np.ones((2, 3), ctx=try_gpu())
X
```

```{.python .input}
%%tab pytorch
X = torch.ones(2, 3, device=try_gpu())
X
```

```{.python .input}
%%tab tensorflow
with try_gpu():
    X = tf.ones((2, 3))
X
```

```{.python .input}
%%tab jax
# Alapértelmezés szerint a JAX GPU-ra vagy TPU-ra helyezi a tömböket, ha elérhetők
X = jax.device_put(jnp.ones((2, 3)), try_gpu())
X
```

Feltéve, hogy legalább két GPU áll rendelkezésre, a következő kód (**véletlenszerű `Y` tenzort hoz létre a második GPU-n.**)

```{.python .input}
%%tab mxnet
Y = np.random.uniform(size=(2, 3), ctx=try_gpu(1))
Y
```

```{.python .input}
%%tab pytorch
Y = torch.rand(2, 3, device=try_gpu(1))
Y
```

```{.python .input}
%%tab tensorflow
with try_gpu(1):
    Y = tf.random.uniform((2, 3))
Y
```

```{.python .input}
%%tab jax
Y = jax.device_put(jax.random.uniform(jax.random.PRNGKey(0), (2, 3)),
                   try_gpu(1))
Y
```

### Másolás

[**Ha kiszámítjuk az `X + Y`-t,
el kell döntenünk, hol végezzük ezt a műveletet.**]
Például, ahogyan :numref:`fig_copyto` ábra mutatja,
átvihetjük az `X`-et a második GPU-ra
és ott végezhetjük el a műveletet.
*Ne* egyszerűen adjuk össze az `X`-et és `Y`-t,
mivel ez kivételt eredményezne.
A futási idejű motor nem tudná, mit tegyen:
nem tud adatot találni ugyanazon az eszközön, és meghibásodik.
Mivel `Y` a második GPU-n él,
oda kell mozgatnunk `X`-et, mielőtt összeadhatjuk a kettőt.

![Adatok másolása ugyanazon az eszközön való művelet végrehajtásához.](../img/copyto.svg)
:label:`fig_copyto`

```{.python .input}
%%tab mxnet
Z = X.copyto(try_gpu(1))
print(X)
print(Z)
```

```{.python .input}
%%tab pytorch
Z = X.cuda(1)
print(X)
print(Z)
```

```{.python .input}
%%tab tensorflow
with try_gpu(1):
    Z = X
print(X)
print(Z)
```

```{.python .input}
%%tab jax
Z = jax.device_put(X, try_gpu(1))
print(X)
print(Z)
```

Most, hogy [**az adatok (mind `Z`, mind `Y`) ugyanazon a GPU-n vannak, összeadhatjuk őket.**]

```{.python .input}
%%tab all
Y + Z
```

:begin_tab:`mxnet`
Képzeld el, hogy a `Z` változód már a második GPU-don él.
Mi történik, ha még mindig meghívjuk a `Z.copyto(gpu(1))` metódust?
Ez másolatot készít és új memóriát oszt ki,
még akkor is, ha a változó már a kívánt eszközön van.
Vannak esetek, amikor – a kód futtatási környezetétől függően –
két változó már ugyanazon az eszközön élhet.
Ezért csak akkor szeretnénk másolatot készíteni, ha a változók
jelenleg különböző eszközökön élnek.
Ilyen esetekben meghívhatjuk az `as_in_ctx` metódust.
Ha a változó már a megadott eszközön él,
akkor ez a művelet semmit sem tesz.
Hacsak kifejezetten nem szeretnél másolatot készíteni,
az `as_in_ctx` a javasolt metódus.
:end_tab:

:begin_tab:`pytorch`
De mi van, ha a `Z` változó már a második GPU-don él?
Mi történik, ha még mindig meghívjuk a `Z.cuda(1)` metódust?
Visszaadja a `Z`-t, ahelyett, hogy másolatot készítene és új memóriát osztana ki.
:end_tab:

:begin_tab:`tensorflow`
Képzeld el, hogy a `Z` változód már a második GPU-don él.
Mi történik, ha még mindig meghívjuk a `Z2 = Z` metódust ugyanazon eszközhatókörben?
Visszaadja a `Z`-t, ahelyett, hogy másolatot készítene és új memóriát osztana ki.
:end_tab:

:begin_tab:`jax`
Képzeld el, hogy a `Z` változód már a második GPU-don él.
Mi történik, ha még mindig meghívjuk a `Z2 = Z` metódust ugyanazon eszközhatókörben?
Visszaadja a `Z`-t, ahelyett, hogy másolatot készítene és új memóriát osztana ki.
:end_tab:

```{.python .input}
%%tab mxnet
Z.as_in_ctx(try_gpu(1)) is Z
```

```{.python .input}
%%tab pytorch
Z.cuda(1) is Z
```

```{.python .input}
%%tab tensorflow
with try_gpu(1):
    Z2 = Z
Z2 is Z
```

```{.python .input}
%%tab jax
Z2 = jax.device_put(Z, try_gpu(1))
Z2 is Z
```

### Mellékjegyzetek

Az emberek azért használnak GPU-kat gépi tanuláshoz,
mert gyorsnak tartják őket.
De a változók átvitele eszközök között lassú: sokkal lassabb, mint a számítás.
Ezért azt szeretnénk, hogy 100%-ig biztos legyél
abban, hogy valami lassút szeretnél csinálni, mielőtt megteheted.
Ha a deep learning keretrendszer egyszerűen automatikusan másolna,
összeomlás nélkül, talán észre sem vennéd,
hogy lassú kódot írtál.

Az adatátvitel nem csak lassú, hanem a párhuzamosítást is sokkal nehezebbé teszi,
mivel meg kell várnunk, amíg az adatokat elküldik (vagy inkább megkapják),
mielőtt folytathatjuk a további műveleteket.
Ezért a másolási műveleteket nagy körültekintéssel kell végrehajtani.
Ökölszabályként sok kis művelet
sokkal rosszabb, mint egy nagy művelet.
Ezenkívül egyszerre több művelet
sokkal jobb, mint sok egyszeri művelet elszórtan a kódban,
hacsak nem tudod, mit csinálsz.
Ez azért van, mert az ilyen műveletek blokkolódhatnak, ha egy eszköznek
meg kell várnia a másikat, mielőtt mást tehetne.
Ez egy kicsit olyan, mint a kávéd sorban megrendelni,
ahelyett, hogy telefonon előre megrendelnéd
és amikor odaérsz, már készen találnád.

Végül, amikor tenzorokat nyomtatunk ki, vagy tenzorokat NumPy formátumba konvertálunk,
ha az adatok nem a főmemóriában vannak,
a keretrendszer először átmásolja őket a főmemóriába,
ami további átviteli terhelést eredményez.
Ami még rosszabb, ez most a retteget globális interpreter zárolásnak van kitéve,
amely mindent várakoztat, amíg a Python befejeződik.


## [**Neurális hálózatok és GPU-k**]

Hasonlóképpen, egy neurális hálózat modell megadhat eszközöket.
A következő kód a modell paramétereit a GPU-ra helyezi.

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(ctx=try_gpu())
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(1))
net = net.to(device=try_gpu())
```

```{.python .input}
%%tab tensorflow
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    net = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1)])
```

```{.python .input}
%%tab jax
net = nn.Sequential([nn.Dense(1)])

key1, key2 = jax.random.split(jax.random.PRNGKey(0))
x = jax.random.normal(key1, (10,))  # Próbabemenet
params = net.init(key2, x)  # Inicializálási hívás
```

A következő fejezetekben számos példát fogunk látni
arra, hogyan kell modelleket GPU-kon futtatni,
egyszerűen azért, mert a modellek valamivel számításigényesebbé válnak.

Például, ha a bemenet egy GPU-n lévő tenzor, a modell ugyanazon a GPU-n számítja ki az eredményt.

```{.python .input}
%%tab mxnet, pytorch, tensorflow
net(X)
```

```{.python .input}
%%tab jax
net.apply(params, x)
```

(**Győzödjünk meg arról, hogy a modell paraméterei ugyanazon a GPU-n vannak tárolva.**)

```{.python .input}
%%tab mxnet
net[0].weight.data().ctx
```

```{.python .input}
%%tab pytorch
net[0].weight.data.device
```

```{.python .input}
%%tab tensorflow
net.layers[0].weights[0].device, net.layers[0].weights[1].device
```

```{.python .input}
%%tab jax
print(jax.tree_util.tree_map(lambda x: x.device(), params))
```

Tegyük lehetővé a tréner számára a GPU támogatást.

```{.python .input}
%%tab mxnet
@d2l.add_to_class(d2l.Module)  #@save
def set_scratch_params_device(self, device):
    for attr in dir(self):
        a = getattr(self, attr)
        if isinstance(a, np.ndarray):
            with autograd.record():
                setattr(self, attr, a.as_in_ctx(device))
            getattr(self, attr).attach_grad()
        if isinstance(a, d2l.Module):
            a.set_scratch_params_device(device)
        if isinstance(a, list):
            for elem in a:
                elem.set_scratch_params_device(device)
```

```{.python .input}
%%tab mxnet, pytorch
@d2l.add_to_class(d2l.Trainer)  #@save
def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
    self.save_hyperparameters()
    self.gpus = [d2l.gpu(i) for i in range(min(num_gpus, d2l.num_gpus()))]

@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_batch(self, batch):
    if self.gpus:
        batch = [d2l.to(a, self.gpus[0]) for a in batch]
    return batch

@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_model(self, model):
    model.trainer = self
    model.board.xlim = [0, self.max_epochs]
    if self.gpus:
        if tab.selected('mxnet'):
            model.collect_params().reset_ctx(self.gpus[0])
            model.set_scratch_params_device(self.gpus[0])
        if tab.selected('pytorch'):
            model.to(self.gpus[0])
    self.model = model
```

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Trainer)  #@save
def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
    self.save_hyperparameters()
    self.gpus = [d2l.gpu(i) for i in range(min(num_gpus, d2l.num_gpus()))]

@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_batch(self, batch):
    if self.gpus:
        batch = [d2l.to(a, self.gpus[0]) for a in batch]
    return batch
```

Röviden, amíg minden adat és paraméter ugyanazon az eszközön van, hatékonyan taníthatunk modelleket. A következő fejezetekben több ilyen példát fogunk látni.

## Összefoglalás

Megadhatunk eszközöket a tároláshoz és számításhoz, például CPU-t vagy GPU-t.
  Alapértelmezés szerint az adatok a főmemóriában jönnek létre,
  majd a CPU-t használják a számításokhoz.
A deep learning keretrendszer megköveteli, hogy a számításhoz szükséges összes bemeneti adat
  ugyanazon az eszközön legyen,
  legyen az CPU vagy ugyanaz a GPU.
Jelentős teljesítményt veszíthetünk, ha gondatlanul mozgatjuk az adatokat.
  Egy tipikus hiba: a veszteség kiszámítása
  minden minibatch esetén a GPU-n, majd visszajelzése a felhasználónak
  a parancssorban (vagy NumPy `ndarray`-be naplózva)
  globális interpreter zárolást vált ki, amely megakasztja az összes GPU-t.
  Sokkal jobb memóriát kiosztani
  a naplózáshoz a GPU-n belül, és csak nagyobb naplókat mozgatni.

## Feladatok

1. Próbálj meg egy nagyobb számítási feladatot, például nagy mátrixok szorzatát,
   és figyeld meg a sebesség különbségét a CPU és GPU között.
   Mi a helyzet egy kis számításigényű feladattal?
1. Hogyan kell olvasni és írni modell paramétereket a GPU-n?
1. Mérd meg, mennyi időt vesz igénybe 1000 darab
   $100 \times 100$-as mátrix-mátrix szorzat kiszámítása
   és a kimeneti mátrix Frobenius-normájának naplózása egy eredménnyel egyenként. Hasonlítsd össze a GPU-n belüli napló tárolásával és csak a végeredmény átvitelével.
1. Mérd meg, mennyi időt vesz igénybe két mátrix-mátrix szorzat
   egyidejű elvégzése két GPU-n. Hasonlítsd össze az egyetlen GPU-n sorban való számítással. Tipp: közel lineáris skálázást kell látnod.

:begin_tab:`mxnet`
[Megbeszélések](https://discuss.d2l.ai/t/62)
:end_tab:

:begin_tab:`pytorch`
[Megbeszélések](https://discuss.d2l.ai/t/63)
:end_tab:

:begin_tab:`tensorflow`
[Megbeszélések](https://discuss.d2l.ai/t/270)
:end_tab:

:begin_tab:`jax`
[Megbeszélések](https://discuss.d2l.ai/t/17995)
:end_tab:
