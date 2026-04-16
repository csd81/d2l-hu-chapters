# Fordítók és értelmezők
:label:`sec_hybridize`

Eddig ez a könyv az imperatív programozásra összpontosított, amely olyan utasításokat használ, mint a `print`, `+` és `if`, hogy megváltoztassák egy program állapotát. Tekintsük az alábbi egyszerű imperatív program példáját.

```{.python .input}
#@tab all
def add(a, b):
    return a + b

def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g

print(fancy_func(1, 2, 3, 4))
```

A Python egy *értelmezett nyelv*. A fenti `fancy_func` függvény kiértékelésekor *sorban* hajtja végre a függvény törzsét alkotó műveleteket. Azaz kiértékeli az `e = add(a, b)` kifejezést, és az eredményt az `e` változóként tárolja, megváltoztatva ezzel a program állapotát. A következő két utasítás `f = add(c, d)` és `g = add(e, f)` hasonlóan hajtódik végre, összeadásokat végez és az eredményeket változókként tárolja. A :numref:`fig_compute_graph` ábra illusztrálja az adatfolyamot.

![Adatfolyam egy imperatív programban.](../img/computegraph.svg)
:label:`fig_compute_graph`

Bár az imperatív programozás kényelmes, nem feltétlenül hatékony. Egyrészt, még ha az `add` függvényt ismételten hívják is a `fancy_func`-ban, a Python egyenként hajtja végre a három függvényhívást. Ha ezeket például GPU-n (vagy akár több GPU-n) hajtják végre, a Python értelmezőből eredő overhead nyomasztóvá válhat. Ráadásul el kell mentenie az `e` és `f` változók értékeit, amíg a `fancy_func` összes utasítása végre nem hajtódik. Ennek oka, hogy nem tudjuk, hogy az `e` és `f` változókat a program más részei fogják-e használni az `e = add(a, b)` és `f = add(c, d)` utasítások végrehajtása után.

## Szimbolikus programozás

Tekintsük az alternatívát, a *szimbolikus programozást*, ahol a számítást általában csak akkor végzik el, ha a folyamat teljesen meg van határozva. Ezt a stratégiát több deep learning keretrendszer alkalmazza, köztük a Theano és a TensorFlow (az utóbbi imperatív kiterjesztéseket is szerzett). Általában a következő lépéseket foglalja magában:

1. A végrehajtandó műveletek definiálása.
1. A műveletek lefordítása végrehajtható programmá.
1. A szükséges bemenetek megadása és a lefordított program meghívása végrehajtásra.

Ez jelentős mértékű optimalizálást tesz lehetővé. Először is sok esetben kihagyhatjuk a Python értelmezőt, ezáltal eltávolítva egy teljesítményszűk keresztmetszetet, amely egy CPU-n egyetlen Python szállal párosított több gyors GPU-nál jelentőssé válhat.
Másodszor, a fordító optimalizálhatja és átírhatja a fenti kódot `print((1 + 2) + (3 + 4))` vagy akár `print(10)` formájába. Ez lehetséges, mivel a fordító a teljes kódot látja, mielőtt gépi utasításokká alakítaná. Például felszabadíthatja a memóriát (vagy soha nem foglalja le), amikor egy változóra már nincs szükség. Vagy teljesen átalakíthatja a kódot egy azzal egyenértékű részre.
Hogy jobban megértsük, tekintsük az alábbi imperatív programozás szimulációját (végül is Python).

```{.python .input}
#@tab all
def add_():
    return '''
def add(a, b):
    return a + b
'''

def fancy_func_():
    return '''
def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g
'''

def evoke_():
    return add_() + fancy_func_() + 'print(fancy_func(1, 2, 3, 4))'

prog = evoke_()
print(prog)
y = compile(prog, '', 'exec')
exec(y)
```

Az imperatív (értelmezett) programozás és a szimbolikus programozás közötti különbségek a következők:

* Az imperatív programozás egyszerűbb. Amikor imperatív programozást alkalmaznak Pythonban, a kód nagy része egyenes és könnyen megírható. Az imperatív programozási kód hibakeresése is egyszerűbb. Ennek oka, hogy könnyebb megszerezni és kinyomtatni az összes releváns közbenső változóértéket, vagy használni a Python beépített hibakereső eszközeit.
* A szimbolikus programozás hatékonyabb és könnyebben hordozható. A szimbolikus programozás megkönnyíti a kód fordítás közbeni optimalizálását, miközben lehetővé teszi a program Pythontól független formátumba való portolását. Ez lehetővé teszi a program futtatását nem Python környezetben, elkerülve ezzel a Python értelmezővel kapcsolatos potenciális teljesítményproblémákat.


## Hibrid programozás

Történelmileg a legtöbb deep learning keretrendszer imperatív vagy szimbolikus megközelítés közül választ. Például a Theano, a TensorFlow (az előbbi által inspirálva), a Keras és a CNTK szimbolikusan formalizálják a modelleket. Ezzel szemben a Chainer és a PyTorch imperatív megközelítést alkalmaz. Az imperatív módot a TensorFlow 2.0-hoz és a Kerashoz adták hozzá a későbbi revíziókban.

:begin_tab:`mxnet`
A Gluon tervezésekor a fejlesztők azon gondolkodtak, hogy lehetséges-e mindkét programozási paradigma előnyeinek kombinálása. Ez egy hibrid modellhez vezetett, amely lehetővé teszi a felhasználók számára, hogy tiszta imperatív programozással fejlesszenek és hibakeressenek, miközben képesek a legtöbb programot szimbolikus programmá alakítani, amelyek akkor futtathatók, ha termékszintű számítási teljesítmény és telepítés szükséges.

A gyakorlatban ez azt jelenti, hogy modelleket a `HybridBlock` vagy `HybridSequential` osztály segítségével építünk. Alapértelmezés szerint mindkettő ugyanúgy hajtódik végre, mint a `Block` vagy `Sequential` osztály az imperatív programozásban.
A `HybridSequential` osztály a `HybridBlock` alosztálya (akárcsak a `Sequential` alosztálya a `Block`-nak). Amikor a `hybridize` függvényt meghívják, a Gluon szimbolikus programozásban használt formára fordítja a modellt. Ez lehetővé teszi a számításintenzív összetevők optimalizálását anélkül, hogy feláldoznánk a modell megvalósítási módját. Az alábbiakban illusztráljuk az előnyöket, a szekvenciális modellekre és blokkokra összpontosítva.
:end_tab:

:begin_tab:`pytorch`
Ahogy fentebb megjegyeztük, a PyTorch imperatív programozáson alapul és dinamikus számítási gráfokat használ. A szimbolikus programozás hordozhatóságának és hatékonyságának kihasználása érdekében a fejlesztők megvizsgálták, hogy lehetséges-e mindkét programozási paradigma előnyeinek kombinálása. Ez egy torchscript-hez vezetett, amely lehetővé teszi a felhasználók számára, hogy tiszta imperatív programozással fejlesszenek és hibakeressenek, miközben képesek a legtöbb programot szimbolikus programmá alakítani, amelyek akkor futtathatók, ha termékszintű számítási teljesítmény és telepítés szükséges.
:end_tab:

:begin_tab:`tensorflow`
Az imperatív programozási paradigma már a TensorFlow 2 alapértelmezése, ami üdvözlendő változás azok számára, akik most ismerkednek a nyelvvel. Azonban ugyanazok a szimbolikus programozási technikák és az ezekből következő számítási gráfok még mindig léteznek a TensorFlow-ban, és a könnyen használható `tf.function` dekorátorral érhetők el. Ez hozta az imperatív programozási paradigmát a TensorFlow-ba, lehetővé téve a felhasználók számára, hogy intuitívabb függvényeket definiáljanak, majd becsomagolják és automatikusan számítási gráfokká fordítsák le őket a TensorFlow csapat által [autograph](https://www.tensorflow.org/api_docs/python/tf/autograph)-nak nevezett funkcióval.
:end_tab:

## A `Sequential` osztály hibridizálása

A hibridizáció működésének legegyszerűbb megértése a több rétegből álló mély hálózatok vizsgálatával érhető el. Hagyományosan a Python értelmezőnek az összes réteg kódját végre kell hajtania, hogy egy utasítást generáljon, amely aztán továbbítható egy CPU-ra vagy GPU-ra. Egyetlen (gyors) számítási eszköz esetén ez nem okoz komoly problémákat. Másrészt, ha fejlett 8 GPU-s szervert használunk, például egy AWS P3dn.24xlarge példányt, a Python nehezen tudja lefoglalni az összes GPU-t. Az egyszálú Python értelmező itt szűk keresztmetszetté válik. Nézzük meg, hogyan kezelhetjük ezt a kód jelentős részeinél a `Sequential` `HybridSequential`-lal való helyettesítésével. Kezdjük egy egyszerű MLP definiálásával.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

# Hálózatgyár
def get_net():
    net = nn.HybridSequential()  
    net.add(nn.Dense(256, activation='relu'),
            nn.Dense(128, activation='relu'),
            nn.Dense(2))
    net.initialize()
    return net

x = np.random.normal(size=(1, 512))
net = get_net()
net(x)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

# Hálózatgyár
def get_net():
    net = nn.Sequential(nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2))
    return net

x = torch.randn(size=(1, 512))
net = get_net()
net(x)
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
from tensorflow.keras.layers import Dense

# Hálózatgyár
def get_net():
    net = tf.keras.Sequential()
    net.add(Dense(256, input_shape = (512,), activation = "relu"))
    net.add(Dense(128, activation = "relu"))
    net.add(Dense(2, activation = "linear"))
    return net

x = tf.random.normal([1,512])
net = get_net()
net(x)
```

:begin_tab:`mxnet`
A `hybridize` függvény meghívásával képesek vagyunk lefordítani és optimalizálni a számítást az MLP-ben. A modell számítási eredménye változatlan marad.
:end_tab:

:begin_tab:`pytorch`
A modell `torch.jit.script` függvénnyel való konvertálásával képesek vagyunk lefordítani és optimalizálni a számítást az MLP-ben. A modell számítási eredménye változatlan marad.
:end_tab:

:begin_tab:`tensorflow`
Korábban a TensorFlow-ban épített összes függvény számítási gráfként épült, és ezért alapértelmezés szerint JIT-fordított volt. Azonban a TensorFlow 2.X és az EagerTensor megjelenésével ez már nem az alapértelmezett viselkedés.
Ezt a funkciót a tf.function-nel engedélyezhetjük újra. A tf.function-t általábban függvény dekorátorként használják, azonban közvetlenül is meghívható normál Python függvényként, az alábbiak szerint. A modell számítási eredménye változatlan marad.
:end_tab:

```{.python .input}
#@tab mxnet
net.hybridize()
net(x)
```

```{.python .input}
#@tab pytorch
net = torch.jit.script(net)
net(x)
```

```{.python .input}
#@tab tensorflow
net = tf.function(net)
net(x)
```

:begin_tab:`mxnet`
Ez szinte túl szépnek tűnik ahhoz, hogy igaz legyen: egyszerűen jelöljük meg a blokkot `HybridSequential`-ként, írjuk meg ugyanazt a kódot, mint korábban, és hívjuk meg a `hybridize` függvényt. Amint ez megtörténik, a hálózat optimalizálódik (az alábbiakban mérjük a teljesítményt). Sajnos ez nem működik varázslatosan minden réteg esetén. Megjegyezzük azonban, hogy egy réteg nem optimalizálódik, ha a `Block` osztályból örököl a `HybridBlock` osztály helyett.
:end_tab:

:begin_tab:`pytorch`
Ez szinte túl szépnek tűnik ahhoz, hogy igaz legyen: írjuk meg ugyanazt a kódot, mint korábban, és egyszerűen konvertáljuk a modellt a `torch.jit.script` segítségével. Amint ez megtörténik, a hálózat optimalizálódik (az alábbiakban mérjük a teljesítményt).
:end_tab:

:begin_tab:`tensorflow`
Ez szinte túl szépnek tűnik ahhoz, hogy igaz legyen: írjuk meg ugyanazt a kódot, mint korábban, és egyszerűen konvertáljuk a modellt a `tf.function` segítségével. Amint ez megtörténik, a hálózat számítási gráfként épül fel a TensorFlow MLIR közbenső reprezentációjában, és erősen optimalizálódik a fordítói szinten a gyors végrehajtás érdekében (az alábbiakban mérjük a teljesítményt).
A `jit_compile = True` jelző explicit hozzáadása a `tf.function()` híváshoz engedélyezi az XLA (Accelerated Linear Algebra, Gyorsított Lineáris Algebra) funkciót a TensorFlow-ban. Az XLA bizonyos esetekben tovább optimalizálhatja a JIT-fordított kódot. A gráf-módú végrehajtás ezen explicit definíció nélkül is engedélyezett, azonban az XLA bizonyos nagy lineáris algebra műveleteket (a deep learning alkalmazásokban látottakhoz hasonlókat) sokkal gyorsabbá tehet, különösen GPU környezetben.
:end_tab:

### Gyorsítás hibridizálással

A fordítással nyert teljesítményjavulás bemutatásához összehasonlítjuk a `net(x)` kiértékeléséhez szükséges időt hibridizálás előtt és után. Először definiáljunk egy osztályt az idő mérésére. Hasznos lesz a fejezet során, amikor a teljesítmény mérésére (és javítására) törekszünk.

```{.python .input}
#@tab all
#@save
class Benchmark:
    """A futási idő mérésére."""
    def __init__(self, description='Done'):
        self.description = description

    def __enter__(self):
        self.timer = d2l.Timer()
        return self

    def __exit__(self, *args):
        print(f'{self.description}: {self.timer.stop():.4f} sec')
```

:begin_tab:`mxnet`
Most kétszer hívhatjuk meg a hálózatot, egyszer hibridizálással és egyszer anélkül.
:end_tab:

:begin_tab:`pytorch`
Most kétszer hívhatjuk meg a hálózatot, egyszer torchscript-tel és egyszer anélkül.
:end_tab:

:begin_tab:`tensorflow`
Most háromszor hívhatjuk meg a hálózatot: egyszer mohó módon hajtva végre, egyszer gráf-módú végrehajtással, és ismét JIT-fordított XLA segítségével.
:end_tab:

```{.python .input}
#@tab mxnet
net = get_net()
with Benchmark('Without hybridization'):
    for i in range(1000): net(x)
    npx.waitall()

net.hybridize()
with Benchmark('With hybridization'):
    for i in range(1000): net(x)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
net = get_net()
with Benchmark('Without torchscript'):
    for i in range(1000): net(x)

net = torch.jit.script(net)
with Benchmark('With torchscript'):
    for i in range(1000): net(x)
```

```{.python .input}
#@tab tensorflow
net = get_net()
with Benchmark('Eager Mode'):
    for i in range(1000): net(x)

net = tf.function(net)
with Benchmark('Graph Mode'):
    for i in range(1000): net(x)
```

:begin_tab:`mxnet`
A fenti eredményekből látható, hogy miután egy `HybridSequential` példány meghívja a `hybridize` függvényt, a számítási teljesítmény javul a szimbolikus programozás alkalmazásával.
:end_tab:

:begin_tab:`pytorch`
A fenti eredményekből látható, hogy miután egy `nn.Sequential` példányt a `torch.jit.script` függvénnyel szkriptesítik, a számítási teljesítmény javul a szimbolikus programozás alkalmazásával.
:end_tab:

:begin_tab:`tensorflow`
A fenti eredményekből látható, hogy miután egy `tf.keras.Sequential` példányt a `tf.function` függvénnyel szkriptesítik, a számítási teljesítmény javul a szimbolikus programozás alkalmazásával a TensorFlow gráf-módú végrehajtásán keresztül.
:end_tab:

### Szerializáció

:begin_tab:`mxnet`
A modellek fordításának egyik előnye, hogy szerializálhatjuk (menthetjük) a modellt és paramétereit a lemezre. Ez lehetővé teszi számunkra, hogy a modellt a választott frontend nyelvtől független módon tároljuk. Ez lehetővé teszi a betanított modellek más eszközökre való telepítését és más frontend programozási nyelvek egyszerű használatát. Ugyanakkor a kód általában gyorsabb, mint ami az imperatív programozással érhető el. Nézzük meg az `export` függvényt működés közben.
:end_tab:

:begin_tab:`pytorch`
A modellek fordításának egyik előnye, hogy szerializálhatjuk (menthetjük) a modellt és paramétereit a lemezre. Ez lehetővé teszi számunkra, hogy a modellt a választott frontend nyelvtől független módon tároljuk. Ez lehetővé teszi a betanított modellek más eszközökre való telepítését és más frontend programozási nyelvek egyszerű használatát. Ugyanakkor a kód általában gyorsabb, mint ami az imperatív programozással érhető el. Nézzük meg a `save` függvényt működés közben.
:end_tab:

:begin_tab:`tensorflow`
A modellek fordításának egyik előnye, hogy szerializálhatjuk (menthetjük) a modellt és paramétereit a lemezre. Ez lehetővé teszi számunkra, hogy a modellt a választott frontend nyelvtől független módon tároljuk. Ez lehetővé teszi a betanított modellek más eszközökre való telepítését, más frontend programozási nyelvek egyszerű használatát, vagy a betanított modell szerveren való futtatását. Ugyanakkor a kód általában gyorsabb, mint ami az imperatív programozással érhető el.
A TensorFlow-ban mentéshez használható alacsony szintű API a `tf.saved_model`.
Nézzük meg a `saved_model` példányt működés közben.
:end_tab:

```{.python .input}
#@tab mxnet
net.export('my_mlp')
!ls -lh my_mlp*
```

```{.python .input}
#@tab pytorch
net.save('my_mlp')
!ls -lh my_mlp*
```

```{.python .input}
#@tab tensorflow
net = get_net()
tf.saved_model.save(net, 'my_mlp')
!ls -lh my_mlp*
```

:begin_tab:`mxnet`
A modell egy (nagy bináris) paraméterfájlra és a modell számítás végrehajtásához szükséges program JSON leírására bomlik. A fájlokat más, Python vagy MXNet által támogatott frontend nyelvek olvashatják, mint a C++, R, Scala és Perl. Nézzük meg a modell leírásának első néhány sorát.
:end_tab:

```{.python .input}
#@tab mxnet
!head my_mlp-symbol.json
```

:begin_tab:`mxnet`
Korábban bemutattuk, hogy a `hybridize` függvény meghívása után a modell kiváló számítási teljesítményt és hordozhatóságot képes elérni. Vegyük figyelembe azonban, hogy a hibridizáció befolyásolhatja a modell rugalmasságát, különösen a vezérlési folyamat tekintetében.

Ezenkívül, ellentétben a `Block` példánnyal, amelynek a `forward` függvényt kell használnia, a `HybridBlock` példánynak a `hybrid_forward` függvényt kell használnia.
:end_tab:

```{.python .input}
#@tab mxnet
class HybridNet(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(HybridNet, self).__init__(**kwargs)
        self.hidden = nn.Dense(4)
        self.output = nn.Dense(2)

    def hybrid_forward(self, F, x):
        print('module F: ', F)
        print('value  x: ', x)
        x = F.npx.relu(self.hidden(x))
        print('result  : ', x)
        return self.output(x)
```

:begin_tab:`mxnet`
A fenti kód egy egyszerű hálózatot valósít meg 4 rejtett egységgel és 2 kimenettel. A `hybrid_forward` függvény egy további `F` argumentumot vesz fel. Erre azért van szükség, mert a kód hibridizálásától függően kissé eltérő könyvtárat (`ndarray` vagy `symbol`) fog használni a feldolgozáshoz. Mindkét osztály nagyon hasonló funkciókat végez, és az MXNet automatikusan meghatározza az argumentumot. Hogy megértsük, mi történik, a függvény hívás részeként kinyomtatjuk az argumentumokat.
:end_tab:

```{.python .input}
#@tab mxnet
net = HybridNet()
net.initialize()
x = np.random.normal(size=(1, 3))
net(x)
```

:begin_tab:`mxnet`
A forward számítás megismétlése ugyanolyan kimenetet fog eredményezni (a részleteket kihagyjuk). Most nézzük meg, mi történik, ha meghívjuk a `hybridize` függvényt.
:end_tab:

```{.python .input}
#@tab mxnet
net.hybridize()
net(x)
```

:begin_tab:`mxnet`
Az `ndarray` helyett most a `symbol` modult használjuk az `F`-hez. Ezenkívül, bár a bemenet `ndarray` típusú, a hálózaton átfolyó adatok most `symbol` típusra konvertálódnak a fordítási folyamat részeként. A függvény hívás megismétlése meglepő eredményt hoz:
:end_tab:

```{.python .input}
#@tab mxnet
net(x)
```

:begin_tab:`mxnet` 
Ez meglehetősen eltérő attól, amit korábban láttunk. A `hybrid_forward`-ban definiált összes print utasítás ki van hagyva. Valóban, a hibridizáció után a `net(x)` végrehajtása többé nem vonja be a Python értelmezőt. Ez azt jelenti, hogy bármely felesleges Python kódot (például print utasításokat) kihagynak egy sokkal gördülékenyebb végrehajtás és jobb teljesítmény javára. Ehelyett az MXNet közvetlenül a C++ backend-et hívja meg. Vegyük figyelembe azt is, hogy néhány függvény nem támogatott a `symbol` modulban (pl. `asnumpy`), és a helyben végzett műveletek, mint az `a += b` és az `a[:] = a + b`, át kell írni `a = a + b` formába. Ennek ellenére a modellek fordítása megéri az erőfeszítést, amikor a sebesség számít. Az előny a kis százalékpontos javulástól a kétszeres sebességig terjedhet, a modell összetettségétől, a CPU sebességétől, valamint a GPU-k sebességétől és számától függően.
:end_tab:

## Összefoglalás


* Az imperatív programozás megkönnyíti az új modellek tervezését, mivel lehetséges vezérlési folyamatot tartalmazó kódot írni, és képes felhasználni a Python szoftver ökoszisztéma nagy részét.
* A szimbolikus programozás megköveteli, hogy a programot meghatározzuk és lefordítsuk, mielőtt végrehajtanánk. Az előny a jobb teljesítmény.

:begin_tab:`mxnet` 
* Az MXNet képes szükség szerint kombinálni mindkét megközelítés előnyeit.
* A `HybridSequential` és `HybridBlock` osztályok által épített modellek képesek az imperatív programokat szimbolikus programmá alakítani a `hybridize` függvény meghívásával.
:end_tab:


## Feladatok


:begin_tab:`mxnet` 
1. Add hozzá az `x.asnumpy()` függvényhívást a `HybridNet` osztály `hybrid_forward` függvényének első sorához. Futtasd a kódot és figyeld meg az észlelt hibákat. Miért következnek be?
1. Mi történik, ha vezérlési folyamatot adunk hozzá, azaz a Python `if` és `for` utasításokat a `hybrid_forward` függvénybe?
1. Tekintsd át az előző fejezetekben érdeklő modelleket. Javíthatod-e azok számítási teljesítményét újraimplementálással?
:end_tab:

:begin_tab:`pytorch,tensorflow` 
1. Tekintsd át az előző fejezetekben érdeklő modelleket. Javíthatod-e azok számítási teljesítményét újraimplementálással?
:end_tab:




:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/360)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/2490)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/2492)
:end_tab:
