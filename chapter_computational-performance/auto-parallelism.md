# Automatikus párhuzamosság
:label:`sec_auto_para`


A deep learning keretrendszerek (pl. MXNet és PyTorch) automatikusan építenek számítási gráfokat a backend-en. A számítási gráf segítségével a rendszer tisztában van az összes függőséggel,
és szelektíven hajthat végre párhuzamosan több egymástól független feladatot a
sebesség javítása érdekében. Például a :numref:`sec_async` szakasz :numref:`fig_asyncgraph` ábrája két változót egymástól függetlenül inicializál. Következésképpen a rendszer dönthet úgy, hogy párhuzamosan hajtja végre ezeket.


Általában egyetlen operátor az összes CPU összes számítási erőforrását vagy egyetlen GPU-t használ. Például a `dot` operátor az összes CPU összes magját (és szálát) felhasználja, még akkor is, ha egyetlen gépen több CPU processzor is van. Ugyanez vonatkozik egyetlen GPU-ra is. Ezért az egyes eszközös számítógépeknél a párhuzamosítás nem igazán hasznos. Több eszköz esetén a dolgok fontosabbak. Bár a párhuzamosítás általában leginkább a több GPU között releváns, a helyi CPU hozzáadása kissé növeli a teljesítményt. Például lásd :citet:`Hadjis.Zhang.Mitliagkas.ea.2016`, amely GPU-t és CPU-t kombináló számítógépes látás modellek tanítására összpontosít. Az automatikusan párhuzamosító keretrendszer kényelmével ugyanezt a célt néhány sor Python kóddal érhetjük el. Tágabb értelemben az automatikus párhuzamos számítás tárgyalása mind CPU-kat, mind GPU-kat használó párhuzamos számításra, valamint a számítás és kommunikáció párhuzamosítására összpontosít.

Vegyük figyelembe, hogy az ebben a szakaszban végzett kísérletek futtatásához legalább két GPU szükséges.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
```

## Párhuzamos számítás GPU-kon

Kezdjük egy referencia munkaterhelés definiálásával a teszteléshez: az alábbi `run` függvény 10 mátrix-mátrix szorzást végez a választott eszközön, két változóba allokált adatokat használva: `x_gpu1` és `x_gpu2`.

```{.python .input}
#@tab mxnet
devices = d2l.try_all_gpus()
def run(x):
    return [x.dot(x) for _ in range(50)]

x_gpu1 = np.random.uniform(size=(4000, 4000), ctx=devices[0])
x_gpu2 = np.random.uniform(size=(4000, 4000), ctx=devices[1])
```

```{.python .input}
#@tab pytorch
devices = d2l.try_all_gpus()
def run(x):
    return [x.mm(x) for _ in range(50)]

x_gpu1 = torch.rand(size=(4000, 4000), device=devices[0])
x_gpu2 = torch.rand(size=(4000, 4000), device=devices[1])
```

:begin_tab:`mxnet`
Most alkalmazzuk a függvényt az adatokra. Annak biztosítása érdekében, hogy a gyorsítótár ne játsszon szerepet az eredményekben, bemelegítjük az eszközöket egy egymenetes futtatással mindkettőn a mérés előtt.
:end_tab:

:begin_tab:`pytorch`
Most alkalmazzuk a függvényt az adatokra. Annak biztosítása érdekében, hogy a gyorsítótár ne játsszon szerepet az eredményekben, bemelegítjük az eszközöket egy egymenetes futtatással mindkettőn a mérés előtt. A `torch.cuda.synchronize()` megvárja, amíg egy CUDA eszközön az összes stream összes kernele befejeződik. Fogad egy `device` argumentumot, azt az eszközt, amelyhez szinkronizálnunk kell. Az aktuális eszközt használja, amelyet a `current_device()` ad meg, ha az eszközargumentum `None` (alapértelmezett).
:end_tab:

```{.python .input}
#@tab mxnet
run(x_gpu1)  # Mindkét eszköz bemelegítése
run(x_gpu2)
npx.waitall()

with d2l.Benchmark('GPU1 time'):
    run(x_gpu1)
    npx.waitall()

with d2l.Benchmark('GPU2 time'):
    run(x_gpu2)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
run(x_gpu1)
run(x_gpu2)  # Az összes eszköz bemelegítése
torch.cuda.synchronize(devices[0])
torch.cuda.synchronize(devices[1])

with d2l.Benchmark('GPU1 time'):
    run(x_gpu1)
    torch.cuda.synchronize(devices[0])

with d2l.Benchmark('GPU2 time'):
    run(x_gpu2)
    torch.cuda.synchronize(devices[1])
```

:begin_tab:`mxnet`
Ha eltávolítjuk a `waitall` utasítást mindkét feladat között, a rendszer szabadon párhuzamosíthatja a számítást mindkét eszközön automatikusan.
:end_tab:

:begin_tab:`pytorch`
Ha eltávolítjuk a `synchronize` utasítást mindkét feladat között, a rendszer szabadon párhuzamosíthatja a számítást mindkét eszközön automatikusan.
:end_tab:

```{.python .input}
#@tab mxnet
with d2l.Benchmark('GPU1 & GPU2'):
    run(x_gpu1)
    run(x_gpu2)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
with d2l.Benchmark('GPU1 & GPU2'):
    run(x_gpu1)
    run(x_gpu2)
    torch.cuda.synchronize()
```

A fenti esetben a teljes végrehajtási idő kisebb, mint a részek összege, mivel a deep learning keretrendszer automatikusan ütemezi a számítást mindkét GPU eszközön, anélkül, hogy a felhasználó részéről kifinomult kódra lenne szükség.



## Párhuzamos számítás és kommunikáció

Sok esetben adatokat kell mozgatni különböző eszközök között, például CPU és GPU között, vagy különböző GPU-k között.
Például
ez akkor fordul elő, amikor elosztott optimalizálást kívánunk végezni, ahol több gyorsítókártya gradienseit kell összesíteni. Szimuláljuk ezt GPU-n végzett számítással, majd az eredmények visszamásolásával a CPU-ra.

```{.python .input}
#@tab mxnet
def copy_to_cpu(x):
    return [y.copyto(npx.cpu()) for y in x]

with d2l.Benchmark('Run on GPU1'):
    y = run(x_gpu1)
    npx.waitall()

with d2l.Benchmark('Copy to CPU'):
    y_cpu = copy_to_cpu(y)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
def copy_to_cpu(x, non_blocking=False):
    return [y.to('cpu', non_blocking=non_blocking) for y in x]

with d2l.Benchmark('Run on GPU1'):
    y = run(x_gpu1)
    torch.cuda.synchronize()

with d2l.Benchmark('Copy to CPU'):
    y_cpu = copy_to_cpu(y)
    torch.cuda.synchronize()
```

:begin_tab:`mxnet`
Ez kissé nem hatékony. Vegyük figyelembe, hogy már elkezdhetjük a `y` részeinek másolását a CPU-ra, miközben a lista többi részét még mindig számítjuk. Ez a helyzet például akkor fordul elő, amikor egy mini-batch gradiensét számítjuk. Néhány paraméter gradiensei korábban lesznek elérhetők, mint másoké. Ezért az a javunkra válik, ha elkezdjük kihasználni a PCI-Express busz sávszélességét, miközben a GPU még fut. A `waitall` eltávolítása mindkét rész között lehetővé teszi, hogy szimuláljuk ezt a forgatókönyvet.
:end_tab:

:begin_tab:`pytorch`
Ez kissé nem hatékony. Vegyük figyelembe, hogy már elkezdhetjük a `y` részeinek másolását a CPU-ra, miközben a lista többi részét még mindig számítjuk. Ez a helyzet például akkor fordul elő, amikor egy mini-batch (backprop) gradiensét számítjuk. Néhány paraméter gradiensei korábban lesznek elérhetők, mint másoké. Ezért az a javunkra válik, ha elkezdjük kihasználni a PCI-Express busz sávszélességét, miközben a GPU még fut. A PyTorch-ban számos függvény, mint a `to()` és a `copy_()`, explicit `non_blocking` argumentumot fogad, amely lehetővé teszi a hívónak, hogy megkerülje a szinkronizálást, ha arra nincs szükség. A `non_blocking=True` beállítása lehetővé teszi, hogy szimuláljuk ezt a forgatókönyvet.
:end_tab:

```{.python .input}
#@tab mxnet
with d2l.Benchmark('Run on GPU1 and copy to CPU'):
    y = run(x_gpu1)
    y_cpu = copy_to_cpu(y)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
with d2l.Benchmark('Run on GPU1 and copy to CPU'):
    y = run(x_gpu1)
    y_cpu = copy_to_cpu(y, True)
    torch.cuda.synchronize()
```

Mindkét művelethez szükséges teljes idő (ahogy várható) kisebb, mint a részek összege.
Vegyük figyelembe, hogy ez a feladat különbözik a párhuzamos számítástól, mivel különböző erőforrást használ: a CPU és a GPU-k közötti buszt. Valójában mindkét eszközön párhuzamosan számíthatnánk és kommunikálhatnánk is egyszerre. Ahogy fentebb megjegyeztük, van egy függőség a számítás és a kommunikáció között: `y[i]`-t ki kell számítani, mielőtt másolni lehetne a CPU-ra. Szerencsére a rendszer képes a `y[i-1]`-et másolni, miközben a `y[i]`-t számítja, a teljes futási idő csökkentése érdekében.

Lezárásként bemutatjuk a számítási gráfot és annak függőségeit egy egyszerű kétréteges MLP esetén, amikor CPU-n és két GPU-n tanítunk, ahogy azt a :numref:`fig_twogpu` ábra mutatja. Eléggé fájdalmas lenne az ebből eredő párhuzamos programot manuálisan ütemezni. Ez az, ahol előnyös egy gráf alapú számítási backend az optimalizáláshoz.

![Egy kétréteges MLP számítási gráfja és annak függőségei CPU-n és két GPU-n.](../img/twogpu.svg)
:label:`fig_twogpu`


## Összefoglalás

* A modern rendszerek különféle eszközöket tartalmaznak, mint például több GPU és CPU. Ezek párhuzamosan, aszinkron módon használhatók.
* A modern rendszerek kommunikációhoz is különböző erőforrásokkal rendelkeznek, mint például PCI Express, tárolók (általában szilárdtest-meghajtók vagy hálózatokon keresztül) és hálózati sávszélesség. Ezek párhuzamosan is felhasználhatók a csúcshatékonyság érdekében.
* A backend javíthatja a teljesítményt automatikus párhuzamos számítással és kommunikációval.

## Feladatok

1. Nyolc műveletet hajtottak végre az ebben a szakaszban definiált `run` függvényben. Nincs köztük függőség. Tervezz egy kísérletet annak ellenőrzésére, hogy a deep learning keretrendszer automatikusan párhuzamosan hajtja-e végre őket.
1. Amikor egy adott operátor munkaterhelése elég kicsi, a párhuzamosítás még egyetlen CPU-n vagy GPU-n is segíthet. Tervezz egy kísérletet ennek ellenőrzésére.
1. Tervezz egy kísérletet, amely párhuzamos számítást használ CPU-kon, GPU-kon, és kommunikációt mindkét eszköz között.
1. Használj debuggert, mint például az NVIDIA [Nsight](https://developer.nvidia.com/nsight-compute-2019_5) eszközét, hogy ellenőrizd, a kódod hatékony-e.
1. Tervezz olyan számítási feladatokat, amelyek összetettebb adatfüggőségeket tartalmaznak, és futtass kísérleteket, hogy meggyőződj arról, hogy helyes eredményeket kapsz, miközben javítod a teljesítményt.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/362)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1681)
:end_tab:
