```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(["pytorch"])
```

# Többfidelitású hiperparaméter-optimalizálás
:label:`sec_mf_hpo`

A neurális hálózatok tanítása még közepes méretű adathalmazokon is költséges lehet.
A konfigurációs térről (:numref:`sec_intro_config_spaces`) függően a hiperparaméter-optimalizáláshoz tízektől akár százakig terjedő függvényértékelésre lehet szükség ahhoz, hogy egy jól teljesítő hiperparaméter-konfigurációt találjunk. Ahogy a :numref:`sec_rs_async` szakaszban láttuk, párhuzamos erőforrások kihasználásával jelentősen csökkenthetjük a HPO teljes falióra-idejét, de ez nem csökkenti a szükséges összes számítási mennyiséget.

Ebben a szakaszban megmutatjuk, hogyan gyorsítható fel a hiperparaméter-konfigurációk értékelése. Az olyan módszerek, mint a véletlen keresés, minden hiperparaméter-értékelésre ugyanannyi erőforrást osztanak ki (például epochok számát vagy tanítóadatpontok számát). A :numref:`img_samples_lc` ábra különböző hiperparaméter-konfigurációkkal tanított neurális hálózatok tanulási görbéit mutatja. Néhány epoch után már vizuálisan is meg tudjuk különböztetni a jól teljesítő és az alulteljesítő konfigurációkat. A tanulási görbék azonban zajosak, ezért még mindig szükség lehet a teljes 100 epochra ahhoz, hogy azonosítsuk a legjobbat.

![Véletlen hiperparaméter-konfigurációk tanulási görbéi](../img/samples_lc.svg)
:label:`img_samples_lc`

A többfidelitású hiperparaméter-optimalizálás több erőforrást oszt ki az ígéretes konfigurációknak, és idő előtt leállítja a gyengén teljesítőket.
Ez felgyorsítja az optimalizálási folyamatot, hiszen ugyanannyi teljes erőforrással több konfigurációt próbálhatunk ki.

Formálisabban kibővítjük a :numref:`sec_definition_hpo` szakaszban megadott definíciónkat úgy, hogy a célfüggvényünk $f(\mathbf{x}, r)$ egy további bemenetet kapjon: $r \in [r_{\mathrm{min}}, r_{max}]$, amely megadja, mennyi erőforrást vagyunk hajlandók a $\mathbf{x}$ konfiguráció értékelésére fordítani. Feltételezzük, hogy az $f(\mathbf{x}, r)$ hiba csökken $r$ növekedésével, míg a $c(\mathbf{x}, r)$ számítási költség növekszik. Tipikusan $r$ a neurális hálózat tanításának epochszámát jelenti, de lehet például a tanító részhalmaz mérete vagy a keresztvalidációs foldok száma is.

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import numpy as np
from scipy import stats
from collections import defaultdict
d2l.set_figsize()
```

## Egymás utáni felezés
:label:`sec_mf_hpo_sh`

A véletlen keresés többfidelitású beállításra való egyik legegyszerűbb adaptációja az *egymás utáni felezés* (*successive halving*) :cite:`jamieson-aistats16,karnin-icml13`. Az alapötlet az, hogy $N$ konfigurációval indulunk, például a konfigurációs térből véletlenszerűen mintavételezve, és mindegyiket csak $r_{\mathrm{min}}$ epochon át tanítjuk. Ezután eldobjuk a legrosszabbul teljesítő próbák egy részét, és a fennmaradókat tovább tanítjuk. A folyamatot ismételve egyre kevesebb próba fut egyre hosszabban, amíg legalább egy próba el nem éri a $r_{max}$ epochot.

Formálisabban vegyünk egy minimum költségkeretet $r_{\mathrm{min}}$ (például 1 epoch), egy maximum költségkeretet $r_{max}$, például az előző példában szereplő `max_epochs` értéket, valamint egy $\eta\in\{2, 3, \dots\}$ felezési konstansot. Az egyszerűség kedvéért tegyük fel, hogy $r_{max} = r_{\mathrm{min}} \eta^K$, ahol $K \in \mathbb{I}$ . Ekkor a kezdeti konfigurációk száma $N = \eta^K$. Definiáljuk a fokok halmazát: $\mathcal{R} = \{ r_{\mathrm{min}}, r_{\mathrm{min}}\eta, r_{\mathrm{min}}\eta^2, \dots, r_{max} \}$.

Az egymás utáni felezés egy köre a következőképpen zajlik. Először futtatjuk a $N$ próbát az első fokig, $r_{\mathrm{min}}$-ig. A validációs hibák sorrendbe rendezése után megtartjuk a legjobb $1 / \eta$ részt (ami $\eta^{K-1}$ konfigurációnak felel meg), a többit pedig elvetjük. A túlélő próbákat a következő fokig ($r_{\mathrm{min}}\eta$ epochig) tanítjuk, majd a folyamat ismétlődik. Minden foknál a próbák $1 / \eta$ része marad életben, és a tanításuk $\eta$-szor nagyobb költségvetéssel folytatódik. Ezzel az $N$ választással végül csak egyetlen próba jut el a teljes $r_{max}$ költségkeretig. Miután egy ilyen felezési kör befejeződött, új kezdeti konfigurációs készlettel kezdjük a következőt, és addig ismételjük, amíg a teljes költségkeret el nem fogy.

![Véletlen hiperparaméter-konfigurációk tanulási görbéi.](../img/sh.svg)

Az egymás utáni felezés megvalósításához a :numref:`sec_api_hpo` szakaszban bemutatott `HPOScheduler` alaposztályból származtatunk. Ez lehetővé teszi, hogy egy általános `HPOSearcher` objektum mintavételezze a konfigurációkat (az alábbi példában ez `RandomSearcher` lesz). Emellett a felhasználónak meg kell adnia a minimum erőforrást $r_{\mathrm{min}}$, a maximum erőforrást $r_{max}$ és az $\eta$ értékét. Az ütemezőn belül egy olyan konfigurációs sort tartunk fenn, amelyet még ki kell értékelni az aktuális $r_i$ fokon. A sort minden alkalommal frissítjük, amikor a következő fokra lépünk.

```{.python .input  n=2}
class SuccessiveHalvingScheduler(d2l.HPOScheduler):  #@save
    def __init__(self, searcher, eta, r_min, r_max, prefact=1):
        self.save_hyperparameters()
        # K kiszámítása, amelyet később a konfigurációk számának meghatározására használunk
        self.K = int(np.log(r_max / r_min) / np.log(eta))
        # A fokszintek meghatározása
        self.rung_levels = [r_min * eta ** k for k in range(self.K + 1)]
        if r_max not in self.rung_levels:
            # Az utolsó fokszintnek r_max-nak kell lennie
            self.rung_levels.append(r_max)
            self.K += 1
        # Nyilvántartás
        self.observed_error_at_rungs = defaultdict(list)
        self.all_observed_error_at_rungs = defaultdict(list)
        # A feldolgozási sor
        self.queue = []
```

Kezdetben a sor üres, és feltöltjük $n = \textrm{prefact} \cdot \eta^{K}$ konfigurációval, amelyeket először a legkisebb fokszinten, $r_{\mathrm{min}}$-en értékelünk ki. A $\textrm{prefact}$ paraméter lehetővé teszi a kód más kontextusban való újrafelhasználását. Jelen szakasz céljaira $\textrm{prefact} = 1$-et rögzítünk. Valahányszor erőforrás szabadul fel és a `HPOTuner` objektum meghívja a `suggest` függvényt, a sorból adunk vissza egy elemet. Ha az egymás utáni felezés egy köre befejeződött — azaz minden túlélő konfigurációt kiértékeltünk a legmagasabb $r_{max}$ erőforrásszinten, és a sor üres —, a teljes folyamatot újrakezdjük egy új, véletlenszerűen mintavételezett konfigurációs készlettel.

```{.python .input  n=12}
%%tab pytorch
@d2l.add_to_class(SuccessiveHalvingScheduler)  #@save
def suggest(self):
    if len(self.queue) == 0:
        # Új egymás utáni felezési kör indítása
        # A konfigurációk száma az első fokszinten:
        n0 = int(self.prefact * self.eta ** self.K)
        for _ in range(n0):
            config = self.searcher.sample_configuration()
            config["max_epochs"] = self.r_min  # r = r_min beállítása
            self.queue.append(config)
    # Egy elem visszaadása a sorból
    return self.queue.pop()
```

Amikor új adatpontot gyűjtöttünk, először frissítjük a kereső modult. Ezután ellenőrizzük, hogy az aktuális fokszinten összegyűjtöttük-e már az összes adatpontot. Ha igen, rendezzük az összes konfigurációt, és a legjobb $\frac{1}{\eta}$ konfigurációt betesszük a sorba.

```{.python .input  n=4}
%%tab pytorch
@d2l.add_to_class(SuccessiveHalvingScheduler)  #@save
def update(self, config: dict, error: float, info=None):
    ri = int(config["max_epochs"])  # r_i fokszint
    # A kereső frissítése, például ha később Bayes-optimalizálást használunk
    self.searcher.update(config, error, additional_info=info)
    self.all_observed_error_at_rungs[ri].append((config, error))
    if ri < self.r_max:
        # Nyilvántartás
        self.observed_error_at_rungs[ri].append((config, error))
        # Meg kell határozni, hány konfigurációt kell ezen a fokszinten kiértékelni
        ki = self.K - self.rung_levels.index(ri)
        ni = int(self.prefact * self.eta ** ki)
        # Ha minden konfigurációt megfigyeltünk ezen az r_i fokszinten,
        # megbecsüljük a legjobb 1 / eta konfigurációt, betesszük őket a sorba,
        # és előléptetjük őket a következő r_{i+1} fokszintre
        if len(self.observed_error_at_rungs[ri]) >= ni:
            kiplus1 = ki - 1
            niplus1 = int(self.prefact * self.eta ** kiplus1)
            best_performing_configurations = self.get_top_n_configurations(
                rung_level=ri, n=niplus1
            )
            riplus1 = self.rung_levels[self.K - kiplus1]  # r_{i+1}
            # A sor nem feltétlenül üres: az új elemeket az elejére szúrjuk be
            self.queue = [
                dict(config, max_epochs=riplus1)
                for config in best_performing_configurations
            ] + self.queue
            self.observed_error_at_rungs[ri] = []  # Visszaállítás
```

A konfigurációkat az aktuális fokszinten mért teljesítményük alapján rendezzük sorba.

```{.python .input  n=4}
%%tab pytorch

@d2l.add_to_class(SuccessiveHalvingScheduler)  #@save
def get_top_n_configurations(self, rung_level, n):
    rung = self.observed_error_at_rungs[rung_level]
    if not rung:
        return []
    sorted_rung = sorted(rung, key=lambda x: x[1])
    return [x[0] for x in sorted_rung[:n]]
```

Nézzük meg, hogyan teljesít az egymás utáni felezés a neurális hálózatos példánkon. A $r_{\mathrm{min}} = 2$, $\eta = 2$, $r_{max} = 10$ értékeket fogjuk használni, így a fokszintek: $2, 4, 8, 10$.

```{.python .input  n=5}
min_number_of_epochs = 2
max_number_of_epochs = 10
eta = 2
num_gpus=1

config_space = {
    "learning_rate": stats.loguniform(1e-2, 1),
    "batch_size": stats.randint(32, 256),
}
initial_config = {
    "learning_rate": 0.1,
    "batch_size": 128,
}
```

Mindössze az ütemezőt kell lecserélnünk az új `SuccessiveHalvingScheduler`-re.

```{.python .input  n=14}
searcher = d2l.RandomSearcher(config_space, initial_config=initial_config)
scheduler = SuccessiveHalvingScheduler(
    searcher=searcher,
    eta=eta,
    r_min=min_number_of_epochs,
    r_max=max_number_of_epochs,
)
tuner = d2l.HPOTuner(
    scheduler=scheduler,
    objective=d2l.hpo_objective_lenet,
)
tuner.run(number_of_trials=30)
```

Megjeleníthetjük az összes kiértékelt konfiguráció tanulási görbéit. A konfigurációk többségét korán leállítjuk, és csak a jobban teljesítők élik meg a $r_{max}$ epochot. Hasonlítsuk ezt össze az egyszerű véletlen kereséssel, amely minden konfigurációnak $r_{max}$ erőforrást osztana ki.

```{.python .input  n=19}
for rung_index, rung in scheduler.all_observed_error_at_rungs.items():
    errors = [xi[1] for xi in rung]
    d2l.plt.scatter([rung_index] * len(errors), errors)
d2l.plt.xlim(min_number_of_epochs - 0.5, max_number_of_epochs + 0.5)
d2l.plt.xticks(
    np.arange(min_number_of_epochs, max_number_of_epochs + 1),
    np.arange(min_number_of_epochs, max_number_of_epochs + 1)
)
d2l.plt.ylabel("validációs hiba")
d2l.plt.xlabel("epochok")
```

Végül érdemes megjegyezni a `SuccessiveHalvingScheduler` megvalósításának egy kisebb bonyolultságát. Tegyük fel, hogy egy munkás szabad feladatot vár, és a `suggest` hívás pillanatában az aktuális fokszint majdnem teljesen feltöltött, ám egy másik munkás még éppen kiértékelést végez. Mivel hiányzik a metrikaérték ettől a munkástól, nem tudjuk meghatározni a legjobb $1 / \eta$ részt a következő fokszint megnyitásához. Ugyanakkor a szabad munkáshoz feladatot szeretnénk rendelni, hogy ne maradjon tétlen. Megoldásként egy új egymás utáni felezési kört indítunk, és a munkást az ottani első próbához rendeljük. Ugyanakkor ha az `update` hívás során egy fokszint lezárul, gondoskodunk arról, hogy az új konfigurációk a sor elejére kerüljenek, így elsőbbséget élveznek a következő körből érkező konfigurációkkal szemben.

## Összefoglalás

Ebben a szakaszban bevezettük a többfidelitású hiperparaméter-optimalizálás fogalmát, ahol feltételezzük, hogy hozzáférünk a célfüggvény olcsón kiértékelhető közelítéseihez, például a tanítás bizonyos számú epochja utáni validációs hibához, amely a teljes epochszám utáni validációs hiba helyettesítőjeként szolgál.
A többfidelitású hiperparaméter-optimalizálás lehetővé teszi a HPO teljes számítási igényének csökkentését, nem csak a falióra-idő csökkentését.

Megvalósítottuk és kiértékeltük az egymás utáni felezést, egy egyszerű, mégis hatékony többfidelitású HPO-algoritmust.


:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/12094)
:end_tab:
