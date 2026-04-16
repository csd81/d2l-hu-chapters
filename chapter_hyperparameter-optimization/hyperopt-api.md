```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(["pytorch"])
```

# Hiperparaméter-optimalizálási API
:label:`sec_api_hpo`

Mielőtt belemerülnénk a módszertanba, először egy olyan alapvető kódszerkezetet tárgyalunk, amely lehetővé teszi különféle HPO-algoritmusok hatékony megvalósítását. Általánosságban az itt vizsgált HPO-algoritmusoknak két döntési primitívet kell implementálniuk: a *keresést* és az *ütemezést*. Először új hiperparaméter-konfigurációkat kell mintavételezniük, ami gyakran valamilyen keresést jelent a konfigurációs térben. Másodszor minden konfigurációhoz ütemezniük kell az értékelést, és dönteniük kell arról, mennyi erőforrást rendelnek hozzá. Amint elkezdünk egy konfigurációt értékelni, azt *próbának* fogjuk nevezni. Ezeket a döntéseket két osztályhoz, a `HPOSearcher`-hez és a `HPOScheduler`-hez rendeljük. Emellett egy `HPOTuner` osztályt is biztosítunk, amely végrehajtja az optimalizálási folyamatot.

Az ütemező és a kereső ilyen koncepciója megtalálható népszerű HPO könyvtárakban is, például a Syne Tune-ban :cite:`salinas-automl22`, a Ray Tune-ban :cite:`liaw-arxiv18` vagy az Optuna-ban :cite:`akiba-sigkdd19`.

```{.python .input  n=2}
%%tab pytorch
import time
from d2l import torch as d2l
from scipy import stats
```

## Kereső

Az alábbiakban definiálunk egy kereső alaposztályt, amely az `sample_configuration` függvényen keresztül új jelölt konfigurációt ad. Ennek a függvénynek az egyik egyszerű megvalósítása az lenne, hogy a konfigurációkat egyenletesen, véletlenszerűen mintavételezzük, ahogy azt a :numref:`sec_what_is_hpo` szakaszban a véletlen keresésnél tettük. A kifinomultabb algoritmusok, például a Bayes-optimalizálás, a korábbi próbák teljesítménye alapján hozzák meg ezeket a döntéseket. Ennek eredményeként ezek az algoritmusok idővel ígéretesebb jelölteket képesek mintavételezni. Hozzáadjuk az `update` függvényt, hogy frissíthessük a korábbi próbák előzményeit, amelyeket aztán a mintavételi eloszlás javítására használhatunk.

```{.python .input  n=3}
%%tab pytorch
class HPOSearcher(d2l.HyperParameters):  #@save
    def sample_configuration() -> dict:
        raise NotImplementedError

    def update(self, config: dict, error: float, additional_info=None):
        pass
```

Az alábbi kód megmutatja, hogyan valósítható meg ebben az API-ban az előző szakaszban bemutatott véletlen keresőnk. Kis kiterjesztésként lehetővé tesszük, hogy a felhasználó az elsőként értékelendő konfigurációt az `initial_config` segítségével megadja, míg a későbbieket véletlenszerűen húzzuk.

```{.python .input  n=4}
%%tab pytorch
class RandomSearcher(HPOSearcher):  #@save
    def __init__(self, config_space: dict, initial_config=None):
        self.save_hyperparameters()

    def sample_configuration(self) -> dict:
        if self.initial_config is not None:
            result = self.initial_config
            self.initial_config = None
        else:
            result = {
                name: domain.rvs()
                for name, domain in self.config_space.items()
            }
        return result
```

## Ütemező

Az új próbák konfigurációinak mintavételezésén túl azt is el kell döntenünk, mikor és mennyi ideig futtassunk egy próbát. A gyakorlatban ezeket a döntéseket a `HPOScheduler` hozza meg, amely az új konfigurációk kiválasztását egy `HPOSearcher`-re bízza. A `suggest` metódus akkor hívódik meg, amikor valamilyen tanítási erőforrás felszabadul. A kereső `sample_configuration` függvényének meghívásán túl olyan paraméterekről is dönthet, mint a `max_epochs` (vagyis hogy meddig tanítsuk a modellt). Az `update` metódus akkor hívódik meg, amikor egy próba új megfigyelést ad vissza.

```{.python .input  n=5}
%%tab pytorch
class HPOScheduler(d2l.HyperParameters):  #@save
    def suggest(self) -> dict:
        raise NotImplementedError
    
    def update(self, config: dict, error: float, info=None):
        raise NotImplementedError
```

Ahhoz, hogy a véletlen keresést, de más HPO-algoritmusokat is megvalósítsunk, elegendő egy alap ütemező, amely mindig új konfigurációt ütemez, amikor új erőforrás válik elérhetővé.

```{.python .input  n=6}
%%tab pytorch
class BasicScheduler(HPOScheduler):  #@save
    def __init__(self, searcher: HPOSearcher):
        self.save_hyperparameters()

    def suggest(self) -> dict:
        return self.searcher.sample_configuration()

    def update(self, config: dict, error: float, info=None):
        self.searcher.update(config, error, additional_info=info)
```

## Hangoló

Végül szükségünk van egy komponensre, amely futtatja az ütemezőt/keresőt, és elvégzi az eredmények nyilvántartását. Az alábbi kód a HPO-próbák szekvenciális végrehajtását valósítja meg, ahol az egyik tanítási feladat a másik után kerül kiértékelésre, és ez alapvető példaként szolgál. Később a *Syne Tune*-t használjuk majd a skálázhatóbb, elosztott HPO-esetekhez.

```{.python .input  n=7}
%%tab pytorch
class HPOTuner(d2l.HyperParameters):  #@save
    def __init__(self, scheduler: HPOScheduler, objective: callable):
        self.save_hyperparameters()
        # Az eredmények nyilvántartása az ábrázoláshoz
        self.incumbent = None
        self.incumbent_error = None
        self.incumbent_trajectory = []
        self.cumulative_runtime = []
        self.current_runtime = 0
        self.records = []

    def run(self, number_of_trials):
        for i in range(number_of_trials):
            start_time = time.time()
            config = self.scheduler.suggest()
            print(f"Trial {i}: config = {config}")
            error = self.objective(**config)
            error = float(d2l.numpy(error.cpu()))
            self.scheduler.update(config, error)
            runtime = time.time() - start_time
            self.bookkeeping(config, error, runtime)
            print(f"    error = {error}, runtime = {runtime}")
```

## A HPO-algoritmusok teljesítményének nyilvántartása

Bármely HPO-algoritmusnál elsősorban a legjobban teljesítő konfigurációra (*incumbens*) és annak validációs hibájára vagyunk kíváncsiak egy adott falióra-idő után. Ezért követjük nyomon az egyes iterációk `runtime` értékét, amely magában foglalja mind az értékelés futtatásának idejét (`objective` hívás), mind a döntéshozatal idejét (`scheduler.suggest` hívás). A továbbiakban a `cumulative_runtime`-ot fogjuk ábrázolni az `incumbent_trajectory` ellenében, hogy szemléltessük a HPO-algoritmus *bármikor mérhető teljesítményét* a `scheduler` (és `searcher`) függvényében. Ez lehetővé teszi, hogy ne csak azt mérjük, milyen jól teljesít a megtalált konfiguráció, hanem azt is, milyen gyorsan találja meg az optimalizáló.

```{.python .input  n=8}
%%tab pytorch
@d2l.add_to_class(HPOTuner)  #@save
def bookkeeping(self, config: dict, error: float, runtime: float):
    self.records.append({"config": config, "error": error, "runtime": runtime})
    # Ellenőrizzük, hogy az utolsó hiperparaméter-konfiguráció
    # jobban teljesít-e, mint az incumbens
    if self.incumbent is None or self.incumbent_error > error:
        self.incumbent = config
        self.incumbent_error = error
    # Hozzáadjuk az éppen legjobb megfigyelt teljesítményt az optimalizálási pályához
    self.incumbent_trajectory.append(self.incumbent_error)
    # Frissítjük a futási időt
    self.current_runtime += runtime
    self.cumulative_runtime.append(self.current_runtime)
```

## Példa: egy konvolúciós neurális hálózat hiperparamétereinek optimalizálása

Most az új véletlen kereső megvalósításunkat használjuk a :numref:`sec_lenet` szakaszban bemutatott `LeNet` konvolúciós neurális hálózat *batch size*-ának és *learning rate*-jének optimalizálására. Kezdjük a célfüggvény definiálásával, amely ezúttal ismét a validációs hiba lesz.

```{.python .input  n=9}
%%tab pytorch
def hpo_objective_lenet(learning_rate, batch_size, max_epochs=10):  #@save
    model = d2l.LeNet(lr=learning_rate, num_classes=10)
    trainer = d2l.HPOTrainer(max_epochs=max_epochs, num_gpus=1)
    data = d2l.FashionMNIST(batch_size=batch_size)
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
    trainer.fit(model=model, data=data)
    validation_error = trainer.validation_error()
    return validation_error
```

A konfigurációs teret is definiálnunk kell. Emellett az elsőként értékelendő konfiguráció a :numref:`sec_lenet` szakaszban használt alapértelmezett beállítás lesz.

```{.python .input  n=10}
config_space = {
    "learning_rate": stats.loguniform(1e-2, 1),
    "batch_size": stats.randint(32, 256),
}
initial_config = {
    "learning_rate": 0.1,
    "batch_size": 128,
}
```

Most már elindíthatjuk a véletlen keresést:

```{.python .input}
searcher = RandomSearcher(config_space, initial_config=initial_config)
scheduler = BasicScheduler(searcher=searcher)
tuner = HPOTuner(scheduler=scheduler, objective=hpo_objective_lenet)
tuner.run(number_of_trials=5)
```

Az alábbiakban ábrázoljuk az incumbens optimalizálási pályáját, hogy megkapjuk a véletlen keresés bármikor mérhető teljesítményét:

```{.python .input  n=11}
board = d2l.ProgressBoard(xlabel="time", ylabel="error")
for time_stamp, error in zip(
    tuner.cumulative_runtime, tuner.incumbent_trajectory
):
    board.draw(time_stamp, error, "random search", every_n=1)
```

## HPO-algoritmusok összehasonlítása

Ahogy a tanítási algoritmusoknál vagy modellarchitektúráknál, itt is fontos megérteni, hogyan lehet a különböző HPO-algoritmusokat a legjobban összehasonlítani. Minden HPO-futás két fő véletlenszerűségforrástól függ: a tanítási folyamat véletlen hatásaitól, például a véletlen súlyinicializálástól vagy a mini-batch-ek sorrendjétől, valamint magának a HPO-algoritmusnak a belső véletlenszerűségétől, például a véletlen keresés mintavételezésétől. Ezért különböző algoritmusok összehasonlításakor elengedhetetlen, hogy minden kísérletet többször futtassunk, és statisztikákat, például átlagot vagy mediánt, jelentsünk az algoritmus több ismétléséből álló populációra, eltérő véletlenszám-generátor magokkal.

Ennek szemléltetésére összehasonlítjuk a véletlen keresést (lásd :numref:`sec_rs`) és a Bayes-optimalizálást :cite:`snoek-nips12` egy előrecsatolt neurális hálózat hiperparamétereinek hangolásán. Mindkét algoritmust $50$-szer értékeltük, különböző véletlen magokkal. A folytonos vonal az incumbens átlagos teljesítményét jelzi ezekben az $50$ ismétlésben, a szaggatott vonal pedig a szórást. Látható, hogy a véletlen keresés és a Bayes-optimalizálás nagyjából ugyanúgy teljesít körülbelül 1000 másodpercig, de a Bayes-optimalizálás képes a korábbi megfigyeléseket felhasználva jobb konfigurációkat azonosítani, és ezután gyorsan felülmúlja a véletlen keresést.


![Példa bármikor mérhető teljesítményábrára két algoritmus (A és B) összehasonlításához.](../img/example_anytime_performance.svg)
:label:`example_anytime_performance`

## Összefoglalás

Ez a szakasz egy egyszerű, mégis rugalmas felületet mutatott be különféle HPO-algoritmusok megvalósításához, amelyeket ebben a fejezetben vizsgálunk. Hasonló felületek megtalálhatók népszerű nyílt forráskódú HPO keretrendszerekben is. Azt is áttekintettük, hogyan hasonlíthatók össze a HPO-algoritmusok, és milyen buktatókra kell figyelni.

## Gyakorlatok

1. Ennek a gyakorlatnak a célja egy valamivel nehezebb HPO-feladat célfüggvényének implementálása és reálisabb kísérletek futtatása. A :numref:`sec_dropout` szakaszban megvalósított, két rejtett rétegű `DropoutMLP` MLP-t fogjuk használni.
    1. Kódold le a célfüggvényt, amelynek a modell összes hiperparaméterétől és a `batch_size`-tól kell függenie. Használd a `max_epochs=50` értéket. A GPU-k itt nem segítenek, ezért `num_gpus=0`. Tipp: módosítsd a `hpo_objective_lenet` függvényt.
    2. Válassz értelmes keresési teret, ahol a `num_hiddens_1`, `num_hiddens_2` egész számok az $[8, 1024]$ intervallumban, a dropout értékek $[0, 0.95]$ között vannak, míg a `batch_size` $[16, 384]$ tartományba esik. Adj kódot a `config_space`-hoz, megfelelő `scipy.stats` eloszlások használatával.
    3. Futtass véletlen keresést ezen a példán `number_of_trials=20` értékkel, és ábrázold az eredményeket. Először feltétlenül értékeld ki a :numref:`sec_dropout` alapértelmezett konfigurációját, amely a következő: `initial_config = {'num_hiddens_1': 256, 'num_hiddens_2': 256, 'dropout_1': 0.5, 'dropout_2': 0.5, 'lr': 0.1, 'batch_size': 256}`.
2. Ebben a gyakorlatban egy új keresőt fogsz megvalósítani (`HPOSearcher` alosztályát), amely a korábbi adatok alapján hoz döntéseket. A `probab_local`, `num_init_random` paraméterektől függ. Az `sample_configuration` metódusa a következőképpen működik: az első `num_init_random` hívásnál ugyanazt teszi, mint a `RandomSearcher.sample_configuration`. Ezután, `1 - probab_local` valószínűséggel, szintén ugyanazt teszi, mint a `RandomSearcher.sample_configuration`. Ellenkező esetben kiválasztja az eddig legkisebb validációs hibát elérő konfigurációt, véletlenszerűen kiválasztja annak egyik hiperparaméterét, és annak értékét véletlenszerűen mintavételezi, mint a `RandomSearcher.sample_configuration` esetében, miközben minden más értéket változatlanul hagy. Visszaadja ezt a konfigurációt, amely mindenben megegyezik az eddigi legjobb konfigurációval, kivéve ezt az egy hiperparamétert.
    1. Kódold le ezt az új `LocalSearcher`-t. Tipp: a keresődnek konstruktorargumentumként szüksége lesz a `config_space`-ra. Nyugodtan használhatsz `RandomSearcher` típusú tagot. Az `update` metódust is meg kell valósítanod.
    2. Futtasd újra az előző gyakorlat kísérletét, de a `RandomSearcher` helyett az új keresődet használd. Kísérletezz különböző `probab_local`, `num_init_random` értékekkel. Ugyanakkor ne feledd, hogy a különböző HPO-módszerek tisztességes összehasonlításához a kísérleteket többször meg kell ismételni, és ideális esetben több benchmark feladatot is figyelembe kell venni.


:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/12092)
:end_tab:
