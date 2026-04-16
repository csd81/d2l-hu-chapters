```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(["pytorch"])
#required_libs("syne-tune[gpsearchers]==0.3.2")
```

# Aszinkron véletlen keresés
:label:`sec_rs_async`

Ahogy a :numref:`sec_api_hpo` szakaszban láttuk, a véletlen keresés jó hiperparaméter-konfigurációt akár órák vagy napok alatt is találhat csak a hiperparaméter-konfigurációk drága kiértékelése miatt. A gyakorlatban gyakran rendelkezésünkre áll egy erőforráskészlet, például ugyanazon a gépen több GPU, vagy több gép egy-egy GPU-val. Ez felveti a kérdést: *hogyan osztható el hatékonyan a véletlen keresés?*

Általánosságban megkülönböztetünk szinkron és aszinkron párhuzamos hiperparaméter-optimalizálást (lásd :numref:`distributed_scheduling`). Szinkron beállításban megvárjuk, amíg az összes párhuzamosan futó próba befejeződik, mielőtt elindítanánk a következő batch-et. Tekintsünk olyan konfigurációs tereket, amelyek olyan hiperparamétereket tartalmaznak, mint a szűrők száma vagy egy mély neurális háló rétegeinek száma. Azok a hiperparaméter-konfigurációk, amelyek több réteget vagy szűrőt tartalmaznak, természetesen tovább tartanak, és ugyanazon batch többi próbájának meg kell várnia a szinkronizációs pontokat (a :numref:`distributed_scheduling` szürke területét), mielőtt folytathatnánk az optimalizálást.

Aszinkron beállításban azonnal új próbát ütemezünk, amint erőforrás szabaddá válik. Ez optimálisan kihasználja az erőforrásainkat, mivel elkerülhető minden szinkronizációs többletköltség. A véletlen keresésnél minden új hiperparaméter-konfiguráció függetlenül kerül kiválasztásra, és különösen nem használ fel korábbi értékelések megfigyeléseit. Ez azt jelenti, hogy a véletlen keresés triviálisan párhuzamosítható aszinkron módon. Ez már nem ilyen egyszerű a kifinomultabb módszereknél, amelyek korábbi megfigyelések alapján döntenek (lásd :numref:`sec_sh_async`). Bár több erőforráshoz van szükségünk, mint szekvenciális esetben, az aszinkron véletlen keresés lineáris gyorsulást mutat: egy adott teljesítményt $K$-szor gyorsabban érünk el, ha $K$ próbát párhuzamosan tudunk futtatni. 


![A hiperparaméter-optimalizálási folyamat elosztása szinkron vagy aszinkron módon. A szekvenciális beállításhoz képest csökkenthetjük a teljes falióra-időt úgy, hogy a teljes számítási igény változatlan marad. A szinkron ütemezés tétlen dolgozókhoz vezethet a lemaradó próbák miatt.](../img/distributed_scheduling.svg)
:label:`distributed_scheduling`

Ebben a notebookban az aszinkron véletlen keresést vizsgáljuk, ahol a próbák ugyanazon a gépen több Python-folyamatban futnak. Az elosztott feladatütemezés és végrehajtás nehezen implementálható a semmiből. A *Syne Tune*-t :cite:`salinas-automl22` fogjuk használni, amely egyszerű felületet biztosít az aszinkron HPO-hoz. A Syne Tune különböző végrehajtási back-endekkel működik, és az érdeklődő olvasót arra biztatjuk, hogy tanulmányozza az egyszerű API-ját az elosztott HPO jobb megértéséhez.

```{.python .input}
from d2l import torch as d2l
import logging
logging.basicConfig(level=logging.INFO)
from syne_tune.config_space import loguniform, randint
from syne_tune.backend.python_backend import PythonBackend
from syne_tune.optimizer.baselines import RandomSearch
from syne_tune import Tuner, StoppingCriterion
from syne_tune.experiments import load_experiment
```

## Célfüggvény

Először egy új célfüggvényt kell definiálnunk úgy, hogy az a teljesítményt a `report` callbacken keresztül visszaküldje a Syne Tune-nak.

```{.python .input  n=34}
def hpo_objective_lenet_synetune(learning_rate, batch_size, max_epochs):
    from d2l import torch as d2l    
    from syne_tune import Reporter

    model = d2l.LeNet(lr=learning_rate, num_classes=10)
    trainer = d2l.HPOTrainer(max_epochs=1, num_gpus=1)
    data = d2l.FashionMNIST(batch_size=batch_size)
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
    report = Reporter() 
    for epoch in range(1, max_epochs + 1):
        if epoch == 1:
            # Inicializáljuk a Trainer állapotát
            trainer.fit(model=model, data=data) 
        else:
            trainer.fit_epoch()
        validation_error = d2l.numpy(trainer.validation_error().cpu())
        report(epoch=epoch, validation_error=float(validation_error))
```

Megjegyzendő, hogy a Syne Tune `PythonBackend` back-endje megköveteli, hogy a függőségeket a függvénydefiníción belül importáljuk.

## Aszinkron ütemező

Először meghatározzuk, hány dolgozó értékelje párhuzamosan a próbákat. Azt is meg kell adnunk, mennyi ideig szeretnénk futtatni a véletlen keresést, vagyis a teljes falióra-idő felső korlátját.

```{.python .input  n=37}
n_workers = 2  # Nem lehet nagyobb, mint az elérhető GPU-k száma

max_wallclock_time = 12 * 60  # 12 perc
```

Ezután megadjuk, melyik mérőszámot szeretnénk optimalizálni, és hogy minimalizálni vagy maximalizálni akarjuk-e azt. A `metric` értékének meg kell egyeznie a `report` callbacknek átadott argumentum nevével.

```{.python .input  n=38}
mode = "min"
metric = "validation_error"
```

Az előző példából származó konfigurációs teret használjuk. A Syne Tune-ban ezt a szótárat állandó attribútumok átadására is lehet használni a tanító szkriptnek. Ezt a lehetőséget használjuk a `max_epochs` átadására. Emellett megadjuk az elsőként értékelendő konfigurációt az `initial_config` segítségével.

```{.python .input  n=39}
config_space = {
    "learning_rate": loguniform(1e-2, 1),
    "batch_size": randint(32, 256),
    "max_epochs": 10,
}
initial_config = {
    "learning_rate": 0.1,
    "batch_size": 128,
}
```

Ezután meg kell adnunk a feladatok végrehajtási back-endjét. Itt csak a lokális gépen történő futtatást nézzük, ahol a párhuzamos feladatok alfolyamatként futnak. Nagy léptékű HPO esetén ezt klaszteren vagy felhőkörnyezetben is futtathatnánk, ahol minden próba egy teljes példányt használ.

```{.python .input  n=40}
trial_backend = PythonBackend(
    tune_function=hpo_objective_lenet_synetune,
    config_space=config_space,
)
```

Most létrehozhatjuk az aszinkron véletlen keresés ütemezőjét, amely működésében hasonló a :numref:`sec_api_hpo` szakaszban bemutatott `BasicScheduler`-hez.

```{.python .input  n=41}
scheduler = RandomSearch(
    config_space,
    metric=metric,
    mode=mode,
    points_to_evaluate=[initial_config],
)
```

A Syne Tune rendelkezik egy `Tuner`-rel is, amely központosítja a fő kísérleti ciklust és az eredmények nyilvántartását, valamint közvetíti az ütemező és a back-end közötti interakciókat.

```{.python .input  n=42}
stop_criterion = StoppingCriterion(max_wallclock_time=max_wallclock_time)

tuner = Tuner(
    trial_backend=trial_backend,
    scheduler=scheduler, 
    stop_criterion=stop_criterion,
    n_workers=n_workers,
    print_update_interval=int(max_wallclock_time * 0.6),
)
```

Futtassuk le az elosztott HPO-kísérletünket. A leállítási feltétel szerint körülbelül 12 percig fog futni.

```{.python .input  n=43}
tuner.run()
```

Az összes kiértékelt hiperparaméter-konfiguráció naplója elmentődik további elemzésre. A hangolás bármely pontján könnyen lekérhetjük az addig kapott eredményeket, és ábrázolhatjuk az incumbens pályáját.

```{.python .input  n=46}
d2l.set_figsize()
tuning_experiment = load_experiment(tuner.name)
tuning_experiment.plot()
```

## Az aszinkron optimalizálási folyamat vizualizálása

Az alábbiakban szemléltetjük, hogyan alakulnak az egyes próbák tanulási görbéi az aszinkron optimalizálási folyamat során (a plot minden színe egy próbát jelöl). Egy adott pillanatban egyszerre annyi próba fut, ahány dolgozónk van. Amint egy próba befejeződik, azonnal elindítjuk a következőt, anélkül hogy megvárnánk a többieket. Az üresjáratot az aszinkron ütemezés minimálisra csökkenti.

```{.python .input  n=45}
d2l.set_figsize([6, 2.5])
results = tuning_experiment.results

for trial_id in results.trial_id.unique():
    df = results[results["trial_id"] == trial_id]
    d2l.plt.plot(
        df["st_tuner_time"],
        df["validation_error"],
        marker="o"
    )
    
d2l.plt.xlabel("falióra-idő")
d2l.plt.ylabel("célfüggvény")
```

## Összefoglalás

Jelentősen csökkenthetjük a véletlen keresés várakozási idejét, ha a próbákat párhuzamos erőforrásokra osztjuk. Általánosságban megkülönböztetünk szinkron és aszinkron ütemezést. A szinkron ütemezés azt jelenti, hogy csak akkor mintavételezünk új hiperparaméter-konfigurációs batch-et, amikor az előző batch befejeződött. Ha vannak lemaradó próbák - vagyis olyan próbák, amelyek tovább tartanak, mint a többiek -, a dolgozóknak meg kell várniuk a szinkronizációs pontokat. Az aszinkron ütemezés viszont azonnal új hiperparaméter-konfigurációt értékel, amint erőforrás válik elérhetővé, így biztosítja, hogy minden dolgozó folyamatosan elfoglalt legyen. Bár a véletlen keresés aszinkron módon könnyen elosztható, és a tényleges algoritmuson nem igényel változtatást, más módszereknél némi további módosítás szükséges.

## Gyakorlatok

1. Tekintsük a :numref:`sec_dropout` szakaszban megvalósított `DropoutMLP` modellt, amelyet a :numref:`sec_api_hpo` 1. gyakorlatában is használtunk.
    1. Valósíts meg egy `hpo_objective_dropoutmlp_synetune` célfüggvényt a Syne Tune-nal való használathoz. Ügyelj arra, hogy a függvény minden epoch után jelezze vissza a validációs hibát.
    2. A :numref:`sec_api_hpo` 1. gyakorlatának beállítását használva hasonlítsd össze a véletlen keresést és a Bayes-optimalizálást. Ha SageMaker-t használsz, nyugodtan élj a Syne Tune benchmarking lehetőségeivel a párhuzamos kísérletek futtatásához. Tipp: a Bayes-optimalizálás `syne_tune.optimizer.baselines.BayesianOptimization` néven érhető el.
    3. Ehhez a feladathoz legalább 4 CPU-maggal rendelkező példányra van szükség. A fent használt módszerek egyikével (véletlen keresés, Bayes-optimalizálás) futtass kísérleteket `n_workers=1`, `n_workers=2`, `n_workers=4` értékekkel, és hasonlítsd össze az eredményeket (incumbens pályák). Legalább a véletlen keresés esetén lineáris skálázást kell tapasztalnod a dolgozók számához képest. Tipp: robusztus eredményekhez érdemes minden kísérletet több alkalommal megismételni.
2. *Haladó feladat*. Ennek a feladatnak a célja egy új ütemező megvalósítása a Syne Tune-ban.
    1. Hozz létre egy virtuális környezetet, amely tartalmazza mind a [d2lbook](https://github.com/d2l-ai/d2l-en/blob/master/INFO.md#installation-for-developers), mind a [syne-tune](https://syne-tune.readthedocs.io/en/latest/getting_started.html) forrásait.
    2. Valósítsd meg a :numref:`sec_api_hpo` 2. gyakorlatából származó `LocalSearcher`-t új keresőként a Syne Tune-ban. Tipp: olvasd el [ezt az útmutatót](https://syne-tune.readthedocs.io/en/latest/tutorials/developer/README.html). Alternatívaként követheted ezt a [példát](https://syne-tune.readthedocs.io/en/latest/examples.html#launch-hpo-experiment-with-home-made-scheduler) is.
    3. Hasonlítsd össze az új `LocalSearcher`-edet a `RandomSearch`-sel a `DropoutMLP` benchmarkon.


:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/12093)
:end_tab:
