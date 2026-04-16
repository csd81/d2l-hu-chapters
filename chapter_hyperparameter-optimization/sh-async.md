```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(["pytorch"])
#required_libs("syne-tune[gpsearchers]==0.3.2")
```

# Aszinkron egymás utáni felezés

:label:`sec_sh_async`

Ahogy a :numref:`sec_rs_async` szakaszban láttuk, a HPO-t felgyorsíthatjuk a hiperparaméter-konfigurációk kiértékelésének elosztásával több példány vagy egyetlen példány több CPU-ja / GPU-ja között. Azonban a véletlen kereséssel ellentétben az egymás utáni felezés (SH) aszinkron futtatása elosztott környezetben nem magától értetődő. Mielőtt eldönthetjük, melyik konfigurációt futtassuk következőnek, először össze kell gyűjtenünk az összes megfigyelést az aktuális fokszinten. Ez megköveteli a munkások szinkronizálását minden fokszintnél. Például a legalsó $r_{\mathrm{min}}$ fokszintnél először ki kell értékelnünk mind a $N = \eta^K$ konfigurációt, mielőtt előléptethetnénk közülük $\frac{1}{\eta}$ részt a következő fokszintre.

Minden elosztott rendszerben a szinkronizáció jellemzően tétlenségi időt jelent a munkások számára. Először is, a tanítási idő a hiperparaméter-konfigurációk között nagy szórást mutat. Például, ha a rétegenként alkalmazott szűrők száma egy hiperparaméter, akkor a kevesebb szűrőt tartalmazó hálózatok gyorsabban végeznek a tanítással, mint a több szűrőt tartalmazók, ami lassú munkások miatt tétlenségi időt okoz. Ezen felül egy fokszint üres helyeinek száma nem mindig osztható a munkások számával, így előfordulhat, hogy egyes munkások egy teljes batch idejére tétlenné válnak.

A :numref:`synchronous_sh` ábra $\eta=2$ értékű szinkron SH ütemezését mutatja négy különböző próba és két munkás esetén. Az 1. epochban elkezdjük a Trial-0 és a Trial-1 kiértékelését, majd amint ezek befejeződtek, azonnal folytatjuk a következő két próbával. Meg kell várnunk, amíg a Trial-2 is befejeződik — ami lényegesen több időt vesz igénybe, mint a többi próba —, mielőtt előléptethetnénk a legjobb két próbát, azaz a Trial-0-t és a Trial-3-at a következő fokszintre. Ez tétlenségi időt okoz a Worker-1 számára. Majd folytatjuk az 1. fokkal. Itt is a Trial-3 tovább tart, mint a Trial-0, ami a Worker-0 számára további tétlenségi időt eredményez. Amint elérjük a 2. fokot, csak a legjobb próba, a Trial-0 marad, amely csupán egy munkást foglal el. Hogy a Worker-1 ne tétlenkedjen ez idő alatt, az SH legtöbb megvalósítása már a következő körrel folytatja, és új próbák (pl. Trial-4) kiértékelését kezdi el az első fokszinten.

![Szinkron egymás utáni felezés két munkással.](../img/sync_sh.svg)
:label:`synchronous_sh`

Az aszinkron egymás utáni felezés (ASHA) :cite:`li-arxiv18` az SH-t aszinkron párhuzamos forgatókönyvhöz igazítja. Az ASHA alapötlete az, hogy amint legalább $\eta$ megfigyelést gyűjtöttünk az aktuális fokszinten, azonnal előléptetjük a konfigurációkat a következő fokszintre. Ez a döntési szabály nem optimális előléptetéseket eredményezhet: olyan konfigurációk léphetnek elő a következő fokszintre, amelyek utólag visszatekintve nem teljesítenek jobban, mint az azonos fokszinten lévő többi konfiguráció. Másrészt ezzel megszabadulunk az összes szinkronizálási ponttól. A gyakorlatban ezek a nem optimális kezdeti előléptetések csak csekély hatással vannak a teljesítményre, részben azért, mert a hiperparaméter-konfigurációk sorrendje a fokszintek között általában meglehetősen konzisztens, részben pedig azért, mert a fokszintek idővel növekednek, és egyre jobban tükrözik az adott szinten mért metrikaértékek eloszlását. Ha egy munkás szabad, de egyetlen konfiguráció sem léptethető elő, egy új konfigurációt indítunk $r = r_{\mathrm{min}}$-nel, azaz az első fokszintről.

A :numref:`asha` ábra ugyanazon konfigurációk ütemezését mutatja ASHA esetén. Amint a Trial-1 befejeződik, összegyűjtjük két próba eredményeit (azaz a Trial-0 és a Trial-1 eredményeit), és azonnal előléptetjük a jobbat (Trial-0) a következő fokszintre. Miután a Trial-0 befejeződik az 1. fokon, ott még túl kevés próba szerepel ahhoz, hogy további előléptetést lehessen végrehajtani. Ezért folytatjuk a 0. fokkal, és kiértékeljük a Trial-3-at. Amint a Trial-3 befejeződik, a Trial-2 még folyamatban van. Ebben a pillanatban 3 próba lett kiértékelve a 0. fokon, és egy próba már az 1. fokon is. Mivel a Trial-3 gyengébben teljesít, mint a Trial-0 a 0. fokon, és $\eta=2$, egyelőre egyetlen új próbát sem tudunk előléptetni, és a Worker-1 inkább a Trial-4-et kezdi el az elejéről. Amikor azonban a Trial-2 befejeződik, és gyengébben teljesít, mint a Trial-3, az utóbbit előléptetjük az 1. fokra. Ezt követően 2 kiértékelés gyűlt össze az 1. fokon, ami azt jelenti, hogy a Trial-0-t most már elő tudjuk léptetni a 2. fokra. Egyidejűleg a Worker-1 folytatja az új próbák (azaz a Trial-5) kiértékelését a 0. fokon.


![Aszinkron egymás utáni felezés (ASHA) két munkással.](../img/asha.svg)
:label:`asha`

```{.python .input}
from d2l import torch as d2l
import logging
logging.basicConfig(level=logging.INFO)
import matplotlib.pyplot as plt
from syne_tune.config_space import loguniform, randint
from syne_tune.backend.python_backend import PythonBackend
from syne_tune.optimizer.baselines import ASHA
from syne_tune import Tuner, StoppingCriterion
from syne_tune.experiments import load_experiment
```

## Célfüggvény

A *Syne Tune* keretrendszert ugyanazzal a célfüggvénnyel fogjuk használni, mint a
:numref:`sec_rs_async` szakaszban.

```{.python .input  n=54}
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

Ugyanazt a konfigurációs teret fogjuk használni, mint korábban:

```{.python .input  n=55}
min_number_of_epochs = 2
max_number_of_epochs = 10
eta = 2

config_space = {
    "learning_rate": loguniform(1e-2, 1),
    "batch_size": randint(32, 256),
    "max_epochs": max_number_of_epochs,
}
initial_config = {
    "learning_rate": 0.1,
    "batch_size": 128,
}
```

## Aszinkron ütemező

Először megadjuk az egyszerre próbákat kiértékelő munkások számát. Azt is meg kell határoznunk, mennyi ideig szeretnénk futtatni a véletlen keresést, ehhez felső korlátot állítunk be a teljes falióra-időre.

```{.python .input  n=56}
n_workers = 2  # Nem lehet nagyobb, mint az elérhető GPU-k száma
max_wallclock_time = 12 * 60  # 12 perc
```

Az ASHA futtatásához szükséges kód az aszinkron véletlen kereséshez képest csupán egyszerű módosítást igényel.

```{.python .input  n=56}
mode = "min"
metric = "validation_error"
resource_attr = "epoch"

scheduler = ASHA(
    config_space,
    metric=metric,
    mode=mode,
    points_to_evaluate=[initial_config],
    max_resource_attr="max_epochs",
    resource_attr=resource_attr,
    grace_period=min_number_of_epochs,
    reduction_factor=eta,
)
```

Itt a `metric` és a `resource_attr` a `report` visszahívásnál használt kulcsneveket adja meg, a `max_resource_attr` pedig azt jelzi, hogy a célfüggvény melyik bemenete felel meg $r_{\mathrm{max}}$-nak. Emellett a `grace_period` adja meg $r_{\mathrm{min}}$-t, a `reduction_factor` pedig $\eta$ értékét. A Syne Tune-t a korábbiak szerint futtathatjuk (ez körülbelül 12 percet vesz igénybe):

```{.python .input  n=57}
trial_backend = PythonBackend(
    tune_function=hpo_objective_lenet_synetune,
    config_space=config_space,
)

stop_criterion = StoppingCriterion(max_wallclock_time=max_wallclock_time)
tuner = Tuner(
    trial_backend=trial_backend,
    scheduler=scheduler,
    stop_criterion=stop_criterion,
    n_workers=n_workers,
    print_update_interval=int(max_wallclock_time * 0.6),
)
tuner.run()
```

Megjegyzendő, hogy az ASHA egy olyan változatát futtatjuk, ahol az alulteljesítő próbákat korán leállítjuk. Ez eltér a :numref:`sec_mf_hpo_sh` szakaszban megvalósított verziótól, ahol minden tanítási feladatot rögzített `max_epochs` értékkel indítunk. Az utóbbi esetben egy jól teljesítő próba, amely eléri a teljes 10 epochot, először 1, majd 2, majd 4, majd 8 epochig tanul, minden alkalommal az elejéről kezdve. Az ilyen típusú szünet-és-folytatás ütemezés hatékonyan megvalósítható az egyes epochok utáni tanítási állapot elmentésével (checkpointing), de itt elkerüljük ezt a plusz bonyolultságot. A kísérlet befejezése után lekérhetjük és ábrázolhatjuk az eredményeket.

```{.python .input  n=59}
d2l.set_figsize()
e = load_experiment(tuner.name)
e.plot()
```

## Az optimalizálási folyamat vizualizálása

Ismét megjelenítjük az összes próba tanulási görbéit (az ábra minden színe egy-egy próbát jelöl). Hasonlítsuk ezt össze az aszinkron véletlen kereséssel a :numref:`sec_rs_async` szakaszban. Ahogy a :numref:`sec_mf_hpo` szakaszban az egymás utáni felezésnél is láttuk, a próbák nagy része 1 vagy 2 epochnál leáll ($r_{\mathrm{min}}$ vagy $\eta * r_{\mathrm{min}}$ esetén). A próbák azonban nem ugyanazon a ponton állnak le, mert epochonként különböző mennyiségű időt igényelnek. Ha az ASHA helyett standard egymás utáni felezést alkalmaznánk, szinkronizálnunk kellene a munkásokat, mielőtt a konfigurációkat a következő fokszintre léptethetnénk.

```{.python .input  n=60}
d2l.set_figsize([6, 2.5])
results = e.results
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

A véletlen kereséshez képest az egymás utáni felezés nem olyan egyszerűen futtatható aszinkron elosztott környezetben. A szinkronizálási pontok elkerülése érdekében a konfigurációkat a lehető leggyorsabban előléptetjük a következő fokszintre, még akkor is, ha ez egyes nem megfelelő konfigurációk előléptetését jelenti. A gyakorlatban ez általában nem okoz nagy visszaesést, és az aszinkron kontra szinkron ütemezés nyeresége rendszerint jóval nagyobb, mint a nem optimális döntéshozatal miatti veszteség.


:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/12101)
:end_tab:
