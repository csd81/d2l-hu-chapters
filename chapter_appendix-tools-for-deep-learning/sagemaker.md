# Az Amazon SageMaker használata
:label:`sec_sagemaker`

A deep learning alkalmazások
olyan nagy számítási erőforrást igényelhetnek,
amely könnyen meghaladhatja
a helyi gépünk kapacitását.
A felhőalapú számítástechnikai szolgáltatások
lehetővé teszik, hogy
a könyv GPU-igényes kódját
egyszerűbben futtassuk
nagyobb teljesítményű számítógépek segítségével.
Ez a szakasz bemutatja,
hogyan használható az Amazon SageMaker
a könyv kódjának futtatásához.

## Regisztráció

Először létre kell hoznunk egy fiókot a https://aws.amazon.com/ oldalon.
A fokozott biztonság érdekében
javasolt a kétfaktoros hitelesítés használata.
Érdemes továbbá részletes számlázási és költési riasztásokat beállítani,
hogy elkerüljük a kellemetlen meglepetéseket,
például ha elfelejtjük leállítani a futó példányokat.
Az AWS-fiókba való bejelentkezés után
menjünk a [konzolunkhoz](http://console.aws.amazon.com/) és keressük az "Amazon SageMaker" lehetőséget (lásd: :numref:`fig_sagemaker`),
majd kattintsunk rá a SageMaker panel megnyitásához.

![A SageMaker panel megkeresése és megnyitása.](../img/sagemaker.png)
:width:`300px`
:label:`fig_sagemaker`

## SageMaker-példány létrehozása

Ezután hozzunk létre egy notebook-példányt a :numref:`fig_sagemaker-create` ábrán leírtak szerint.

![SageMaker-példány létrehozása.](../img/sagemaker-create.png)
:width:`400px`
:label:`fig_sagemaker-create`

A SageMaker számos [példánytípust](https://aws.amazon.com/sagemaker/pricing/instance-types/) kínál különböző számítási teljesítménnyel és árakkal.
Notebook-példány létrehozásakor
megadhatjuk a nevét és típusát.
A :numref:`fig_sagemaker-create-2` ábrán az `ml.p3.2xlarge` típust választjuk: egy Tesla V100 GPU-val és 8 magos CPU-val ez a példány elég erős a könyv nagy részéhez.

![A példánytípus kiválasztása.](../img/sagemaker-create-2.png)
:width:`400px`
:label:`fig_sagemaker-create-2`

:begin_tab:`mxnet`
A teljes könyv ipynb formátumban, SageMakerrel való futtatáshoz elérhető a https://github.com/d2l-ai/d2l-en-sagemaker oldalon. Megadhatjuk ennek a GitHub-tárolónak az URL-jét (:numref:`fig_sagemaker-create-3`), hogy a SageMaker klónozza azt a példány létrehozásakor.
:end_tab:

:begin_tab:`pytorch`
A teljes könyv ipynb formátumban, SageMakerrel való futtatáshoz elérhető a https://github.com/d2l-ai/d2l-pytorch-sagemaker oldalon. Megadhatjuk ennek a GitHub-tárolónak az URL-jét (:numref:`fig_sagemaker-create-3`), hogy a SageMaker klónozza azt a példány létrehozásakor.
:end_tab:

:begin_tab:`tensorflow`
A teljes könyv ipynb formátumban, SageMakerrel való futtatáshoz elérhető a https://github.com/d2l-ai/d2l-tensorflow-sagemaker oldalon. Megadhatjuk ennek a GitHub-tárolónak az URL-jét (:numref:`fig_sagemaker-create-3`), hogy a SageMaker klónozza azt a példány létrehozásakor.
:end_tab:

![A GitHub-tároló megadása.](../img/sagemaker-create-3.png)
:width:`400px`
:label:`fig_sagemaker-create-3`

## Példány futtatása és leállítása

A példány létrehozása
néhány percet vehet igénybe.
Ha elkészült,
kattintsunk a mellette lévő "Open Jupyter" hivatkozásra (:numref:`fig_sagemaker-open`), hogy
szerkeszthessük és futtathassuk a könyv összes Jupyter notebookját
ezen a példányon
(a :numref:`sec_jupyter` lépéseihez hasonlóan).

![A Jupyter megnyitása a létrehozott SageMaker-példányon.](../img/sagemaker-open.png)
:width:`400px`
:label:`fig_sagemaker-open`


A munka befejezése után
ne felejtsük el leállítani a példányt,
hogy elkerüljük a további díjakat (:numref:`fig_sagemaker-stop`).

![SageMaker-példány leállítása.](../img/sagemaker-stop.png)
:width:`300px`
:label:`fig_sagemaker-stop`

## Notebookok frissítése

:begin_tab:`mxnet`
Ennek a nyílt forráskódú könyvnek a notebookjai rendszeresen frissülnek a GitHubon található [d2l-ai/d2l-en-sagemaker](https://github.com/d2l-ai/d2l-en-sagemaker) tárolóban.
A legújabb verzióra való frissítéshez
nyissunk meg egy terminált a SageMaker-példányon (:numref:`fig_sagemaker-terminal`).
:end_tab:

:begin_tab:`pytorch`
Ennek a nyílt forráskódú könyvnek a notebookjai rendszeresen frissülnek a GitHubon található [d2l-ai/d2l-pytorch-sagemaker](https://github.com/d2l-ai/d2l-pytorch-sagemaker) tárolóban.
A legújabb verzióra való frissítéshez
nyissunk meg egy terminált a SageMaker-példányon (:numref:`fig_sagemaker-terminal`).
:end_tab:


:begin_tab:`tensorflow`
Ennek a nyílt forráskódú könyvnek a notebookjai rendszeresen frissülnek a GitHubon található [d2l-ai/d2l-tensorflow-sagemaker](https://github.com/d2l-ai/d2l-tensorflow-sagemaker) tárolóban.
A legújabb verzióra való frissítéshez
nyissunk meg egy terminált a SageMaker-példányon (:numref:`fig_sagemaker-terminal`).
:end_tab:


![Terminál megnyitása a SageMaker-példányon.](../img/sagemaker-terminal.png)
:width:`300px`
:label:`fig_sagemaker-terminal`

Érdemes lehet véglegesíteni a helyi módosításainkat, mielőtt lekérjük a frissítéseket a távoli tárolóból.
Ellenkező esetben egyszerűen töröljük az összes helyi módosítást
a terminálban a következő parancsokkal:

:begin_tab:`mxnet`

```bash
cd SageMaker/d2l-en-sagemaker/
git reset --hard
git pull
```


:end_tab:

:begin_tab:`pytorch`

```bash
cd SageMaker/d2l-pytorch-sagemaker/
git reset --hard
git pull
```


:end_tab:

:begin_tab:`tensorflow`

```bash
cd SageMaker/d2l-tensorflow-sagemaker/
git reset --hard
git pull
```


:end_tab:

## Összefoglalás

* Az Amazon SageMaker segítségével notebook-példányt hozhatunk létre a könyv GPU-igényes kódjának futtatásához.
* A notebookokat az Amazon SageMaker-példányon lévő terminálon keresztül frissíthetjük.


## Gyakorlatok


1. Szerkesszük és futtassuk bármelyik GPU-t igénylő fejezetet az Amazon SageMaker segítségével.
1. Nyissunk meg egy terminált a könyv összes notebookját tároló helyi könyvtár eléréséhez.


[Discussions](https://discuss.d2l.ai/t/422)
