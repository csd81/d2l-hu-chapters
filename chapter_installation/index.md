# Telepítés
:label:`chap_installation`

Ahhoz, hogy el tudjunk indulni, szükségünk lesz egy környezetre a Python,
a Jupyter Notebook, a megfelelő könyvtárak és a könyv futtatásához
szükséges kód számára.

## Miniconda telepítése

A legegyszerűbb megoldás a
[Miniconda](https://conda.io/en/latest/miniconda.html) telepítése.
Fontos, hogy Python 3.x verzióra lesz szükséged. Ha a gépeden már telepítve
van a conda, a következő lépéseket kihagyhatod.

Látogass el a Miniconda weboldalára, és válaszd ki a rendszerednek
megfelelő verziót a Python 3.x verziód és a géped architektúrája alapján.
Tegyük fel, hogy a Python-verziód 3.9
(ezt a verziót teszteltük). Ha macOS-t használsz, akkor azt a bash
szkriptet töltsd le, amelynek nevében szerepel a "MacOSX" szöveg, lépj a
letöltési helyre, és futtasd a telepítést az alábbi módon
(Intel-alapú Mac példáján):

```bash
# A fájlnév változhat
sh Miniconda3-py39_4.12.0-MacOSX-x86_64.sh -b
```


Linuxon azt a fájlt kell letöltened, amelynek nevében szerepel a
"Linux" szó, majd a letöltési helyen futtasd a következőt:

```bash
# A fájlnév változhat
sh Miniconda3-py39_4.12.0-Linux-x86_64.sh -b
```


Windows alatt a Minicondát az
[online útmutatója](https://conda.io/en/latest/miniconda.html) alapján
töltheted le és telepítheted. Windowsban a parancsok futtatásához keresd
meg a `cmd` programot, amellyel megnyithatod a Parancssort.

Ezután inicializáld a shellt, hogy közvetlenül futtathasd a `conda`
parancsot.

```bash
~/miniconda3/bin/conda init
```


Ezután zárd be, majd nyisd meg újra a jelenlegi shelledet.
Most már létre tudsz hozni egy új környezetet:

```bash
conda create --name d2l python=3.9 -y
```


Most aktiválhatjuk a `d2l` környezetet:

```bash
conda activate d2l
```


## A deep learning keretrendszer és a `d2l` csomag telepítése

Mielőtt bármilyen deep learning keretrendszert telepítenél, először
ellenőrizd, hogy van-e megfelelő GPU a gépedben
(egy átlagos laptop kijelzőjét meghajtó GPU a mi céljainkra általában nem
elég). Ha például a számítógépedben NVIDIA GPU van, és telepítetted a
[CUDA-t](https://developer.nvidia.com/cuda-downloads), akkor jó eséllyel
minden készen áll. Ha a gépedben nincs GPU, egyelőre nem kell aggódnod:
az első néhány fejezethez a CPU teljesítménye is bőven elegendő. Arra
viszont később számíts, hogy nagyobb modellek futtatásához jól jön majd a
GPU-hozzáférés.


:begin_tab:`mxnet`

Az MXNet GPU-támogatású változatának telepítéséhez először meg kell
néznünk, melyik CUDA-verzió van telepítve. Ezt az `nvcc --version` vagy a
`cat /usr/local/cuda/version.txt` parancs futtatásával ellenőrizheted.
Tegyük fel, hogy a CUDA 11.2 van telepítve; ekkor a következő parancsot
kell végrehajtanod:

```bash
# macOS- és Linux-felhasználóknak
pip install mxnet-cu112==1.9.1

# Windows-felhasználóknak
pip install mxnet-cu112==1.9.1 -f https://dist.mxnet.io/python
```


A végződést a CUDA-verziódnak megfelelően módosíthatod, például `cu101`
CUDA 10.1-hez, illetve `cu90` CUDA 9.0-hoz.


Ha a gépedben nincs NVIDIA GPU vagy CUDA, akkor a CPU-s változatot a
következőképpen telepítheted:

```bash
pip install mxnet==1.9.1
```


:end_tab:


:begin_tab:`pytorch`

A PyTorchot (az alább megadott verziókat az írás idején teszteltük)
CPU-val vagy GPU-val a következőképpen telepítheted:

```bash
pip install torch==2.0.0 torchvision==0.15.1
```


:end_tab:

:begin_tab:`tensorflow`
A TensorFlow-t CPU-s vagy GPU-s támogatással egyaránt a következőképpen
telepítheted:

```bash
pip install tensorflow==2.12.0 tensorflow-probability==0.20.0
```


:end_tab:

:begin_tab:`jax`
A JAX-ot és a Flax-ot CPU-val vagy GPU-val a következőképpen telepítheted:

```bash
# GPU
pip install "jax[cuda11_pip]==0.4.13" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html flax==0.7.0
```


Ha a gépedben nincs NVIDIA GPU vagy CUDA, akkor a CPU-s változatot a
következőképpen telepítheted:

```bash
# CPU
pip install "jax[cpu]==0.4.13" flax==0.7.0
```


:end_tab:


A következő lépés a saját fejlesztésű `d2l` csomag telepítése, amelybe a
könyv során gyakran használt függvényeket és osztályokat gyűjtöttük össze:

```bash
pip install d2l==1.0.3
```


## A kód letöltése és futtatása

Ezután érdemes letöltened a notebookokat, hogy a könyv minden
kódrészletét futtatni tudd. Egyszerűen kattints a "Notebooks" fülre a
[D2L.ai weboldal](https://d2l.ai/) bármely HTML-oldalának tetején, töltsd
le a kódot, majd csomagold ki. Alternatív megoldásként a notebookokat
parancssorból is letöltheted:

:begin_tab:`mxnet`

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en-1.0.3.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
cd mxnet
```


:end_tab:


:begin_tab:`pytorch`

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en-1.0.3.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
cd pytorch
```


:end_tab:

:begin_tab:`tensorflow`

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en-1.0.3.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
cd tensorflow
```


:end_tab:

:begin_tab:`jax`

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en-1.0.3.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
cd jax
```


:end_tab:

Ha még nincs telepítve az `unzip`, először futtasd a
`sudo apt-get install unzip` parancsot. Ezután a következőképpen
indíthatjuk el a Jupyter Notebook szervert:

```bash
jupyter notebook
```


Ezen a ponton megnyithatod a http://localhost:8888 címet a böngésződben
(lehet, hogy már automatikusan meg is nyílt). Ezután futtathatod a könyv
egyes részeinek kódját. Valahányszor új parancssori ablakot nyitsz, a
D2L notebookok futtatása vagy a csomagok frissítése előtt
(akár a deep learning keretrendszert, akár a `d2l` csomagot frissíted)
futtasd a `conda activate d2l` parancsot a futtatási környezet
aktiválásához.
A környezetből a `conda deactivate` paranccsal léphetsz ki.


:begin_tab:`mxnet`
[Beszélgetések](https://discuss.d2l.ai/t/23)
:end_tab:

:begin_tab:`pytorch`
[Beszélgetések](https://discuss.d2l.ai/t/24)
:end_tab:

:begin_tab:`tensorflow`
[Beszélgetések](https://discuss.d2l.ai/t/436)
:end_tab:

:begin_tab:`jax`
[Beszélgetések](https://discuss.d2l.ai/t/17964)
:end_tab:
