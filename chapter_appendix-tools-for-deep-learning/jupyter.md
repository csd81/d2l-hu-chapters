# A Jupyter Notebookok használata
:label:`sec_jupyter`


Ez a szakasz leírja, hogyan szerkeszthetjük és futtathatjuk a könyv egyes fejezeteiben lévő kódot a Jupyter Notebook segítségével. Győződjünk meg arról, hogy telepítettük a Jupytert és letöltöttük a kódot a :ref:`chap_installation` fejezetben leírtak szerint.
Ha többet szeretnénk megtudni a Jupyterről, tekintsük meg a kiváló oktatóanyagot a [dokumentációjukban](https://jupyter.readthedocs.io/en/latest/).


## A kód helyi szerkesztése és futtatása

Tegyük fel, hogy a könyv kódjának helyi elérési útja `xx/yy/d2l-en/`. Használjuk a parancssort a könyvtárváltáshoz (`cd xx/yy/d2l-en`), majd futtassuk a `jupyter notebook` parancsot. Ha a böngészőnk nem nyitja meg automatikusan, látogassunk el a http://localhost:8888 címre, ahol megjelenik a Jupyter felülete és a könyv kódjait tartalmazó összes mappa, ahogyan azt a :numref:`fig_jupyter00` ábra mutatja.

![A könyv kódjait tartalmazó mappák.](../img/jupyter00.png)
:width:`600px`
:label:`fig_jupyter00`


A notebookfájlokat a weboldalon megjelenő mappákra kattintva érhetjük el.
Ezek általában „.ipynb" kiterjesztésűek.
Az egyszerűség kedvéért hozzunk létre egy ideiglenes „test.ipynb" fájlt.
A rá kattintás után megjelenő tartalom látható a :numref:`fig_jupyter01` ábrán.
Ez a notebook egy markdown cellát és egy kódcellát tartalmaz. A markdown cella tartalma: „This Is a Title" és „This is text.".
A kódcella két sor Python kódot tartalmaz.

![Markdown és kódcellák a „test.ipynb" fájlban.](../img/jupyter01.png)
:width:`600px`
:label:`fig_jupyter01`


Kattintsunk duplán a markdown cellára a szerkesztési módba lépéshez.
Adjuk hozzá a „Hello world." szövegláncot a cella végéhez, ahogyan azt a :numref:`fig_jupyter02` ábra mutatja.

![A markdown cella szerkesztése.](../img/jupyter02.png)
:width:`600px`
:label:`fig_jupyter02`


Ahogy a :numref:`fig_jupyter03` ábra szemlélteti,
kattintsunk a menüsorban a „Cell" $\rightarrow$ „Run Cells" lehetőségre a szerkesztett cella futtatásához.

![A cella futtatása.](../img/jupyter03.png)
:width:`600px`
:label:`fig_jupyter03`

Futtatás után a markdown cella a :numref:`fig_jupyter04` ábrán látható módon jelenik meg.

![A markdown cella futtatás után.](../img/jupyter04.png)
:width:`600px`
:label:`fig_jupyter04`


Ezután kattintsunk a kódcellára. Szorozzuk meg az elemeket 2-vel a kód utolsó sora után, ahogyan azt a :numref:`fig_jupyter05` ábra mutatja.

![A kódcella szerkesztése.](../img/jupyter05.png)
:width:`600px`
:label:`fig_jupyter05`


A cellát billentyűparanccsal is futtathatjuk (alapértelmezés szerint „Ctrl + Enter"), és a kimeneti eredményt a :numref:`fig_jupyter06` ábra szerint kapjuk meg.

![A kódcella futtatása a kimenet előállításához.](../img/jupyter06.png)
:width:`600px`
:label:`fig_jupyter06`


Ha egy notebook több cellát tartalmaz, a menüsorban a „Kernel" $\rightarrow$ „Restart & Run All" lehetőségre kattintva futtathatjuk az összes cellát. A „Help" $\rightarrow$ „Edit Keyboard Shortcuts" menüponton keresztül igény szerint szerkeszthetjük a billentyűparancsokat.

## Speciális lehetőségek

A helyi szerkesztésen túl két dolog különösen fontos: a notebookok szerkesztése markdown formátumban, valamint a Jupyter távoli futtatása.
Az utóbbi akkor lényeges, ha a kódot egy gyorsabb kiszolgálón szeretnénk futtatni.
Az előbbi azért számít, mert a Jupyter natív `.ipynb` formátuma rengeteg kiegészítő adatot tárol, amelyek nem kapcsolódnak közvetlenül a tartalomhoz. Ezek főként a kód futtatásának módjával és helyével függnek össze.
Ez megnehezíti a Git munkáját, és nagyon nehézkessé teszi a hozzájárulások áttekintését.
Szerencsére létezik alternatíva: a natív szerkesztés markdown formátumban.

### Markdown fájlok a Jupyterben

Ha hozzá szeretnénk járulni a könyv tartalmához, a forrásfájlt kell módosítanunk a GitHubon, vagyis az `.md` fájlt, nem az `.ipynb` fájlt.
A `notedown` bővítmény segítségével közvetlenül a Jupyterben is szerkeszthetjük a notebookokat markdown formátumban.


Először telepítsük a notedown bővítményt, futtassuk a Jupyter Notebookot, és töltjük be a bővítményt:

```
pip install d2l-notedown  # Lehet, hogy előbb el kell távolítanod az eredeti notedown csomagot.
jupyter notebook --NotebookApp.contents_manager_class='notedown.NotedownContentsManager'
```


A notedown bővítményt alapértelmezés szerint is bekapcsolhatjuk minden alkalommal, amikor elindítjuk a Jupyter Notebookot.
Először hozzunk létre egy Jupyter Notebook konfigurációs fájlt (ha már létrejött, kihagyhatjuk ezt a lépést).

```
jupyter notebook --generate-config
```


Ezután adjuk hozzá a következő sort a Jupyter Notebook konfigurációs fájl végéhez (Linux vagy macOS rendszeren általában a `~/.jupyter/jupyter_notebook_config.py` elérési úton):

```
c.NotebookApp.contents_manager_class = 'notedown.NotedownContentsManager'
```


Ezt követően elegendő a `jupyter notebook` parancsot futtatni ahhoz, hogy a notedown bővítmény alapértelmezés szerint bekapcsolva legyen.

### Jupyter Notebookok futtatása távoli kiszolgálón

Előfordulhat, hogy Jupyter notebookokat szeretnénk futtatni egy távoli kiszolgálón, és a helyi számítógépünk böngészőjén keresztül elérni őket. Ha Linux vagy macOS van telepítve a helyi gépünkre (Windows is támogatja ezt a funkciót harmadik féltől származó szoftverekkel, például PuTTY-val), akkor portátirányítást alkalmazhatunk:

```
ssh myserver -L 8888:localhost:8888
```


A fenti `myserver` karakterlánc a távoli kiszolgáló címe.
Ezután a http://localhost:8888 címen elérhetjük a Jupyter notebookokat futtató `myserver` távoli kiszolgálót. Ebben a függelékben később részletesen leírjuk, hogyan futtathatunk Jupyter notebookokat AWS-példányokon.

### Időmérés

Az `ExecuteTime` bővítménnyel mérhetjük a Jupyter notebookok egyes kódcelláinak végrehajtási idejét.
A bővítmény telepítéséhez futtassuk a következő parancsokat:

```
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
jupyter nbextension enable execute_time/ExecuteTime
```


## Összefoglalás

* A Jupyter Notebook eszköz segítségével szerkeszthetjük, futtathatjuk és fejleszthetjük a könyv egyes fejezeteit.
* A Jupyter notebookokat portátirányítással futtathatjuk távoli kiszolgálókon.


## Gyakorlatok

1. Szerkesszük és futtassuk a könyvben lévő kódot a Jupyter Notebookkal a helyi gépünkön.
1. Szerkesszük és futtassuk a könyvben lévő kódot a Jupyter Notebookkal *távolról*, portátirányítás segítségével.
1. Hasonlítsuk össze az $\mathbf{A}^\top \mathbf{B}$ és $\mathbf{A} \mathbf{B}$ műveletek futási idejét két, $\mathbb{R}^{1024 \times 1024}$ méretű négyzetes mátrix esetén. Melyik a gyorsabb?


[Discussions](https://discuss.d2l.ai/t/421)
