# AWS EC2-példányok használata
:label:`sec_aws`

Ebben a szakaszban bemutatjuk, hogyan telepíthetjük az összes könyvtárat egy csupasz Linux rendszerre. Korábban, a :numref:`sec_sagemaker` szakaszban tárgyaltuk az Amazon SageMaker használatát, míg egy saját példány felépítése kevesebbe kerül az AWS-en. Az útmutató három lépésből áll:

1. GPU-val rendelkező Linux-példány indítása az AWS EC2-ben.
1. A CUDA telepítése (vagy előre telepített CUDA-val rendelkező Amazon Machine Image használata).
1. A mélytanulás keretrendszer és a könyv kódjának futtatásához szükséges egyéb könyvtárak telepítése.

Ez a folyamat más példányokra (és más felhőkre) is alkalmazható, néhány kisebb módosítással. A továbblépés előtt AWS-fiókot kell létrehoznunk; további részletekért lásd a :numref:`sec_sagemaker` szakaszt.


## EC2-példány létrehozása és futtatása

Az AWS-fiókba való bejelentkezés után kattintsunk az "EC2" gombra (:numref:`fig_aws`) az EC2 panel megnyitásához.

![Az EC2 konzol megnyitása.](../img/aws.png)
:width:`400px`
:label:`fig_aws`

A :numref:`fig_ec2` ábra az EC2 panelt mutatja.

![Az EC2 panel.](../img/ec2.png)
:width:`700px`
:label:`fig_ec2`

### Régió kiválasztása
Válasszunk egy közeli adatközpontot a késleltetés csökkentése érdekében, például "Oregon" (a :numref:`fig_ec2` ábra jobb felső sarkában piros kerettel jelölve). Ha Kínában tartózkodunk,
egy közeli ázsiai-csendes-óceáni régiót választhatunk, például Szöult vagy Tokiót. Felhívjuk a figyelmet,
hogy egyes adatközpontokban nem feltétlenül érhetők el GPU-s példányok.


### Korlátok növelése

A példány kiválasztása előtt ellenőrizzük, hogy vannak-e mennyiségi
korlátozások a bal oldali sávban látható "Limits" elemre kattintva, ahogy a
:numref:`fig_ec2` ábrán látható.
A :numref:`fig_limits` ábra egy ilyen korlátozásra mutat példát. A fiók jelenleg nem nyithat meg "p2.xlarge" példányokat az adott régióban. Ha
szükségünk van egy vagy több példány megnyitására, kattintsunk a "Request limit increase" hivatkozásra,
hogy magasabb példánykvótát igényeljünk.
Az igény elbírálása általában egy munkanapot vesz igénybe.

![Példánymennyiség-korlátozások.](../img/limits.png)
:width:`700px`
:label:`fig_limits`


### Példány indítása

Ezután kattintsunk a :numref:`fig_ec2` ábrán piros kerettel jelölt "Launch Instance" gombra a példány indításához.

Először válasszunk egy megfelelő Amazon Machine Image-et (AMI). Válasszunk egy Ubuntu példányt (:numref:`fig_ubuntu`).


![AMI kiválasztása.](../img/ubuntu-new.png)
:width:`700px`
:label:`fig_ubuntu`

Az EC2 számos különböző példánykonfigurációt kínál. Ez kezdőknek néha megterhelőnek tűnhet. A :numref:`tab_ec2` táblázat felsorolja a különböző alkalmas gépeket.

:Különböző EC2-példánytípusok
:label:`tab_ec2`

| Név  | GPU         | Megjegyzések                          |
|------|-------------|---------------------------------------|
| g2   | Grid K520   | elavult                               |
| p2   | Kepler K80  | régi, de spot-ként gyakran olcsó      |
| g3   | Maxwell M60 | jó kompromisszum                      |
| p3   | Volta V100  | magas teljesítmény FP16-hoz           |
| p4   | Ampere A100 | magas teljesítmény nagyléptékű betanításhoz |
| g4   | Turing T4   | következtetésre optimalizált FP16/INT8 |


Ezek a kiszolgálók több változatban kaphatók, amelyek a felhasznált GPU-k számát jelzik. Például egy p2.xlarge 1 GPU-val rendelkezik, a p2.16xlarge pedig 16 GPU-val és több memóriával. További részletekért lásd az [AWS EC2 dokumentációját](https://aws.amazon.com/ec2/instance-types/) vagy egy [összefoglaló oldalt](https://www.ec2instances.info). A szemléltetés céljára elegendő egy p2.xlarge (a :numref:`fig_p2x` ábrán piros kerettel jelölve).

![Példány kiválasztása.](../img/p2x.png)
:width:`700px`
:label:`fig_p2x`

Fontos megjegyezni, hogy GPU-képes példányt kell használni, megfelelő illesztőprogramokkal és GPU-képes mélytanulás keretrendszerrel. Ellenkező esetben nem élvezhetjük a GPU-k nyújtotta előnyöket.

Ezután ki kell választanunk a példány eléréséhez használt kulcspárt. Ha nincs kulcspárunk, kattintsunk a "Create new key pair" gombra a :numref:`fig_keypair` ábrán, hogy kulcspárt hozzunk létre. Ezt követően
kiválaszthatjuk a korábban létrehozott kulcspárt.
Győződjünk meg arról, hogy letöltjük a kulcspárt, és biztonságos helyen tároljuk,
ha újat hoztunk létre. Ez az egyetlen módja a kiszolgálóra való SSH-bejelentkezésnek.

![Kulcspár kiválasztása.](../img/keypair.png)
:width:`500px`
:label:`fig_keypair`

Ebben a példában az alapértelmezett konfigurációkat tartjuk meg a "Network settings" esetében (az "Edit" gombra kattintva konfigurálhatunk olyan elemeket, mint az alhálózat és a biztonsági csoportok). Csak az alapértelmezett merevlemez méretét növeljük 64 GB-ra (:numref:`fig_disk`). Megjegyezzük, hogy maga a CUDA már 4 GB-ot foglal.

![A merevlemez méretének módosítása.](../img/disk.png)
:width:`700px`
:label:`fig_disk`


Kattintsunk a "Launch Instance" gombra a létrehozott példány elindításához.
Kattintsunk a :numref:`fig_launching` ábrán látható példányazonosítóra a példány állapotának megtekintéséhez.

![Kattintás a példányazonosítóra.](../img/launching.png)
:width:`700px`
:label:`fig_launching`

### Csatlakozás a példányhoz

Ahogy a :numref:`fig_connect` ábra mutatja, miután a példány állapota zöldre vált, jobb egérgombbal kattintsunk a példányra, és válasszuk a `Connect` lehetőséget a példány elérési módjának megtekintéséhez.

![A példány elérési módjának megtekintése.](../img/connect.png)
:width:`700px`
:label:`fig_connect`

Ha új kulcspárt használunk, az SSH csak akkor működik, ha a kulcsfájl nem nyilvánosan hozzáférhető. Menjünk a `D2L_key.pem` fájlt tároló mappába, és
hajtsuk végre a következő parancsot,
hogy a kulcs ne legyen nyilvánosan olvasható:

```bash
chmod 400 D2L_key.pem
```


![A példány elérési és indítási módjának megtekintése.](../img/chmod.png)
:width:`400px`
:label:`fig_chmod`


Most másoljuk ki az SSH parancsot a :numref:`fig_chmod` ábra alsó piros keretéből, és illesszük be a parancssorba:

```bash
ssh -i "D2L_key.pem" ubuntu@ec2-xx-xxx-xxx-xxx.y.compute.amazonaws.com
```


Amikor a parancssor megkérdezi: "Are you sure you want to continue connecting (yes/no)", írjuk be, hogy "yes", majd nyomjuk meg az Entert a példányba való bejelentkezéshez.

A kiszolgáló most már használatra kész.


## A CUDA telepítése

A CUDA telepítése előtt feltétlenül frissítsük a példányt a legújabb illesztőprogramokkal.

```bash
sudo apt-get update && sudo apt-get install -y build-essential git libgfortran3
```


Töltsük le a CUDA 12.1-et. A letöltési hivatkozáshoz látogassunk el az NVIDIA [hivatalos tárolójába](https://developer.nvidia.com/cuda-toolkit-archive), ahogy azt a :numref:`fig_cuda` ábra mutatja.

![A CUDA 12.1 letöltési címének megkeresése.](../img/cuda121.png)
:width:`500px`
:label:`fig_cuda`

Másoljuk ki az utasításokat, és illesszük be a terminálba a CUDA 12.1 telepítéséhez.

```bash
# A hivatkozás és a fájlnév változhat
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```


A program telepítése után futtassuk a következő parancsot a GPU-k megtekintéséhez:

```bash
nvidia-smi
```


Végül adjuk hozzá a CUDA-t a könyvtárak keresési útvonalához, hogy más könyvtárak is megtalálják, például a következő sorok hozzáfűzésével a `~/.bashrc` fájl végéhez.

```bash
export PATH="/usr/local/cuda-12.1/bin:$PATH"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-12.1/lib64
```


## A kód futtatásához szükséges könyvtárak telepítése

A könyv kódjának futtatásához
egyszerűen kövessük a :ref:`chap_installation` lépéseit
Linux felhasználókként az EC2-példányon,
és használjuk a következő tippeket
a távoli Linux kiszolgálón való munkához:

* A Miniconda telepítési oldalán lévő bash szkript letöltéséhez kattintsunk jobb gombbal a letöltési hivatkozásra, és válasszuk a "Copy Link Address" lehetőséget, majd hajtsuk végre a `wget [másolt hivatkozás]` parancsot.
* A `~/miniconda3/bin/conda init` futtatása után hajthatjuk végre a `source ~/.bashrc` parancsot az aktuális shell bezárása és újranyitása helyett.


## A Jupyter Notebook távoli futtatása

A Jupyter Notebook távoli futtatásához SSH-portátirányítást kell alkalmazni. Elvégre a felhőben lévő kiszolgálónak nincs monitorja vagy billentyűzete. Ehhez jelentkezzünk be a kiszolgálóra az asztali számítógépünkről (vagy laptopunkról) a következőképpen:

```
# Ezt a parancsot a helyi parancssorban kell futtatni
ssh -i "/path/to/key.pem" ubuntu@ec2-xx-xxx-xxx-xxx.y.compute.amazonaws.com -L 8889:localhost:8888
```


Ezután menjünk a letöltött könyvkód helyére
az EC2-példányon,
majd futtassuk:

```
conda activate d2l
jupyter notebook
```


A :numref:`fig_jupyter` ábra a Jupyter Notebook futtatása után megjelenő lehetséges kimenetet mutatja. Az utolsó sor a 8888-as port URL-je.

![A Jupyter Notebook futtatása utáni kimenet. Az utolsó sor a 8888-as port URL-je.](../img/jupyter.png)
:width:`700px`
:label:`fig_jupyter`

Mivel a 8889-es portra alkalmaztunk portátirányítást,
másoljuk ki az utolsó sort a :numref:`fig_jupyter` ábra piros keretéből,
cseréljük le az URL-ben a "8888"-at "8889"-re,
és nyissuk meg a helyi böngészőnkben.


## Nem használt példányok bezárása

Mivel a felhőszolgáltatások használati idő alapján kerülnek számlázásra, zárjuk be a nem használt példányokat. Megjegyezzük, hogy vannak alternatívák:

* Egy példány „leállítása" azt jelenti, hogy újra el tudjuk indítani. Ez hasonló egy hagyományos kiszolgáló kikapcsolásához. A leállított példányok után azonban továbbra is kis összegű díjat kell fizetni a megőrzött merevlemez-területért.
* Egy példány „megszüntetése" törli az összes hozzá tartozó adatot. Ez magában foglalja a lemezt is, ezért nem indítható el újra. Ezt csak akkor tegyük, ha biztosak vagyunk abban, hogy a jövőben nem lesz rá szükségünk.

Ha a példányt számos további példány sablonjaként szeretnénk használni,
kattintsunk jobb gombbal a :numref:`fig_connect` ábrán látható példányra, és válasszuk az "Image" $\rightarrow$
"Create" lehetőséget a példány képének létrehozásához. Ha ez kész, válasszuk az
"Instance State" $\rightarrow$ "Terminate" lehetőséget a példány megszüntetéséhez. Legközelebb,
ha ezt a példányt szeretnénk használni, követhetjük a szakasz lépéseit
egy, a mentett képen alapuló példány létrehozásához. Az egyetlen különbség az, hogy a
:numref:`fig_ubuntu` ábrán látható "1. Choose AMI" lépésnél
a bal oldalon a "My AMIs" lehetőséget kell használni a mentett kép kiválasztásához. A létrehozott példány megőrzi a kép merevlemezén tárolt információkat. Például nem kell újratelepíteni a CUDA-t és más futtatókörnyezeteket.


## Összefoglalás

* Igény szerint indíthatunk és állíthatunk le példányokat anélkül, hogy saját számítógépet kellene vásárolnunk és összeállítanunk.
* A GPU-képes mélytanulás keretrendszer használata előtt telepíteni kell a CUDA-t.
* Portátirányítással futtathatjuk a Jupyter Notebookot egy távoli kiszolgálón.


## Gyakorlatok

1. A felhő kényelmet nyújt, de nem olcsó. Nézzük meg, hogyan indíthatunk [spot példányokat](https://aws.amazon.com/ec2/spot/) a költségek csökkentése érdekében.
1. Kísérletezzünk különböző GPU-s kiszolgálókkal. Milyen gyorsak?
1. Kísérletezzünk több GPU-s kiszolgálókkal. Mennyire jól skálázhatók?


[Discussions](https://discuss.d2l.ai/t/423)
