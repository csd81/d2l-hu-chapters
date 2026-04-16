# Markov-döntési folyamat (MDP)
:label:`sec_mdp`
Ebben a szakaszban megvitatjuk, hogyan lehet a megerősítéses tanulási problémákat Markov-döntési folyamatok (MDP-k) segítségével megfogalmazni, és részletesen leírjuk az MDP-k különböző összetevőit.

## Az MDP definíciója

A Markov-döntési folyamat (MDP) :cite:`BellmanMDP` egy modell arra vonatkozóan, hogy egy rendszer állapota hogyan változik, amikor különböző akciókat alkalmaznak rá. Néhány különböző mennyiség együttesen alkotja az MDP-t.

![Egy egyszerű rácsvilág-navigációs feladat, ahol a robotnak nemcsak meg kell találnia az utat a célállomáshoz (zöld házzal jelölve), hanem el kell kerülnie a csapdahelyzeteket is (piros keresztjelekkel jelölve).](../img/mdp.png)
:width:`250px`
:label:`fig_mdp`

* Legyen $\mathcal{S}$ az MDP-ben lévő állapotok halmaza. Konkrét példaként lásd :numref:`fig_mdp`, egy olyan robot esetén, amely egy rácsvilágban navigál. Ebben az esetben $\mathcal{S}$ megfelel azon helyszínek halmazának, ahol a robot bármely adott időlépésben tartózkodhat.
* Legyen $\mathcal{A}$ azon akciók halmaza, amelyeket a robot minden állapotban végrehajthat, pl. „menj előre", „fordulj jobbra", „fordulj balra", „maradj ugyanazon a helyen" stb. Az akciók megváltoztathatják a robot jelenlegi állapotát valamely más állapotra az $\mathcal{S}$ halmazon belül.
* Előfordulhat, hogy nem tudjuk *pontosan*, hogyan mozog a robot, hanem csak valamilyen közelítésig ismerjük. Ezt a helyzetet a megerősítéses tanulásban a következőképpen modellezzük: ha a robot végrehajtja a „menj előre" akciót, kis valószínűséggel megmaradhat a jelenlegi állapotban, kis valószínűséggel „balra fordulhat" stb. Matematikailag ez azt jelenti, hogy definiálunk egy „átmeneti függvényt" $T: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \to [0,1]$, amelyre $T(s, a, s') = P(s' \mid s, a)$ az $s'$ állapot elérésének feltételes valószínűsége, feltéve hogy a robot az $s$ állapotban volt és az $a$ akciót hajtotta végre. Az átmeneti függvény valószínűségeloszlás, ezért $\sum_{s' \in \mathcal{S}} T(s, a, s') = 1$ minden $s \in \mathcal{S}$ és $a \in \mathcal{A}$ esetén, azaz a robotnak valamilyen állapotba kell kerülnie, ha akciót hajt végre.
* Most a „jutalom" $r: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ fogalma segítségével meghatározzuk, hogy mely akciók hasznosak és melyek nem. Azt mondjuk, hogy a robot $r(s,a)$ jutalmat kap, ha az $a$ akciót hajtja végre az $s$ állapotban. Ha az $r(s, a)$ jutalom nagy, ez azt jelzi, hogy az $a$ akció végrehajtása az $s$ állapotban hasznosabb a robot céljának eléréséhez, azaz a zöld házhoz való eljutáshoz. Ha az $r(s, a)$ jutalom kicsi, akkor az $a$ akció kevésbé hasznos e cél eléréséhez. Fontos megjegyezni, hogy a jutalmat a felhasználó (a megerősítéses tanulási algoritmust létrehozó személy) tervezi meg a célt szem előtt tartva.

## Hozam és diszkontálási tényező

A fenti különböző összetevők együttesen alkotnak egy Markov-döntési folyamatot (MDP):
$$\textrm{MDP}: (\mathcal{S}, \mathcal{A}, T, r).$$

Tekintsük azt a helyzetet, amikor a robot egy adott $s_0 \in \mathcal{S}$ állapotban indul, és akciókat hajt végre, amelyek egy trajektóriát eredményeznek:
$$\tau = (s_0, a_0, r_0, s_1, a_1, r_1, s_2, a_2, r_2, \ldots).$$

Minden $t$ időlépésben a robot az $s_t$ állapotban van és végrehajtja az $a_t$ akciót, ami $r_t = r(s_t, a_t)$ jutalmat eredményez. A trajektória *hozama* a robot által az adott trajektória mentén szerzett összes jutalom:
$$R(\tau) = r_0 + r_1 + r_2 + \cdots.$$

A megerősítéses tanulás célja olyan trajektória megtalálása, amelynek a legnagyobb a *hozama*.

Gondoljunk arra a helyzetre, amikor a robot tovább utazik a rácsvilágban anélkül, hogy valaha elérné a célhelyet. Az állapotok és akciók sorozata egy trajektóriában ebben az esetben végtelen hosszú lehet, és bármely ilyen végtelenül hosszú trajektória *hozama* végtelen lesz. Annak érdekében, hogy a megerősítéses tanulás megfogalmazása értelmes maradjon még ilyen trajektóriák esetén is, bevezetjük a $\gamma < 1$ diszkontálási tényező fogalmát. A diszkontált *hozamot* így írjuk:
$$R(\tau) = r_0 + \gamma r_1 + \gamma^2 r_2 + \cdots = \sum_{t=0}^\infty \gamma^t r_t.$$

Figyeljük meg, hogy ha $\gamma$ nagyon kicsi, a robot által a távoli jövőben szerzett jutalmak, például $t = 1000$-nél, erősen diszkontálódnak a $\gamma^{1000}$ tényezővel. Ez arra ösztönzi a robotot, hogy rövid trajektóriákat válasszon, amelyek elérik a célját, nevezetesen a zöld házhoz való eljutást a rácsvilág-példában (lásd :numref:`fig_mdp`). A diszkontálási tényező nagy értékei esetén, például $\gamma = 0.99$-nél, a robot arra ösztönzött, hogy *felfedezzen*, majd megtalálja a legjobb trajektóriát a célhelyre való eljutáshoz.

## A Markov-feltevés megvitatása

Gondoljunk egy új robotra, ahol az $s_t$ állapot a fentiek szerint a helyszín, de az $a_t$ akció a gyorsulás, amelyet a robot alkalmaz a kerekein, egy elvont parancs, például „menj előre" helyett. Ha ennek a robotnak $s_t$ állapotban nem nulla sebessége van, akkor a következő $s_{t+1}$ helyszín függ a korábbi $s_t$ helyszíntől, az $a_t$ gyorsulásoktól, és a robot sebességétől $t$ időpontban, amely arányos $s_t - s_{t-1}$-gyel. Ez azt jelzi, hogy a következőt kellene írnunk:

$$s_{t+1} = \textrm{valamilyen függvény}(s_t, a_t, s_{t-1});$$

a „valamilyen függvény" esetünkben Newton mozgástörvénye lenne. Ez meglehetősen különbözik az átmeneti függvényünktől, amely egyszerűen csak $s_t$-től és $a_t$-től függ.

A Markov-rendszerek olyan rendszerek, ahol a következő $s_{t+1}$ állapot csak a jelenlegi $s_t$ állapot és a jelenlegi állapotban végrehajtott $a_t$ akció függvénye. Markov-rendszerekben a következő állapot nem függ attól, hogy a múltban milyen akciókat hajtottak végre, vagy attól, hogy a robot a múltban milyen állapotokban volt. Például az új robot, amelynél a gyorsulás az akció, nem Markov-tulajdonságú, mert a következő $s_{t+1}$ helyszín függ az előző $s_{t-1}$ állapottól a sebességen keresztül. Úgy tűnhet, hogy egy rendszer Markov-tulajdonsága korlátozó feltételezés, de nem az. A Markov-döntési folyamatok még mindig képesek valós rendszerek nagyon széles osztályát modellezni. Például az új robotunk esetén, ha az $s_t$ állapotot a $(\textrm{helyszín}, \textrm{sebesség})$ párnak választjuk, akkor a rendszer Markov-tulajdonságú, mert a következő állapota $(\textrm{helyszín}_{t+1}, \textrm{sebesség}_{t+1})$ csak a jelenlegi állapottól $(\textrm{helyszín}_t, \textrm{sebesség}_t)$ és a jelenlegi állapotban végrehajtott $a_t$ akciótól függ.

## Összefoglalás
A megerősítéses tanulási problémát általában Markov-döntési folyamatok segítségével modellezik. A Markov-döntési folyamatot (MDP) négy egyed $(\mathcal{S}, \mathcal{A}, T, r)$ négyes-ével definiálják, ahol $\mathcal{S}$ az állapottér, $\mathcal{A}$ az akciótér, $T$ az MDP átmeneti valószínűségeit kódoló átmeneti függvény, és $r$ az azonnali jutalom, amelyet egy adott állapotban egy akció végrehajtásával kapunk.


## Feladatok

1. Tegyük fel, hogy egy MDP-t szeretnénk tervezni a [MountainCar](https://www.gymlibrary.dev/environments/classic_control/mountain_car/) problémához.
    1. Mi lenne az állapotok halmaza?
    2. Mi lenne az akciók halmaza?
    3. Mik lennének a lehetséges jutalomfüggvények?
2. Hogyan terveznél MDP-t egy Atari-játékhoz, például a [Pong játékhoz](https://www.gymlibrary.dev/environments/atari/pong/)?

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/12084)
:end_tab:
