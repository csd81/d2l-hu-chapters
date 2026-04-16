```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(["pytorch"])
#required_libs("setuptools==66", "wheel==0.38.4", "gym==0.21.0")
```

# Értékiteráció
:label:`sec_valueiter`

Ebben a szakaszban arról lesz szó, hogyan válasszuk ki a legjobb cselekvést a robot számára minden egyes állapotban a trajektória *hozamának* maximalizálása érdekében. Bemutatjuk az értékiteráció nevű algoritmust, és implementáljuk azt egy fagyott tavon közlekedő szimulált robot számára.

## Sztochasztikus stratégia

A $\pi(a \mid s)$ jelölésű sztochasztikus stratégia (röviden: stratégia) egy feltételes eloszlás a cselekvések $a \in \mathcal{A}$ felett, adott az $s \in \mathcal{S}$ állapot esetén: $\pi(a \mid s) \equiv P(a \mid s)$. Például ha a robotnak négy cselekvése van: $\mathcal{A}=$ {balra megy, lefelé megy, jobbra megy, felfelé megy}. Az $s \in \mathcal{S}$ állapotban érvényes stratégia egy ilyen $\mathcal{A}$ cselekvéskészlet esetén kategorikus eloszlás, ahol a négy cselekvés valószínűsége lehet $[0.4, 0.2, 0.1, 0.3]$; egy másik $s' \in \mathcal{S}$ állapotban ugyanazon négy cselekvés valószínűsége $\pi(a \mid s')$ lehet $[0.1, 0.1, 0.2, 0.6]$. Megjegyezzük, hogy bármely $s$ állapotra teljesüljön $\sum_a \pi(a \mid s) = 1$. A determinisztikus stratégia a sztochasztikus stratégia speciális esete, amelyben a $\pi(a \mid s)$ eloszlás csak egyetlen cselekvéshez rendel nem nulla valószínűséget, például $[1, 0, 0, 0]$ a négy cselekvéses példánkban.

A jelölés egyszerűsítése érdekében a feltételes eloszlást $\pi(a \mid s)$ helyett gyakran $\pi(s)$-sel jelöljük.

## Értékfüggvény

Képzeljük el, hogy a robot egy $s_0$ állapotból indul, és minden egyes időpillanatban először mintát vesz egy cselekvésre a stratégiából: $a_t \sim \pi(s_t)$, majd végrehajtja a cselekvést, amelynek eredménye a következő $s_{t+1}$ állapot. A trajektória $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$ eltérő lehet attól függően, hogy az egyes közbülső pillanatokban melyik $a_t$ cselekvést mintázzák. Az összes ilyen trajektória átlagos *hozamát* $R(\tau) = \sum_{t=0}^\infty \gamma^t r(s_t, a_t)$ a következőképpen definiáljuk:
$$V^\pi(s_0) = E_{a_t \sim \pi(s_t)} \Big[ R(\tau) \Big] = E_{a_t \sim \pi(s_t)} \Big[ \sum_{t=0}^\infty \gamma^t r(s_t, a_t) \Big],$$

ahol $s_{t+1} \sim P(s_{t+1} \mid s_t, a_t)$ a robot következő állapota, és $r(s_t, a_t)$ az azonnali jutalom, amelyet az $a_t$ cselekvés végrehajtásával kap a robot a $t$ időpillanatban az $s_t$ állapotban. Ezt nevezzük a $\pi$ stratégia „értékfüggvényének". Egyszerűen fogalmazva, a $V^\pi(s_0)$ jelölésű $s_0$ állapot értéke a $\pi$ stratégia szerint az a várható $\gamma$-diszkontált *hozam*, amelyet a robot kap, ha az $s_0$ állapotból indul, és minden időpillanatban a $\pi$ stratégia szerint cselekszik.

Ezután a trajektóriát két szakaszra bontjuk: (i) az első szakasz az $a_0$ cselekvés végrehajtásával megfelelő $s_0 \to s_1$ átmenetre, és (ii) egy második szakasz, amely az ezt követő $\tau' = (s_1, a_1, r_1, \ldots)$ trajektória. A megerősítéses tanulás összes algoritmusának alapötlete, hogy az $s_0$ állapot értéke felírható az első szakaszban kapott átlagos jutalom és az összes lehetséges következő $s_1$ állapotra átlagolt értékfüggvény összegeként. Ez meglehetősen intuitív, és a Markov-feltételezésünkből fakad: az aktuális állapotból elérhető átlagos hozam egyenlő a következő állapotból elérhető átlagos hozam és a következő állapotba jutás átlagos jutalmának összegével. Matematikailag a két szakaszt a következőképpen írjuk fel:

$$V^\pi(s_0) = r(s_0, a_0) + \gamma\ E_{a_0 \sim \pi(s_0)} \Big[ E_{s_1 \sim P(s_1 \mid s_0, a_0)} \Big[ V^\pi(s_1) \Big] \Big].$$
:eqlabel:`eq_dynamic_programming`

Ez a dekompozíció rendkívül hatékony: ez a dinamikus programozás elvének alapja, amelyre az összes megerősítéses tanulási algoritmus épül. Figyeljük meg, hogy a második szakaszban két várható érték szerepel: egy az első szakaszban a sztochasztikus stratégia által választott $a_0$ cselekvés felett, és egy másik a választott cselekvésből eredő lehetséges $s_1$ állapotok felett. A :eqref:`eq_dynamic_programming` egyenletet a Markov-döntési folyamat (MDP) átmeneti valószínűségeivel a következőképpen írhatjuk:

$$V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \Big[ r(s,  a) + \gamma\  \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V^\pi(s') \Big];\ \textrm{for all } s \in \mathcal{S}.$$
:eqlabel:`eq_dynamic_programming_val`

Fontos megjegyezni, hogy a fenti azonosság minden $s \in \mathcal{S}$ állapotra teljesül, mivel bármely ebből az állapotból induló trajektóriát két szakaszra bonthatjuk.

## Cselekvés-értékfüggvény

Az implementációkban gyakran hasznos fenntartani a „cselekvési érték" függvényt, amely szorosan kapcsolódik az értékfüggvényhez. Ez a cselekvés-értékfüggvény az $s_0$-ból induló trajektória átlagos *hozamaként* van definiálva, de az első szakasz cselekvése rögzített $a_0$-ra:

$$Q^\pi(s_0, a_0) = r(s_0, a_0) + E_{a_t \sim \pi(s_t)} \Big[ \sum_{t=1}^\infty \gamma^t r(s_t, a_t) \Big],$$

Megjegyezzük, hogy a várható értékben szereplő összeg $t=1,\ldots, \infty$-től indul, mivel az első szakasz jutalma ebben az esetben rögzített. A trajektóriát ismét két részre bonthatjuk és felírhatjuk:

$$Q^\pi(s, a) = r(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \sum_{a' \in \mathcal{A}} \pi(a' \mid s')\ Q^\pi(s', a');\ \textrm{ for all } s \in \mathcal{S}, a \in \mathcal{A}.$$
:eqlabel:`eq_dynamic_programming_q`

Ez a változat a cselekvés-értékfüggvény :eqref:`eq_dynamic_programming_val` egyenletének analógja.

## Optimális sztochasztikus stratégia

Mind az értékfüggvény, mind a cselekvés-értékfüggvény a robot által választott stratégiától függ. A továbbiakban az „optimális stratégiát" vizsgáljuk, amely maximális átlagos *hozamot* ér el:
$$\pi^* = \underset{\pi}{\mathrm{argmax}} V^\pi(s_0).$$

Az összes lehetséges sztochasztikus stratégia közül az optimális $\pi^*$ stratégia éri el a legnagyobb átlagos diszkontált *hozamot* az $s_0$ állapotból induló trajektóriákra. Az optimális stratégia értékfüggvényét és cselekvés-értékfüggvényét $V^* \equiv V^{\pi^*}$ és $Q^* \equiv Q^{\pi^*}$ jelöli.

Figyeljük meg, hogy egy determinisztikus stratégia esetén bármely adott állapotban csak egyetlen cselekvés lehetséges a stratégia szerint. Ez a következőt adja:

$$\pi^*(s) = \underset{a \in \mathcal{A}}{\mathrm{argmax}} \Big[ r(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a)\ V^*(s') \Big].$$

Egy jó emlékeztetési szabály: az $s$ állapotban az optimális cselekvés (determinisztikus stratégia esetén) az, amelyik maximalizálja az első szakasz $r(s, a)$ jutalmának és a második szakasz összes lehetséges következő $s'$ állapotán átlagolt, a következő $s'$ állapotból induló trajektóriák átlagos *hozamának* összegét.

## A dinamikus programozás elve

A :eqref:`eq_dynamic_programming` vagy :eqref:`eq_dynamic_programming_q` egyenletekben végzett fejtegetésünk algoritmussá alakítható az optimális értékfüggvény $V^*$ vagy a cselekvés-értékfüggvény $Q^*$ kiszámítására. Figyeljük meg, hogy
$$ V^*(s) = \sum_{a \in \mathcal{A}} \pi^*(a \mid s) \Big[ r(s,  a) + \gamma\  \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V^*(s') \Big];\ \textrm{for all } s \in \mathcal{S}.$$

Egy determinisztikus optimális $\pi^*$ stratégia esetén, mivel az $s$ állapotban csak egyetlen cselekvés vehető fel, a következőt is felírhatjuk:

$$V^*(s) = \mathrm{argmax}_{a \in \mathcal{A}} \Big\{ r(s,a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V^*(s') \Big\}$$

minden $s \in \mathcal{S}$ állapotra. Ezt az azonosságot a „dinamikus programozás elvének" nevezzük :cite:`BellmanDPPaper,BellmanDPBook`. Richard Bellman fogalmazta meg az 1950-es években, és úgy emlékezhetünk rá, mint „egy optimális trajektória folytatása szintén optimális".

## Értékiteráció

A dinamikus programozás elvét algoritmussá alakíthatjuk az optimális értékfüggvény megkeresésére, amelyet értékiterációnak nevezünk. Az értékiteráció mögötti kulcsötlet, hogy ezt az azonosságot olyan kényszerfeltételek halmazaként tekintjük, amelyek összekapcsolják a $V^*(s)$ értékeket a különböző $s \in \mathcal{S}$ állapotokban. Az értékfüggvényt tetszőleges $V_0(s)$ értékekkel inicializáljuk minden $s \in \mathcal{S}$ állapotra. A $k^{\textrm{th}}$ iteráció során az értékiteráció algoritmus az értékfüggvényt a következőképpen frissíti:

$$V_{k+1}(s) = \max_{a \in \mathcal{A}} \Big\{ r(s,  a) + \gamma\  \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V_k(s') \Big\};\ \textrm{for all } s \in \mathcal{S}.$$

Kiderül, hogy $k \to \infty$ esetén az értékiteráció algoritmus által becsült értékfüggvény konvergál az optimális értékfüggvényhez, függetlenül a $V_0$ inicializálástól:
$$V^*(s) = \lim_{k \to \infty} V_k(s);\ \textrm{for all states } s \in \mathcal{S}.$$

Ugyanez az értékiteráció algoritmus cselekvés-értékfüggvénnyel is felírható:
$$Q_{k+1}(s, a) = r(s, a) + \gamma \max_{a' \in \mathcal{A}} \sum_{s' \in \mathcal{S}} P(s' \mid s, a) Q_k (s', a');\ \textrm{ for all } s \in \mathcal{S}, a \in \mathcal{A}.$$

Ebben az esetben a $Q_0(s, a)$ értékeket tetszőleges értékekre inicializáljuk minden $s \in \mathcal{S}$ és $a \in \mathcal{A}$ esetén. Ismét fennáll, hogy $Q^*(s, a) = \lim_{k \to \infty} Q_k(s, a)$ minden $s \in \mathcal{S}$ és $a \in \mathcal{A}$ esetén.

## Stratégiaértékelés

Az értékiteráció lehetővé teszi az optimális értékfüggvény kiszámítását, azaz az optimális determinisztikus $\pi^*$ stratégia $V^{\pi^*}$ értékfüggvényét. Hasonló iteratív frissítésekkel bármely más, potenciálisan sztochasztikus $\pi$ stratégiához tartozó értékfüggvényt is kiszámíthatjuk. Ismét tetszőleges értékekre inicializáljuk a $V^\pi_0(s)$ értékeket minden $s \in \mathcal{S}$ állapotra, és a $k^{\textrm{th}}$ iteráció során elvégezzük a frissítéseket:

$$    V^\pi_{k+1}(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \Big[ r(s,  a) + \gamma\  \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V^\pi_k(s') \Big];\ \textrm{for all } s \in \mathcal{S}.$$

Ezt az algoritmust stratégiaértékelésnek nevezzük, és hasznos az adott stratégiához tartozó értékfüggvény kiszámításához. Ismét igaz, hogy $k \to \infty$ esetén ezek a frissítések konvergálnak a helyes értékfüggvényhez, függetlenül a $V_0$ inicializálástól:

$$V^\pi(s) = \lim_{k \to \infty} V^\pi_k(s);\ \textrm{for all states } s \in \mathcal{S}.$$

A $\pi$ stratégia cselekvés-értékfüggvényének $Q^\pi(s, a)$ kiszámítására vonatkozó algoritmus analóg.

## Az értékiteráció implementációja
:label:`subsec_valueitercode`
A következőkben bemutatjuk, hogyan kell implementálni az értékiterációt az [Open AI Gym](https://gym.openai.com) FrozenLake nevű navigációs feladatához. Először be kell állítani a környezetet az alábbi kód szerint.

```{.python .input}
%%tab all

%matplotlib inline
import numpy as np
import random
from d2l import torch as d2l

seed = 0  # Véletlenszám-generátor magja
gamma = 0.95  # Diszkontálási tényező
num_iters = 10  # Iterációk száma
random.seed(seed)  # Véletlen mag beállítása a reprodukálhatóság érdekében
np.random.seed(seed)

# A környezet beállítása
env_info = d2l.make_env('FrozenLake-v1', seed=seed)
```

A FrozenLake környezetben a robot egy $4 \times 4$-es rácson (ezek az állapotok) mozog „fel" ($\uparrow$), „le" ($\rightarrow$), „balra" ($\leftarrow$) és „jobbra" ($\rightarrow$) cselekvésekkel. A környezet számos lyuk (H) cellát, jégcellát (F) és egy célcellát (G) tartalmaz, amelyek mind ismeretlenek a robot számára. A feladat egyszerűsítése érdekében feltételezzük, hogy a robot cselekvései megbízhatóak, azaz $P(s' \mid s, a) = 1$ minden $s \in \mathcal{S}, a \in \mathcal{A}$ esetén. Ha a robot eléri a célt, a kísérlet véget ér, és a robot $1$ értékű jutalmat kap a cselekvéstől függetlenül; minden más állapotban a jutalom $0$ minden cselekvés esetén. A robot célja, hogy megtanuljon egy olyan stratégiát, amely egy adott kezdőhelyzetből (S) (ez $s_0$) eljut a célhelyzetbe (G), maximalizálva a *hozamot*.

A következő függvény implementálja az értékiterációt, ahol az `env_info` MDP-vel és környezettel kapcsolatos információkat tartalmaz, a `gamma` pedig a diszkontálási tényező:

```{.python .input}
%%tab all

def value_iteration(env_info, gamma, num_iters):
    env_desc = env_info['desc']  # 2D tömb: megmutatja, mit jelent minden elem
    prob_idx = env_info['trans_prob_idx']
    nextstate_idx = env_info['nextstate_idx']
    reward_idx = env_info['reward_idx']
    num_states = env_info['num_states']
    num_actions = env_info['num_actions']
    mdp = env_info['mdp']

    V  = np.zeros((num_iters + 1, num_states))
    Q  = np.zeros((num_iters + 1, num_states, num_actions))
    pi = np.zeros((num_iters + 1, num_states))

    for k in range(1, num_iters + 1):
        for s in range(num_states):
            for a in range(num_actions):
                # Kiszámítja: \sum_{s'} p(s'\mid s,a) [r + \gamma v_k(s')]
                for pxrds in mdp[(s,a)]:
                    # mdp(s,a): [(p1,next1,r1,d1),(p2,next2,r2,d2),..]
                    pr = pxrds[prob_idx]  # p(s'\mid s,a)
                    nextstate = pxrds[nextstate_idx]  # Következő állapot
                    reward = pxrds[reward_idx]  # Jutalom
                    Q[k,s,a] += pr * (reward + gamma * V[k - 1, nextstate])
            # A maximális érték és a maximális cselekvés rögzítése
            V[k,s] = np.max(Q[k,s,:])
            pi[k,s] = np.argmax(Q[k,s,:])
    d2l.show_value_function_progress(env_desc, V[:-1], pi[:-1])

value_iteration(env_info=env_info, gamma=gamma, num_iters=num_iters)
```

A fenti képek a stratégiát (a nyíl jelzi a cselekvést) és az értékfüggvényt mutatják (a szín változása azt mutatja, hogyan változik az értékfüggvény az idővel a sötét színnel jelzett kezdeti értéktől a világos színnel jelzett optimális értékig). Amint láthatjuk, az értékiteráció 10 iteráció után megtalálja az optimális értékfüggvényt, és a célállapot (G) bármely állapotból elérhető, amennyiben az nem H-cella. Az implementáció másik érdekes aspektusa, hogy az optimális értékfüggvény megtalálása mellett automatikusan megkaptuk az ehhez az értékfüggvényhez tartozó optimális stratégiát ($\pi^*$) is.


## Összefoglalás
Az értékiteráció algoritmus mögötti fő ötlet a dinamikus programozás elvének alkalmazása az adott állapotból elérhető optimális átlagos hozam megtalálásához. Megjegyezzük, hogy az értékiteráció implementálásához teljes mértékben ismernünk kell a Markov-döntési folyamatot (MDP-t), vagyis az átmeneti és jutalomfüggvényeket.


## Feladatok

1. Próbáld meg növelni a rácsméretet $8 \times 8$-ra. A $4 \times 4$-es ráccsal összehasonlítva hány iteráció szükséges az optimális értékfüggvény megtalálásához?
1. Mi az értékiteráció algoritmus számítási bonyolultsága?
1. Futtasd le újra az értékiteráció algoritmust $\gamma$ értékével (azaz a fenti kódban „gamma"-val) $0$, $0.5$ és $1$ esetén, és elemezd az eredményeket.
1. Hogyan befolyásolja a $\gamma$ értéke az értékiteráció konvergenciájához szükséges iterációk számát? Mi történik, ha $\gamma=1$?

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/12005)
:end_tab:
