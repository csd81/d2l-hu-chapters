```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(["pytorch"])
#required_libs("setuptools==66", "wheel==0.38.4", "gym==0.21.0")
```

# Q-tanulás
:label:`sec_qlearning`

Az előző szakaszban az értékiteráció algoritmusát tárgyaltuk, amely megköveteli a teljes Markov-döntési folyamat (MDP) ismeretét, például az átmeneti és jutalomfüggvényekét. Ebben a szakaszban a Q-tanulást :cite:`Watkins.Dayan.1992` vizsgáljuk, amely egy olyan algoritmus, amellyel az értékfüggvény megtanulható anélkül, hogy feltétlenül ismernénk az MDP-t. Ez az algoritmus megtestesíti a megerősítéses tanulás központi gondolatát: lehetővé teszi a robot számára, hogy saját maga gyűjtsön adatokat.
<!-- , instead of relying upon the expert. -->

## A Q-tanulás algoritmusa

Idézzük fel, hogy a :ref:`sec_valueiter` szakaszban a cselekvés-értékfüggvényre vonatkozó értékiteráció a következő frissítésnek felel meg:

$$Q_{k+1}(s, a) = r(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \max_{a' \in \mathcal{A}} Q_k (s', a'); \ \textrm{for all } s \in \mathcal{S} \textrm{ and } a \in \mathcal{A}.$$

Amint tárgyaltuk, ennek az algoritmusnak az implementálásához ismerni kell az MDP-t, különösen a $P(s' \mid s, a)$ átmeneti függvényt. A Q-tanulás mögötti kulcsötlet, hogy a fenti kifejezésben az összes $s' \in \mathcal{S}$ feletti összeget felváltja a robot által meglátogatott állapotok feletti összeggel. Ez lehetővé teszi, hogy elkerüljük az átmeneti függvény ismeretének szükségességét.

## A Q-tanulás alapjául szolgáló optimalizálási feladat

Képzeljük el, hogy a robot egy $\pi_e(a \mid s)$ stratégiát alkalmaz cselekvések végrehajtásához. Akárcsak az előző fejezetben, $n$ trajektóriából álló adathalmazt gyűjt, mindegyik $T$ időlépésből: $\{ (s_t^i, a_t^i)_{t=0,\ldots,T-1}\}_{i=1,\ldots, n}$. Idézzük fel, hogy az értékiteráció valójában kényszerfeltételek halmaza, amelyek összekapcsolják a különböző állapotok és cselekvések $Q^*(s, a)$ cselekvési értékét. A $\pi_e$ által gyűjtött adatok segítségével az értékiteráció egy közelítő változatát a következőképpen implementálhatjuk:

$$\hat{Q} = \min_Q \underbrace{\frac{1}{nT} \sum_{i=1}^n \sum_{t=0}^{T-1} (Q(s_t^i, a_t^i) - r(s_t^i, a_t^i) - \gamma \max_{a'} Q(s_{t+1}^i, a'))^2}_{\stackrel{\textrm{def}}{=} \ell(Q)}.$$
:eqlabel:`q_learning_optimization_problem`

Először figyeljük meg a hasonlóságokat és különbségeket e kifejezés és a fenti értékiteráció között. Ha a robot $\pi_e$ stratégiája egyenlő lenne az optimális $\pi^*$ stratégiával, és végtelen mennyiségű adatot gyűjtene, akkor ez az optimalizálási feladat azonos lenne az értékiteráció alapjának optimalizálási feladatával. De míg az értékiteráció megköveteli $P(s' \mid s, a)$ ismeretét, az optimalizálási célfüggvénynek nincs ilyen tagja. Nem csaltunk: mivel a robot a $\pi_e$ stratégiát alkalmazza az $a_t^i$ cselekvés végrehajtásához az $s_t^i$ állapotban, a következő $s_{t+1}^i$ állapot az átmeneti függvényből vett minta. Az optimalizálási célnak tehát szintén van hozzáférése az átmeneti függvényhez, de implicit módon, a robot által gyűjtött adatok formájában.

Az optimalizálási feladatunk változói $Q(s, a)$ minden $s \in \mathcal{S}$ és $a \in \mathcal{A}$ esetén. A célértéket gradient descent segítségével minimalizálhatjuk. Az adathalmazban minden $(s_t^i, a_t^i)$ párra felírhatjuk:

$$\begin{aligned}Q(s_t^i, a_t^i) &\leftarrow Q(s_t^i, a_t^i) - \alpha \nabla_{Q(s_t^i,a_t^i)} \ell(Q) \\&=(1 - \alpha) Q(s_t^i,a_t^i) - \alpha \Big( r(s_t^i, a_t^i) + \gamma \max_{a'} Q(s_{t+1}^i, a') \Big),\end{aligned}$$
:eqlabel:`q_learning`

ahol $\alpha$ a tanulási ráta. Valós feladatokban általában, amikor a robot eléri a célhelyet, a trajektóriák véget érnek. Az ilyen terminális állapot értéke nulla, mivel a robot nem tesz további cselekvéseket ezen állapoton túl. A frissítést az ilyen állapotok kezelésére a következőképpen kell módosítanunk:

$$Q(s_t^i, a_t^i) =(1 - \alpha) Q(s_t^i,a_t^i) - \alpha \Big( r(s_t^i, a_t^i) + \gamma (1 - \mathbb{1}_{s_{t+1}^i \textrm{ is terminal}} )\max_{a'} Q(s_{t+1}^i, a') \Big).$$

ahol $\mathbb{1}_{s_{t+1}^i \textrm{ is terminal}}$ egy indikátorváltozó, amely $1$, ha $s_{t+1}^i$ terminális állapot, egyébként $0$. Az adathalmazba nem tartozó $(s, a)$ állapot-cselekvés párok értéke $-\infty$-re van állítva. Ezt az algoritmust Q-tanulásnak nevezzük.

Ezeknek a frissítéseknek a $\hat{Q}$ megoldása alapján, amely az optimális $Q^*$ értékfüggvény közelítése, könnyen megkapjuk az ehhez az értékfüggvényhez tartozó optimális determinisztikus stratégiát:

$$\hat{\pi}(s) = \mathrm{argmax}_{a} \hat{Q}(s, a).$$

Előfordulhatnak olyan helyzetek, amikor több determinisztikus stratégia felel meg ugyanannak az optimális értékfüggvénynek; az ilyen egyenlő eseteket tetszőlegesen lehet feloldani, mivel ugyanolyan értékfüggvényük van.

## Felfedezés a Q-tanulásban

A robot által adatgyűjtésre használt $\pi_e$ stratégia kritikus a Q-tanulás jó működéséhez. Végső soron az $s'$ feletti várható értéket felváltottuk a robot által gyűjtött adatokkal a $P(s' \mid s, a)$ átmeneti függvény helyett. Ha a $\pi_e$ stratégia nem látogatja meg az állapot-cselekvés tér változatos részeit, akkor könnyen elképzelhető, hogy a $\hat{Q}$ becslésünk rosszul közelíti az optimális $Q^*$-ot. Azt is fontos megjegyezni, hogy ilyen esetben a $Q^*$ becslése *minden* $s \in \mathcal{S}$ állapotra rossz lesz, nemcsak a $\pi_e$ által meglátogatottakra. Ez azért van, mert a Q-tanulás célját (vagy az értékiterációt) olyan kényszerfeltételek határozzák meg, amelyek összekapcsolják az összes állapot-cselekvés pár értékét. Ezért kritikus fontosságú a megfelelő $\pi_e$ stratégia kiválasztása az adatgyűjtéshez.

Ezt a problémát enyhíthetjük egy teljesen véletlenszerű $\pi_e$ stratégia kiválasztásával, amely egyenletesen véletlenszerűen mintáz cselekvéseket az $\mathcal{A}$-ból. Egy ilyen stratégia minden állapotot meglátogatna, de sok trajektóriára van szüksége, mielőtt ezt megtenné.

Így jutunk el a Q-tanulás második kulcsgondolatához: a felfedezéshez. A Q-tanulás tipikus implementációi összekapcsolják a $Q$ aktuális becslését és a $\pi_e$ stratégiát, és beállítják:

$$\pi_e(a \mid s) = \begin{cases}\mathrm{argmax}_{a'} \hat{Q}(s, a') & \textrm{with prob. } 1-\epsilon \\ \textrm{uniform}(\mathcal{A}) & \textrm{with prob. } \epsilon,\end{cases}$$
:eqlabel:`epsilon_greedy`

ahol $\epsilon$ az „felfedezési paraméter", amelyet a felhasználó választ meg. A $\pi_e$ stratégiát felfedezési stratégiának nevezzük. Ez a konkrét $\pi_e$ $\epsilon$-mohó felfedezési stratégiának hívják, mivel $1-\epsilon$ valószínűséggel az optimális cselekvést választja (az aktuális $\hat{Q}$ becslés szerint), de $\epsilon$ valószínűséggel véletlenszerűen fedez fel. Használhatjuk az úgynevezett softmax felfedezési stratégiát is:

$$\pi_e(a \mid s) = \frac{e^{\hat{Q}(s, a)/T}}{\sum_{a'} e^{\hat{Q}(s, a')/T}};$$

ahol a $T$ hiperparamétert hőmérsékletnek nevezzük. Az $\epsilon$-mohó stratégiában a nagy $\epsilon$ érték hasonlóan hat, mint a softmax stratégiában a nagy $T$ hőmérsékleti érték.

Fontos megjegyezni, hogy ha olyan felfedezést választunk, amely az aktuális cselekvés-értékfüggvény $\hat{Q}$ becslésétől függ, akkor az optimalizálási feladatot rendszeres időközönként újra kell oldanunk. A Q-tanulás tipikus implementációi minden egyes cselekvés végrehajtása után (a $\pi_e$ segítségével) egy mini-batch frissítést végeznek a gyűjtött adathalmaz néhány állapot-cselekvés párjával (általában az előző időlépésből gyűjtöttekkel).

## A Q-tanulás „önkorrekciós" tulajdonsága

A robot által Q-tanulás során gyűjtött adathalmaz az idővel növekszik. Mind a felfedezési stratégia $\pi_e$, mind a $\hat{Q}$ becslés fejlődik, ahogy a robot több adatot gyűjt. Ez kulcsbetekintést ad arról, miért működik jól a Q-tanulás. Tekintsünk egy $s$ állapotot: ha egy adott $a$ cselekvés nagy értékkel rendelkezik az aktuális $\hat{Q}(s,a)$ becslés szerint, akkor mind az $\epsilon$-mohó, mind a softmax felfedezési stratégia nagyobb valószínűséggel választja ezt a cselekvést. Ha ez a cselekvés valójában *nem* az ideális cselekvés, akkor az ebből a cselekvésből eredő jövőbeli állapotoknak alacsony jutalmuk lesz. A Q-tanulás célfüggvényének következő frissítése ezért csökkenti a $\hat{Q}(s,a)$ értéket, ami csökkenti ennek a cselekvésnek a kiválasztási valószínűségét, amikor a robot legközelebb az $s$ állapotba kerül. A rossz cselekvések — például azok, amelyek értékét $\hat{Q}(s,a)$-ban túlbecsülik — a robot által fel lesznek fedezve, de értékük a Q-tanulás célfüggvényének következő frissítésénél korrigálódik. A jó cselekvések — amelyek $\hat{Q}(s, a)$ értéke nagy — a robot által többször kerülnek felfedezésre, és ezáltal megerősítésre kerülnek. Ez a tulajdonság felhasználható annak bizonyítására, hogy a Q-tanulás konvergálhat az optimális stratégiához még akkor is, ha egy véletlenszerű $\pi_e$ stratégiával kezd :cite:`Watkins.Dayan.1992`.

Az a képesség, hogy nem csupán új adatokat gyűjt, hanem a megfelelő típusú adatokat is, a megerősítéses tanulási algoritmusok alapvető jellemzője, és ez különbözteti meg őket a felügyelt tanulástól. A Q-tanulás mély neurális hálózatokkal kombinálva (amelyeket a DQN fejezetben fogunk látni) felelős a megerősítéses tanulás újjáéledéséért :cite:`mnih2013playing`.

## A Q-tanulás implementációja

Most bemutatjuk, hogyan kell implementálni a Q-tanulást az [Open AI Gym](https://gym.openai.com) FrozenLake feladatán. Megjegyezzük, hogy ez ugyanaz a beállítás, mint amelyet a :ref:`sec_valueiter` kísérletben alkalmaztunk.

```{.python .input}
%%tab all

%matplotlib inline
import numpy as np
import random
from d2l import torch as d2l

seed = 0  # Véletlenszám-generátor magja
gamma = 0.95  # Diszkontfaktor
num_iters = 256  # Iterációk száma
alpha   = 0.9  # Tanulási ráta
epsilon = 0.9  # Epsilon az epsilon-mohó algoritmusban
random.seed(seed)  # Véletlenszám-generátor inicializálása
np.random.seed(seed)

# A környezet beállítása
env_info = d2l.make_env('FrozenLake-v1', seed=seed)
```

A FrozenLake környezetben a robot egy $4 \times 4$-es rácson (ezek az állapotok) mozog „fel" ($\uparrow$), „le" ($\rightarrow$), „balra" ($\leftarrow$) és „jobbra" ($\rightarrow$) cselekvésekkel. A környezet számos lyuk (H) cellát, jégcellát (F) és egy célcellát (G) tartalmaz, amelyek mind ismeretlenek a robot számára. A feladat egyszerűsítése érdekében feltételezzük, hogy a robot cselekvései megbízhatóak, azaz $P(s' \mid s, a) = 1$ minden $s \in \mathcal{S}, a \in \mathcal{A}$ esetén. Ha a robot eléri a célt, a kísérlet véget ér, és a robot $1$ értékű jutalmat kap a cselekvéstől függetlenül; minden más állapotban a jutalom $0$ minden cselekvés esetén. A robot célja, hogy megtanuljon egy olyan stratégiát, amely egy adott kezdőhelyzetből (S) (ez $s_0$) eljut a célhelyzetbe (G), maximalizálva a *hozamot*.

Először az $\epsilon$-mohó módszert implementáljuk a következőképpen:

```{.python .input}
%%tab all

def e_greedy(env, Q, s, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()

    else:
        return np.argmax(Q[s,:])

```

Most készen állunk a Q-tanulás implementálására:

```{.python .input}
%%tab all

def q_learning(env_info, gamma, num_iters, alpha, epsilon):
    env_desc = env_info['desc']  # 2D tömb, amely meghatározza az egyes rácselemek jelentését
    env = env_info['env']  # 2D tömb, amely meghatározza az egyes rácselemek jelentését
    num_states = env_info['num_states']
    num_actions = env_info['num_actions']

    Q  = np.zeros((num_states, num_actions))
    V  = np.zeros((num_iters + 1, num_states))
    pi = np.zeros((num_iters + 1, num_states))

    for k in range(1, num_iters + 1):
        # Környezet visszaállítása
        state, done = env.reset(), False
        while not done:
            # Cselekvés kiválasztása az adott állapothoz, majd végrehajtás a környezetben
            action = e_greedy(env, Q, state, epsilon)
            next_state, reward, done, _ = env.step(action)

            # Q-frissítés:
            y = reward + gamma * np.max(Q[next_state,:])
            Q[state, action] = Q[state, action] + alpha * (y - Q[state, action])

            # Lépés a következő állapotba
            state = next_state
        # Maximális érték és cselekvés rögzítése csak vizualizációs célból
        for s in range(num_states):
            V[k,s]  = np.max(Q[s,:])
            pi[k,s] = np.argmax(Q[s,:])
    d2l.show_Q_function_progress(env_desc, V[:-1], pi[:-1])

q_learning(env_info=env_info, gamma=gamma, num_iters=num_iters, alpha=alpha, epsilon=epsilon)

```

Ez az eredmény azt mutatja, hogy a Q-tanulás körülbelül 250 iteráció után megtalálja az optimális megoldást erre a feladatra. Ha azonban összehasonlítjuk ezt az eredményt az értékiteráció algoritmusának eredményével (lásd: :ref:`subsec_valueitercode`), láthatjuk, hogy az értékiteráció algoritmusnak sokkal kevesebb iterációra van szüksége az optimális megoldás megtalálásához ennél a feladatnál. Ez azért van, mert az értékiteráció algoritmusnak hozzáférése van a teljes MDP-hez, míg a Q-tanulásnak nincs.


## Összefoglalás
A Q-tanulás az egyik legalapvetőbb megerősítéses tanulási algoritmus. A megerősítéses tanulás legutóbbi sikerének középpontjában áll, leginkább a videójátékok játszásának megtanulásában :cite:`mnih2013playing`. A Q-tanulás implementálásához nem szükséges teljes mértékben ismerni a Markov-döntési folyamatot (MDP-t), vagyis az átmeneti és jutalomfüggvényeket.

## Feladatok

1. Próbáld meg növelni a rácsméretet $8 \times 8$-ra. A $4 \times 4$-es ráccsal összehasonlítva hány iteráció szükséges az optimális értékfüggvény megtalálásához?
1. Futtasd le újra a Q-tanulás algoritmust $\gamma$ értékével (azaz a fenti kódban „gamma"-val) $0$, $0.5$ és $1$ esetén, és elemezd az eredményeket.
1. Futtasd le újra a Q-tanulás algoritmust $\epsilon$ értékével (azaz a fenti kódban „epsilon"-nal) $0$, $0.5$ és $1$ esetén, és elemezd az eredményeket.

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/12103)
:end_tab:
