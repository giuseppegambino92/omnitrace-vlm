# omnitrace

[English](README.md) · **Italiano**

<a href="https://paypal.me/giuseppegambino92"><img src="https://www.paypalobjects.com/it_IT/IT/i/btn/btn_donateCC_LG.gif" alt="Fai una donazione con PayPal"></a>

**RAG visuale multimodale a late interaction · correlatore causale cross-modale di timeline · motore di sicurezza per l'IA**

omnitrace è costruito da principi primi: `numpy`, `scipy`, `Pillow`, `sqlite3`, più
`FastAPI` / `Typer` / `Rich` per le interfacce. Non usa framework di ML, modelli di embedding
né OCR, e ogni test e benchmark gira offline e in modo deterministico. Un VLM (locale tramite
Ollama, oppure qualsiasi endpoint compatibile con OpenAI) può essere collegato, in via
facoltativa, per *leggere* l'evidenza che omnitrace ha già recuperato e localizzato.

| Sottosistema | Cosa fa |
|---|---|
| **A · RAG spaziale a patch senza OCR** | Codifica ogni pagina o screenshot come una griglia 32×32 di patch multi-vettore (feature testuali contestualizzate dal grafo di layout, descrittori di colore/gradiente, coordinate spaziali). Recupera con MaxSim in stile ColPali (float32 / int8 / 2-bit / 1-bit) e restituisce una heatmap di attivazione 2-D più il **bounding box** della regione che contiene la risposta, come citazione verificabile. |
| **B · RCA cross-modale** | Riduce le registrazioni dello schermo ai keyframe (hashing percettivo consapevole dei cicli), rileva i sintomi visibili nei pixel e nel DOM, costruisce un DAG causale che rispetta il tempo (frame ↔ DOM ↔ span ↔ log) e ordina gli span candidati a causa radice, ciascuno con il suo percorso causale fino al sintomo. |
| **C · Shield anti-iniezione + loop breaker** | Analizza documenti recuperati, testo del DOM, alt text e output dei tool alla ricerca di override delle istruzioni, falsificazione di ruoli, contrabbando via Unicode invisibile, payload codificati e dirottamento dei tool. Applica spotlighting/datamarking e traccia la provenienza (taint) delle chiamate ai tool. Ferma i loop degli agenti tramite l'entropia semantica delle azioni. |

### Architettura

```text
      pagine / screenshot              video·DOM·span·log          testo non fidato · tool call
               │                                │                                │
┌──────────────▼─────────────┐   ┌──────────────▼─────────────┐   ┌──────────────▼─────────────┐
│ A  RAG spaziale a patch    │   │ B  RCA cross-modale        │   │ C  shield + loop breaker   │
│ grafo di layout            │   │ keyframe + animazioni      │   │ Unicode · lessico · decode │
│ vettori di patch 32×32     │   │ sintomi da pixel + DOM     │   │ rischio noisy-OR ≥ 0,8     │
│ f32 · int8 · 2-bit · 1-bit │   │ DAG causale ordinato       │   │ spotlight · ToolCallGuard  │
│ shortlist 1-bit → MaxSim   │   │ evidenza · origine · reach │   │ entropia azioni · cicli    │
│ heatmap soft-AND           │   │                            │   │                            │
│ → bbox + ritaglio          │   │ → span radice + percorso   │   │ → blocca / ferma           │
└──────────────┬─────────────┘   └──────────────┬─────────────┘   └──────────────┬─────────────┘
               │ testo regioni recuperate ──────┼───────────────────────────────▶┤
               ▼                                ▼                                ▼
         indice SQLite                     report RCA                  verdetto / spotlight
               │                                │                                │
               └────────────────────────────────┼────────────────────────────────┘
                                                ▼
                      server MCP (HTTP /mcp · stdio) · CLI Typer · TUI Rich
                                                │
                                                ▼
                     agente / VLM opzionale (legge solo il ritaglio citato)
```

> **Leggere prima dei numeri.** Tutti i dati dei benchmark sono **sintetici**: sono generati dal
> codice in `evals/datasets/` ed esportati per l'ispezione. Le euristiche sono state sviluppate
> sugli split *dev*, perciò la suite riporta anche split **held-out** (seed del generatore mai
> visti, più un corpus di iniezioni scritto separatamente). Gli split held-out condividono però
> i template del dev: dimostrano robustezza alla randomizzazione, non a nuovi tipi di documenti,
> applicazioni o attacchi. Vedi [Limiti](#limiti).

---

## Risultati

Prodotti da `python -m evals.benchmark_suite`. Il blocco seguente viene riscritto a ogni
esecuzione; i numeri grezzi sono in [`evals/latest_report.json`](evals/latest_report.json).

<!-- METRICS:START -->
_Misurato il 2026-11-14 06:14 su Darwin arm64 (Python 3.11.18), omnitrace 1.0.0, durata totale della suite 14 s._

| Benchmark | Metrica | Obiettivo | **omnitrace** | Miglior baseline |
|---|---|---|---|---|
| 1. Grounding 2-D (dev, 60 query) | Region Recall@1 (IoU>=0,5), int8 | >= 90% | **100,0%** | BM25 a chunk piatti 0,0% / BM25 per regione 46,7% |
| 1. Grounding 2-D (seed held-out 11, 23, 42) | Region Recall@1, int8 | >= 90% | **100,0%** | BM25 a chunk piatti 0,0% |
| 2. Deduplicazione frame (200 frame) | frame ridondanti eliminati | > 90% | **92,0%** | dHash globale 95,5% |
| 2. Deduplicazione frame | latenza per frame (mediana di 7 esecuzioni) | < 0,5 ms | **0,133 ms** | - |
| 2. Deduplicazione frame | recall delle transizioni (keyframe) | 100% | **100,0%** | dHash globale 41,7% |
| 3. RCA cross-modale (dev, 10 tracce) | Top-1 span causa radice | >= 90% | **100,0%** | ultimo span in errore prima del sintomo 10,0% |
| 3. RCA cross-modale (seed held-out) | Top-1 | >= 90% | **100,0% / 100,0%** | - |
| 4. Iniezione + loop (dev, 30 avversari / 35 benigni) | recall | >= 95% | **100,0%** | - |
| 4. Iniezione + loop (dev) | tasso di falsi positivi | 0% | **0,0%** | - |
| 4. Iniezione (held-out, 12 / 12) | recall / FPR | >= 95% / 0% | **100,0% / 0,0%** | - |
| 4. Latenza dello shield (documenti dev) | p95 / max per documento | < 2 ms | **0,20 / 0,47 ms** | - |

**Quantizzazione (grounding dev, Recall@1 di pagina / Recall@1 di regione, byte per vettore di patch):**

| precisione | R@1 pagina | R@1 regione | IoU medio | byte/patch | query p50 |
|---|---|---|---|---|---|
| float32 | 100,0% | 100,0% | 1,000 | 4096 | 5,5 ms |
| int8 | 100,0% | 100,0% | 1,000 | 1028 | 5,4 ms |
| int2 | 100,0% | 100,0% | 1,000 | 260 | 5,4 ms |
| binary | 100,0% | 100,0% | 1,000 | 132 | 5,8 ms |

**Ablazioni / misure aggiuntive:**

- ranking di pagina con MaxSim ColBERT grezzo (non calibrato): R@1 pagina 96,7%, R@1 regione 96,7%
- bbox dalla sola heatmap di attivazione (senza proposte dallo structure layer, modalità screenshot grezzo): R@1 regione 61,7%, IoU medio 0,617
- il BM25 a chunk piatti trova la *pagina* giusta nel 95,0% dei casi, ma il suo box di citazione ha IoU medio 0,050
- RCA senza i frame video (solo DOM + span + log): Top-1 80,0%
- baseline RCA (dev): primo span in errore 0,0%, ultimo span in errore prima del sintomo 10,0%, span più lungo 0,0%, span con più log di errore 0,0%
- latenza media della RCA 14,1 ms per traccia (inclusa l'estrazione dei keyframe da 60 frame), 67 span per traccia
- loop breaker: 100,0% dei loop fermati in media dopo 4,8 passi, 0,0% di arresti errati, ~52800 token risparmiati per loop su un budget di 60k
- codifica dell'indice 34 ms/pagina (1024 patch x 1024 dimensioni)
<!-- METRICS:END -->

### Ripetibilità

`python -m evals.repeat_runs` esegue l'intera suite 3 volte di fila, ciascuna in un processo
nuovo, e avvia ogni esecuzione solo quando il carico medio a 1 minuto è al massimo pari a una soglia
(predefinita 0,35 per CPU logica, `--max-load`; il blocco seguente riporta la soglia effettivamente
usata e il carico all'avvio di ogni esecuzione). Ogni metrica non temporale deve essere identica bit per bit tra le esecuzioni (la
pipeline è deterministica). Le metriche di tempo sono riportate come intervalli `[min, max]` e
i loro obiettivi devono valere in ogni esecuzione. I dettagli sono in
[`evals/repeatability_report.json`](evals/repeatability_report.json).

<!-- REPEAT:START -->
_3 esecuzioni consecutive il 2026-11-14 06:14 su Darwin arm64 (14 CPU logiche), ciascuna in un processo separato; carico a 1 minuto per core all'avvio di ogni esecuzione: 0,331, 0,304, 0,277 (soglia 0,500)._

**270 metriche non temporali confrontate: tutte identiche bit per bit in ogni esecuzione.**

| Metrica di tempo | Obiettivo | Esec. 1 | Esec. 2 | Esec. 3 | Intervallo [min, max] | CV | Obiettivo in ogni esecuzione |
|---|---|---|---|---|---|---|---|
| Deduplicazione frame, ms/frame (mediana di 7) | < 0,500 | 0,133 | 0,134 | 0,133 | [0,133, 0,134] | 0,1% | sì |
| Latenza shield p95, ms/doc | < 2,00 | 0,195 | 0,196 | 0,196 | [0,195, 0,196] | 0,4% | sì |
| Latenza shield max, ms/doc | < 2,00 | 0,460 | 0,475 | 0,471 | [0,460, 0,475] | 1,3% | sì |
| Query visuale p50 (int8), ms | - | 5,35 | 5,34 | 5,38 | [5,34, 5,38] | 0,3% | - |
| Latenza media RCA per traccia, ms | - | 14,15 | 14,40 | 14,07 | [14,07, 14,40] | 1,0% | - |
| Loop breaker, ms/passo | - | 0,046 | 0,046 | 0,046 | [0,046, 0,046] | 0,2% | - |

| Metrica di accuratezza | Esec. 1 | Esec. 2 | Esec. 3 |
|---|---|---|---|
| Grounding R@1 regione (dev, int8) | 100,0% | 100,0% | 100,0% |
| Grounding R@1 regione (held-out) | 100,0% | 100,0% | 100,0% |
| Frame eliminati | 92,0% | 92,0% | 92,0% |
| Recall transizioni | 100,0% | 100,0% | 100,0% |
| RCA Top-1 (dev) | 100,0% | 100,0% | 100,0% |
| Recall iniezioni + loop (dev) | 100,0% | 100,0% | 100,0% |
| FPR iniezioni + loop (dev) | 0,0% | 0,0% | 0,0% |
| Recall iniezioni (held-out) | 100,0% | 100,0% | 100,0% |
<!-- REPEAT:END -->

### Cosa misurano i benchmark

1. **Grounding 2-D.** 15 pagine (1024×1024), tre per ciascun tipo: tabelle di specifiche (due
   tabelle per pagina), schemi P&ID, specifiche ingegneristiche a 2–3 colonne, interfacce di
   applicazioni e grafici a barre. Le query sono 60. Un successo richiede la pagina corretta **e**
   IoU ≥ 0,5 tra il box restituito e la regione di risposta di riferimento, che può essere una
   cella di tabella, un nodo dello schema, un paragrafo, un widget o una barra del grafico. Il
   vocabolario è condiviso di proposito tra le pagine (ogni tabella ha "Max Voltage"), quindi un
   successo richiede l'entità giusta *e* la relazione 2-D giusta.
   *BM25 a chunk piatti* è la classica pipeline RAG: linearizzazione in ordine di lettura, finestre
   da 40 token e una citazione pari al box che unisce le parole del chunk. *BM25 per regione*
   indicizza ogni elemento di layout separatamente, ma senza contesto 2-D.
2. **Deduplicazione dei frame.** 200 frame rumorosi (σ = 1,5 livelli di grigio) con 12 transizioni
   reali, tra cui il cambio di una sola cifra in un badge, uno spinner di 25 frame, l'apertura e
   la chiusura di un modale, un toast e una dissolvenza. La recall è calcolata sul frame esatto.
   La baseline conserva i frame il cui dHash globale è cambiato di più di 4 bit.
3. **RCA cross-modale.** 10 sessioni fallite di un negozio online (12 s, 60 frame, circa 66 span
   su circa 14 servizi). Le cause radice sono: un timeout di statement sul DB tre hop più in
   basso, una deriva di schema dietro un HTTP 200, uno stallo di Elasticsearch di 30 s, un token
   rifiutato con 401, un 503 nascosto dietro un 502 del gateway, CORS (status 0), un 429 del
   gateway, campi di prezzo nulli, una deriva nella configurazione dei feature flag e tre retry
   in deadlock. Distrattori presenti in ogni traccia: una favicon in 404, un 503 risolto da un
   retry, un consumer di email non correlato che va in errore, un cron job lento ma sano, errori
   che avvengono *dopo* il sintomo, un 500 concorrente non correlato e un innocuo badge rosso
   "SALE".
4. **Sicurezza.** 30 documenti tecnici benigni, ognuno costruito attorno a un caso negativo
   difficile: righe di installazione `curl … | sh`, JSON di ruolo nella documentazione di API per
   LLM, riferimenti SDK a `send_email()`, secret Kubernetes in base64, JWT, liste SHA-256, ZWJ
   delle emoji, ZWNJ persiano, unità di misura greche, chiavi YAML `system:`, "do not email
   credentials", *agenti* umani di supporto che "should" fare qualcosa. Più 25 payload inseriti
   in contenitori realistici: README, recensioni, testo bianco nei PDF, JSON di tool, inviti di
   calendario, aria-label ed EXIF. Più 5 traiettorie di agente cicliche e 5 che progrediscono.

---

## Avvio rapido

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest                                   # 63 test, ~8 s, completamente offline
python -m evals.benchmark_suite          # ~20-40 s; scrive evals/latest_report.json + i blocchi metriche di entrambi i README
python -m evals.repeat_runs --update-metrics   # 3 esecuzioni con gate sul carico + intervalli di ripetibilità in entrambi i README
omnitrace dash                           # citazione con heatmap, timeline causale e allarmi di sicurezza nel terminale
```

### CLI

```bash
# A · indicizza una cartella di immagini di pagina (+ structure layer facoltativo <nome>.json) e interrogala
omnitrace index ./pages --db pages.sqlite            # oppure: omnitrace index --demo
omnitrace query "max current of P-204" --db pages.sqlite --crop citation.png
omnitrace query "max current of P-204" --vlm ollama:qwen2.5vl     # il VLM legge solo il ritaglio citato

# B · analisi della causa radice di un'esecuzione fallita (JSON TraceBundle: span, log, dom, frame facoltativi)
omnitrace rca trace.json          # oppure: omnitrace rca --demo list / --demo payment_db_timeout

# C · analizza contenuto non fidato (exit code 2 se bloccato), con spotlighting facoltativo
cat page.html | omnitrace shield - --source dom_text --spotlight datamark

omnitrace serve                    # MCP su HTTP streamable a http://127.0.0.1:8765/mcp
omnitrace serve --stdio            # MCP su stdio
```

File di structure layer affiancato alla pagina (`page.json`), cioè ciò che forniscono il layer di
testo di un PDF, uno snapshot del DOM o un export SVG:

```json
{"regions": [
  {"id": "h2", "bbox": [430, 124, 560, 158], "text": "Max Current", "role": "header", "parent": "spec"},
  {"id": "r3c2", "bbox": [430, 260, 560, 294], "text": "12.5 A", "role": "cell", "parent": "spec"},
  {"id": "spec", "bbox": [40, 124, 960, 464], "text": "Electrical ratings", "role": "container"}
]}
```

I ruoli che portano significato di layout sono `header`, `rowheader`, `label`, `heading`,
`caption`, `title`, `cell`, `container` e `node`/`widget`/`bar`/`text`. Anche le pagine senza
file affiancato funzionano: in quel caso il bounding box deriva dalla sola heatmap (vedi
l'ablazione).

### MCP

Tool: `omnitrace_visual_query`, `omnitrace_rca`, `omnitrace_sanitize`, `omnitrace_loop_check`.
Il testo delle regioni restituito da `omnitrace_visual_query` viene a sua volta analizzato e
racchiuso tra delimitatori con nonce, perché il testo recuperato non è fidato.

```json
{"mcpServers": {"omnitrace": {"command": "omnitrace", "args": ["serve", "--stdio"],
                              "env": {"OMNITRACE_DB": "/abs/path/pages.sqlite"}}}}
```

---

## Come funziona

### Le formule in breve

La soglia di rumore è $`\nu = 2.5/\sqrt{976}`$ (il livello di rumore dell'hashing), $`m_{i,d}`$
è la migliore similarità di patch del token di query $`i`$ sulla pagina $`d`$, e $`s_{ij}`$ è la
sua similarità con la patch $`j`$.

Ranking delle pagine (late interaction calibrata e pesata con IDF; un token la cui pagina
migliore resta sotto 0,2 è considerato senza corrispondenza e contribuisce 0):

```math
S(q,d) = \sum_i \mathrm{idf}_i \cdot \mathrm{clip}\!\left(\frac{m_{i,d} - \nu}{\max_{d'} m_{i,d'} - \nu},\ 0,\ 1\right)
```

Heatmap di attivazione (AND morbido, media armonica pesata sui token informativi):

```math
H_j = \left(\frac{\sum_i w_i \,(\varepsilon + n_{ij})^{-1}}{\sum_i w_i}\right)^{-1} - \varepsilon,
\qquad n_{ij} = \frac{\max(0,\ s_{ij} - \nu)}{\max_k s_{ik} - \nu}
```

Ranking delle cause radice ed evidenza di uno span (noisy-OR):

```math
\text{score}(s) = \max_a \ \text{sev}(a) \cdot \text{reach}(s \to a) \cdot e(s) \cdot o(s),
\qquad e(s) = 1 - \prod_k \bigl(1 - e_k(s)\bigr)
```

Decisione dello shield e regola di entropia del loop breaker:

```math
\text{risk} = 1 - \prod_f \bigl(1 - \text{sev}_f\bigr) \ge 0.8 \Rightarrow \text{block},
\qquad -\sum_a \hat p(a) \log_2 \hat p(a) \le 1 \ \wedge\ \text{novelty} \le 0.2 \Rightarrow \text{halt}
```

### A · Encoder spaziale a patch e late interaction (`vision/`)

* **Grafo di layout** (`patch_encoder.build_layout_graph`). Le regioni foglia sono i nodi. Ogni
  nodo riceve archi di contesto dalla sua intestazione di riga o etichetta (peso 1,0),
  dall'intestazione di colonna (1,0), dal titolo di sezione più vicino sopra di esso nella stessa
  colonna (1,0), dal vicino immediatamente a sinistra/sopra (0,3 se quel vicino è un descrittore
  o il nodo è una cella), dalla didascalia del contenitore (0,3) e dai connettori dello schema
  (0,6). Le intestazioni sorelle non si descrivono a vicenda. Il vettore della regione è
  `normalize(own + λ(n)·ctx)`, dove `ctx` satura a norma unitaria e `λ` diminuisce al crescere
  del testo proprio della regione. Un numero isolato vive quindi delle sue intestazioni, mentre
  un'etichetta ricca descrive soprattutto se stessa.
* **Feature testuali**: feature hashing con segno (BLAKE2b, 976 dimensioni). Ogni parola è
  distribuita su 8 slot e ogni trigramma di caratteri su 2, così le collisioni dell'hash si
  comportano come piccolo rumore gaussiano (σ ≈ 1/√976) invece che come rari picchi grandi. I
  numeri brevi si legano al sostantivo precedente: "line 1" diventa `line-1`.
* **Patch**: i vettori delle regioni vengono proiettati sulla griglia 32×32, pesati con
  `|r∩p| / min(|r|,|p|)`. I paragrafi lunghi sono divisi in fasce che seguono il flusso delle
  parole, così ogni patch porta le sue parole locali. Ogni patch riceve anche 32 dimensioni visive
  (istogramma della tinta a 12 bin centrato sul rosso, frazioni acromatiche, istogramma a 8 bin
  dell'orientazione del gradiente, densità dei bordi, momenti di luminanza e colore) e 16
  dimensioni posizionali. Parole di query come *red*, *orange*, *top* e *bottom* vengono mappate su
  questi sottospazi.
* **Quantizzazione**: una rotazione ortogonale casuale con seed fisso (preserva i prodotti
  scalari e distribuisce l'energia in modo uniforme), poi int8 (absmax per vettore), 2-bit
  (livelli gaussiani di Lloyd-Max ±0,45σ/±1,51σ) e pacchetti di segni a 1 bit valutati in modo
  asimmetrico come `⟨q, sign(p)⟩·‖p‖·√(π/2D)`. La ricerca avviene in due stadi: una shortlist a
  1 bit, poi un rerank alla precisione richiesta.
* **Punteggio di pagina**: MaxSim con calibrazione per token sul corpus. Ogni token di query
  contribuisce `idf · clip((m − noise)/(best − noise), 0, 1)`, così un codice di componente raro
  dentro un'etichetta lunga conta quanto una parola comune trovata in un'intestazione di una sola
  parola. Il MaxSim ColBERT grezzo è mantenuto come ablazione.
* **Heatmap**: un AND morbido (media armonica pesata). La similarità di ogni token è normalizzata
  rispetto alla sua patch migliore e ripulita dal rumore. I token che corrispondono meglio al
  titolo della pagina contano come *ambito* della pagina e sono esclusi dal grounding. Se nessuna
  patch soddisfa tutti i token, il token che blocca la congiunzione viene scartato in modo greedy.
  Vince così la cella in cui si incontrano i vincoli di riga *e* di colonna, non l'intestazione
  di colonna.
* **Bounding box**: le proposte dello structure layer sono valutate con
  `0.75·soft-AND(embedding della regione) + 0.25·calore medio delle patch`. I descrittori
  (intestazioni, etichette) le cui parole sono semplicemente ripetute dalla query vengono
  penalizzati, perché indicano la risposta invece di contenerla. Senza structure layer, il box è
  la componente connessa sopra soglia attorno al picco della heatmap.

### B · Keyframe, sintomi e DAG causale (`vision/phash_dedup.py`, `correlator/`)

* **Keyframe**: una riduzione a box intera per frame (import Pillow senza copia), poi un dHash a
  64 bit e una griglia 18×32 di medie per tessera. La soglia di cambiamento si adatta al rumore
  (`max(0.9, 8·σ̂)`, con σ̂ ricavata dalla mediana delle differenze assolute). Le sequenze di
  cambiamenti confinate al massimo al 3% delle tessere sono trattate come **animazioni**, e se ne
  conservano solo l'inizio e la fine stabilizzata. Le stesse sequenze alimentano il rilevatore di
  spinner bloccati.
* **Sintomi dai pixel**: nuove regioni rosse sature e persistenti (banner di errore), regioni
  ambra (toast di avviso), periferia oscurata attorno a un box centrale luminoso (modali),
  tessere con texture che diventano piatte e restano tali (regioni vuote) e animazioni più lunghe
  di 3 s (stalli). Sintomi dal DOM: classi di errore e `role=alert`, testo avvelenato (`NaN`,
  `undefined`…), indicatori di attesa che non si chiudono mai e contenitori svuotati.
* Gli archi del **DAG** vanno dalla causa all'effetto e rispettano tutti il tempo:
  * span figlio → span padre (0,9)
  * fine dello span client → mutazione del DOM. Il peso è l'affinità tra il nodo DOM (selettore,
    attributi, testo) e l'intera traccia a valle della richiesta. Un banner "Payment failed" si
    lega alla traccia che ha toccato `payment-service`. Un cambiamento del DOM che sembra un
    errore e una risposta fallita sono inoltre coerenti tra loro.
  * inizio della richiesta → indicatore di attesa
  * mutazione del DOM → sintomo (sovrapposizione dei bbox entro la finestra visiva)
  * richiesta ancora in corso → stallo
  * un debole fallback puramente temporale
* **Ranking**: `severity(a) · reach(s→a) · evidence(s) · origin(s)`.
  * L'evidenza è un noisy-OR di classe dello status, flag di errore, log ERROR/FATAL (più alta se
    c'è il testo di un'eccezione), z-score robusto della latenza rispetto agli span della stessa
    operazione, deriva dello schema della risposta rispetto allo schema maggioritario
    dell'operazione e campi nulli.
  * L'origine vale 0,3 quando il fallimento dello span è interamente spiegato da un figlio con la
    stessa classe di evidenza (il 502 sopra un 503), e 0,15 quando un retry ha recuperato il
    fallimento.

### C · Shield e loop breaker (`security/`)

* **Unicode**: contrabbando tramite caratteri tag (decodificati e rianalizzati); caratteri a
  larghezza zero dentro le parole, con esenzione per lo ZWJ delle emoji, lo ZWNJ indiano/persiano
  e un BOM iniziale; codifiche binarie a larghezza zero (decodificate); override bidirezionali
  (severità alta nel testo da sinistra a destra, bassa nel testo RTL); contrabbando di byte tramite
  variation selector; parole latine che contengono sosia cirillici/greci (decodificate con una
  mappa di caratteri confondibili). Unità come μs e kΩ non vengono segnalate.
* Le **regole lessicali** girano su più viste normalizzate (leetspeak, lettere spaziate, markdown
  rimosso) dietro filtri letterali economici. Coprono l'override delle istruzioni (inglese,
  francese, spagnolo, tedesco, italiano, portoghese), la riassegnazione del ruolo, i compiti
  iniettati, l'occultamento ("don't tell the user"), l'esfiltrazione e la divulgazione di segreti,
  l'aggiramento delle protezioni, le pretese di autorità, gli imperativi rivolti all'IA e l'invio
  di dati verso un indirizzo email.
* **Contesto**: lo stesso costrutto dentro un esempio di codice delimitato o un codice inline pesa
  molto meno. Sono esenti i consigli in forma negativa ("do not email credentials") e i rimandi
  alla struttura del documento ("ignore the previous *section's* instructions"). I canali di
  metadati (alt text, aria, EXIF, DOM nascosto) ricevono un aumento di severità.
* **Contrabbando codificato**: le sequenze base64, esadecimali e percent-encoded ad alta entropia
  vengono decodificate (ricorsivamente, fino a profondità 3). Solo il testo in chiaro che sembra
  testo viene rianalizzato, quindi hash, firme JWT e immagini passano.
* **Dirottamento dei tool**: le invocazioni imperative, JSON o sintattiche di tool sensibili hanno
  severità alta solo quando la frase contiene un bersaglio (email, URL, `~/.ssh`, `.env`) o
  segreti, oppure si rivolge all'IA. Vengono rilevati anche URL di esfiltrazione tramite immagini
  markdown con segnaposto di template, e pipe-to-shell fuori dagli esempi di codice.
* **Decisione**: il rischio è il noisy-OR delle severità dei rilevamenti; si blocca quando il
  rischio è ≥ 0,8.
* **Spotlighting**: delimitatori con nonce che il contenuto non può falsificare, più datamarking
  (spazi sostituiti da `ˆ`) o base64. `ToolCallGuard` blocca le chiamate a tool sensibili i cui
  argomenti copiano contenuto non fidato, o che il contenuto non fidato ha richiesto.
* **Loop breaker**: gli argomenti vengono canonicalizzati (rimossi UUID, timestamp, epoch, id
  esadecimali, nonce) e trasformati in embedding. Una chiamata si unisce a un *prototipo di
  azione* esistente quando il coseno è ≥ 0,85, oppure ≥ 0,45 se ha anche prodotto la stessa
  osservazione: così vengono riconosciuti i retry parafrasati. Il loop breaker si ferma (a) su un
  ciclo di periodo ≤4 ripetuto 3 volte con osservazioni identiche, oppure (b) quando l'entropia
  delle azioni è ≤ 1 bit con ≤ 20% di coppie (azione, osservazione) nuove su 10 passi. La
  paginazione e il polling con risultati che cambiano contano come progresso.

---

## Struttura del repository

```
src/omnitrace/
  config.py                     tutti i parametri (dataclass frozen; override via env OMNITRACE_DB / OMNITRACE_GRID)
  textfeat.py                   tokenizer + feature hashing con segno condivisi da tutti i sottosistemi
  vision/phash_dedup.py         dHash / pHash, estrattore di keyframe consapevole dei cicli
  vision/patch_encoder.py       grafo di layout, encoder a patch, quantizzatori int8 / 2-bit / 1-bit
  vision/spatial_maxsim.py      VisualIndex (SQLite), MaxSim, heatmap, grounding dei bbox, ritagli
  vision/vlm_adapters.py        adattatori NullVLM / Ollama / compatibili OpenAI, answer_with_citation
  correlator/timeline_dag.py    modello TraceBundle, rilevatori di sintomi visivi + DOM, DAG causale
  correlator/rca_engine.py      evidenza, fattori di origine, ranking, RCAReport
  security/injection_shield.py  InjectionShield, spotlighting, ToolCallGuard
  security/loop_breaker.py      LoopBreaker
  interfaces/{mcp_server,cli,tui}.py
evals/
  benchmark_suite.py            i quattro benchmark, baseline, ablazioni, scrittura dei README (en + it)
  repeat_runs.py                esecuzioni ripetute con gate sul carico della macchina + verdetto di ripetibilità
  baselines.py                  BM25 a chunk piatti, BM25 per regione
  datasets/                     generatori (layouts, ui_sim, traces, injection_corpus) + dati esportati
  latest_report.json            ultimo report completo dei benchmark
  repeatability_report.json     valori per esecuzione, intervalli, verdetto
tests/                          63 test (quello con VLM reale parte se Ollama ha un modello visivo)
```

---

## Limiti

* **Dati sintetici, scritti dallo stesso autore.** I generatori, le query, i payload di iniezione
  e le euristiche sono stati scritti tutti dalla stessa persona. Gli split held-out cambiano i
  seed casuali, e il set held-out delle iniezioni è stato scritto dopo aver congelato i
  rilevatori. Anche così, nessuno di questi è un dato reale indipendente, e il 100% qui non va
  letto come 100% sui vostri documenti. Validate sul vostro corpus prima di farci affidamento.
* **Nessuna semantica oltre l'hashing lessicale.** I sinonimi ("amperage" per "current") non
  corrispondono. Un encoder appreso (ad es. un vero checkpoint ColPali) colmerebbe questa lacuna;
  indice, quantizzatori e logica di grounding resterebbero invariati.
* **I box precisi richiedono uno structure layer.** Dai soli pixel, il box della heatmap ha una
  granularità di 32 px (vedi l'ablazione). I documenti puramente raster richiederebbero un
  passaggio di OCR o VLM per produrre le regioni.
* **La RCA presuppone span client lato browser e record del DOM** con bounding box, come quelli
  prodotti dall'instrumentazione web di OpenTelemetry più un mutation observer. Senza il video, i
  due sintomi visibili solo come modale sono invisibili (vedi l'ablazione).
* **Lo shield è un sistema a regole.** Segnalerà documenti *sulla* prompt injection che citano
  attacchi alla lettera, e un attaccante determinato può trovare formulazioni fuori dalle sue
  regole. È un pre-filtro da affiancare allo spotlighting e al controllo delle chiamate ai tool,
  non un sostituto di un design dei tool basato sul minimo privilegio.
* Le latenze sono misurate con CPython a singolo thread sulla macchina indicata nell'intestazione
  dei risultati. Dipendono dall'hardware e dal carico in background; il blocco sulla ripetibilità
  ne mostra la variabilità su esecuzioni a macchina scarica.

## Licenza

[MIT](LICENSE) © 2026 Giuseppe Gambino

## Supporta il progetto

Se questo progetto ti è utile, puoi sostenerne lo sviluppo con una donazione su PayPal:

<a href="https://paypal.me/giuseppegambino92"><img src="https://www.paypalobjects.com/it_IT/IT/i/btn/btn_donateCC_LG.gif" alt="Fai una donazione con PayPal"></a>
