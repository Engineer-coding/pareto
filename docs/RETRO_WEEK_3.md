# Pareto — Hafta 3 Retrospektifi

> _Semantic cache haftası. Pareto'nun "cost-optimized" iddiasının ilk
> somut kanıtı. Bu hafta öğrendiğim en önemli şey: cache bedava değil.
> Threshold bir kalite-tasarruf dengesidir, ve false hit gerçek bir
> tehlikedir. Sayıyla dürüstlük bu hafta sınandı._

**Tarih aralığı:** Hafta 3, 7 gün
**Ana hedef:** Semantic cache (LRU + embedding similarity) + cache observability
**Sonuç:** Her iki hedef tutturuldu. Bonus: false hit fenomeni keşfedildi ve sayısallaştırıldı.

---

## 1. Bir Bakışta

Hafta 3 sonunda Pareto:

- **Custom LRU cache** — `OrderedDict` tabanlı, `peek`/`get`/`touch` ayrımı, hit/miss counters
- **SemanticCache** — embedding cosine similarity threshold, numpy-vectorized lookup
- **Composite cache key** — (embedding, retriever, top_k, model) filter
- **Chunk-level invalidation** — deterministic chunk ID'leri ile staleness check
- **NaiveRAG cache entegrasyonu** — fast path, embed-once-use-twice
- **Pickle persistence** — atomic write, graceful load, version guard
- **`pareto ask` cache flag'leri** — `--no-cache`, `--cache-threshold`, `--cache-capacity`, `--cache-path`
- **Cache observability** — SQLite schema migration (cache_hit, cache_similarity), `pareto stats` cache metrics, savings projection
- **45-query cache test set** — 30 original + 15 paraphrase (1 Türkçe cross-lingual)
- **Cost projection dokümanı** — provider bazında, quality-adjusted

Bir yeni modül: `pareto/cache/` (lru.py + semantic_cache.py).

---

## 2. Hafta 3 Baseline Rakamları (Measured)

### Cache Hit Rate (45-query test set, 30 original + 15 paraphrase)

| Threshold | Hit Rate | Keyword Coverage | Koşum süresi | Verdict |
|---|---|---|---|---|
| 0.85 | **71.1%** | 0.634 | 11 dk | Çok agresif — false hit'ler kaliteyi bozuyor |
| 0.92 | **24.4%** | 0.878 | 33 dk | Güvenli, hafif sızıntı |
| ~0.93 (target) | ~18-22% (est.) | ~0.90+ (est.) | — | Cumartesi tuning hedefi |

### Speedup (latency)

| Ölçüm | Değer |
|---|---|
| Cache hit (CLI, persisted) | 76 ms |
| Fresh LLM call | 96,000 ms |
| **CLI speedup** | **1264x** |
| **REPL speedup** | **1467x** |
| **Avg (miss vs hit, by cache_hit)** | **488x** |

### Cost Projection (GPT-4o, 100K query/gün, conservative 24.4% hit)

| Model | Annual (no cache) | Saved @24.4% |
|---|---|---|
| GPT-4o | $93,075 | $22,710 |
| Claude Sonnet 4.6 | $122,640 | $29,924 |
| Claude Haiku 4.5 | $32,704 | $7,980 |

---

## 3. Günlük Akış

### Pazartesi — Cache Modülü İskeleti
- `pareto/cache/lru.py` — Generic `LRUCache[K, V]`, OrderedDict tabanlı
- `pareto/cache/semantic_cache.py` — `SemanticCache` + `CacheEntry` + `CacheHit`
- Numpy-vectorized cosine similarity lookup
- Composite filter (retriever + top_k + model), staleness check
- 8 smoke test geçti
- **Önemli veri:** "What is GDPR?" ↔ "Explain GDPR" similarity = 0.9191 (threshold 0.92'nin altında)

### Salı — RAG Entegrasyonu + CLI Persistence
- NaiveRAG `cache` parametresi, fast path (`_cache_hit_response`)
- Embed-once-use-twice tasarımı
- **REPL test: 1467x speedup**
- Pickle persistence (atomic write, graceful load, SAVE_FORMAT_VERSION)
- `pareto ask --no-cache/--cache-threshold/--cache-capacity/--cache-path`
- **CLI test: 1264x speedup**, paraphrase attribution ("Reusing cached answer from...")

### Çarşamba — Cache Observability
- SQLite idempotent schema migration (cache_hit + cache_similarity columns)
- `log()` cache info `response.extra`'dan otomatik extract
- `aggregate()` cache metrics, `total_savings()` phantom cost
- `slow_queries()` cache-hit exclusion
- `pareto stats` overall Cache section + Savings tablosu + `--by cache_hit`
- **288 saniye tasarruf rakamı** ilk kez görünür

### Perşembe — Cache Benchmark + Cost Projection
- A: 45-query test set (`queries_with_dupes.yaml`)
- B: İki threshold koşumu (0.92 + 0.85)
  - **False hit fenomeni keşfedildi**
  - Retrieval metrics artifaktı tespit edildi
  - Threshold-quality tradeoff sayısallaştırıldı
- C: `docs/COST_PROJECTION.md` — quality-adjusted projeksiyon

### Cuma — Bu Retro + README v1.6

### Cumartesi — Threshold Tuning (planlı)
- 0.90/0.93/0.94/0.95 ara noktaları
- Optimal threshold kesinleştirme (false hit sıfırlanan + makul hit rate)

### Pazar — Hafta 4 Hazırlık
- Adaptive query router tasarımı

---

## 4. Production-Grade Bug Fix'ler (Hafta 3)

### Bug 10: Python Truthiness Tuzağı
**Sebep:** NaiveRAG `__repr__`'da `if self.cache` kullandım. Python'da `__len__` tanımlı container 0 length'inde `bool() == False` döner. Boş cache "None" gibi göründü.
**Çözüm:** `is not None` (idiomatic) + cache class'larına `__bool__ = True` (defensive).
**Öğreti:** Empty list, empty dict, custom `__len__=0` — hepsi truthiness'te None'la karışır. İki katmanlı savunma.

### Bug 11: Duplicate Retriever Build Bloğu
**Sebep:** `pareto benchmark` komutunda retriever build bloğu kopyala-yapıştır ile iki kez yer alıyordu. İkincisi birincinin üzerine yazıyordu (zararsız ama gereksiz, ilkinin comment indent'i de bozuktu).
**Çözüm:** Tek temiz blok.
**Öğreti:** Salı'da `--retriever` flag eklerken oluşmuş. Code review veya linter yakalayabilirdi.

---

## 5. Hafta 3'te Üretilen Kalıcı Veriler

`benchmarks/` ve `docs/` altında, git'e committed:

- `benchmarks/results/cache_hit_092.json` — conservative threshold koşumu
- `benchmarks/results/cache_hit_085.json` — aggressive threshold koşumu
- `benchmarks/queries/queries_with_dupes.yaml` — 45-query cache test set
- `docs/COST_PROJECTION.md` — provider bazında tasarruf projeksiyonu
- `benchmarks/results/cache.pkl` (gitignored) — persistent cache state

---

## 6. Açık Backlog

### Cache İyileştirme
- [ ] Cumartesi threshold tuning (0.90/0.93/0.94/0.95, false hit rate ölçümü)
- [ ] Benchmark retrieval metrics cache-hit exclusion (cache ON koşumunda artifakt)
- [ ] Cache entry'de retrieved object saklamak (retrieval metrics doğru hesabı için, memory cost)
- [ ] `pareto cache evict-stale` CLI komutu (proaktif invalidation)

### Hafta 4 Hazırlığı
- [ ] Adaptive query router tasarımı
- [ ] Router signals: query length, language, NO_ANSWER probability
- [ ] Cache + router etkileşimi (cache hit'te router atlanır mı?)

### Görünürlük
- [ ] README v1.6 cache section (Cuma B)
- [ ] LinkedIn post #2 + #3 (hâlâ unposted, Hafta 1-2'den)

---

## 7. Karşılaşılan Zorluklar — Tekrar Yapsam Neyi Farklı Yapardım?

**1. Retrieval metrics çöküşü beni bir an paniğe soktu.** Cache benchmark'ında hit@k 100%'den 27%'ye düştü. İlk içgüdü "retrieval bozuldu" oldu. Sonra `retrieved: []` gördüm — cache hit retrieval'ı bypass ediyor, metrik sıfır sayıyor. **Öğreti:** Şaşırtıcı bir metrik düşüşünde önce ölçüm mekaniğini kontrol et, sonra gerçek başarısızlık varsay. Artifakt vs gerçek failure ayrımı.

**2. Threshold default'unu 0.92 seçtim ama paraphrase recall'u düşük.** Pazartesi "Explain GDPR" 0.9191 ölçtüm — yakın miss. 0.90 seçseydim demo paraphrase'leri hit olurdu. Ama Perşembe'de 0.85 koşumunda false hit fenomenini görünce anladım: düşük threshold tehlikeli. **Öğreti:** Threshold seçimi tek bir paraphrase'e göre değil, false hit rate'e göre yapılmalı. Cumartesi sistematik test.

**3. Benchmark CLI'da cache integration yaparken duplicate retriever bloğunu fark etmedim.** Salı'da eklediğim kopyala-yapıştır hatası Perşembe'de patch verirken ortaya çıktı. **Öğreti:** Her CLI patch sonrası `pareto --help` + sanity koşumu — Bug 11 daha erken yakalanırdı.

**4. False hit forensics'i daha erken yapabilirdim.** 0.85 koşumunda 32 hit görünce "harika!" dedim, sonra keyword coverage 0.634'ü görünce durdum. 32 hit > 15 paraphrase matematiği false hit'i kanıtladı. **Öğreti:** Hit rate tek başına yanıltıcı; her zaman quality metric ile birlikte oku.

---

## 8. Hafta 4 Hazırlığı — Adaptive Query Router

### Vizyon
Şu ana kadar her sorgu aynı pipeline'dan geçiyor: hybrid retrieval + LLM. Ama sorgular farklı:
- Basit factual sorgu → ucuz model yeter
- Karmaşık multi-hop → güçlü model gerek
- NO_ANSWER ihtimali yüksek → BM25-only (Hafta 2 bulgusu: refusal accuracy 100%)
- Cache hit → hiç router gerek yok

### Router Signals (tasarım)
- Query length (kısa → basit?)
- Language detection (Türkçe → hybrid kesin)
- NO_ANSWER probability (Hafta 2'deki refusal pattern)
- Retriever selection (sorgu tipine göre dense/bm25/hybrid)
- Model tiering (basit → mini, karmaşık → full)

### Cache + Router Etkileşimi
Cache lookup router'dan **önce** mi sonra mı? Önce mantıklı — cache hit varsa router gereksiz. Bu Hafta 4'ün ilk tasarım kararı.

### Geçilmesi Gereken Bar
Hafta 3 cache + Hafta 2 hybrid baseline korunmalı. Router **kaliteyi düşürmeden** maliyeti azaltmalı (model tiering ile).

---

## 9. Hafta 3'ün Mülakat Anlatım Kartları (Yeni 25 Kart, 27-51)

Hafta 1'in 8 + Hafta 2'nin 18 kartına ek olarak (toplam 51).

**En güçlü 12 (demo/mülakat öncelikli):**

| # | Konu | Anahtar Mesaj |
|---|---|---|
| 28 | Cache Invalidation via Deterministic IDs | Hafta 1'in idempotent ID'leri Hafta 3'te faiz ödedi |
| 32 | Phantom Cost | "Ödemediğin ne kadar" first-class metric |
| 33 | Python Truthiness Tuzağı | `__len__=0` container False döner; `is not None` |
| 39 | Idempotent Schema Migration | PRAGMA table_info; eski DB zarif yükselir |
| 43 | NULL-Aware Migration | Eski rows NULL; CASE WHEN ile "miss" sayılır |
| 46 | Measurement Artifact vs Real Failure | Metrik düşüşünde önce mekaniği kontrol et |
| 47 | Threshold-Quality Tradeoff | 0.85: 71% hit ama kw_cov 0.634 (false hits) |
| 48 | False Hit Forensics | 32 hit > 15 paraphrase = 17 false; query-level kanıt |
| 50 | Quality-Adjusted Savings | False hit "tasarruf" değil, kalite regresyonu |
| 38 | 1264x Live Speedup | Demo'da iki ardışık ask, ikincisi instant |
| 41 | 488x Average Speedup | by cache_hit agregasyon: 55s vs 113ms |
| 51 | Latency Savings ≠ Cost Savings | Local model bile cache'ten faydalanır (latency) |

**Ek 13 kart (ikincil ama değerli):**
27 (Custom LRU), 29 (Threshold+Stale Composite), 30 (Threshold Empirical), 31 (Cache Fast Path), 34 (Atomic Write), 35 (Graceful Loading + Version Guard), 36 (CLI Persistence Decision), 37 (Reusing-Cached-Answer Demo Gold), 40 (Phantom Cost First-Class), 42 (Multi-Source Aggregation), 44 (Realistic Paraphrase Distribution), 45 (Cross-Lingual + Trick Paraphrases), 49 (Cache Speeds Up Benchmarks).

---

## 10. Hafta 3'ün Kişisel Notları

Bu hafta toplam ~14-16 saat. Hafta 2 ile benzer yoğunluk. Cache derin bir konu — basit görünüyor ("cevabı sakla, tekrar ver") ama invalidation, threshold tuning, false hit, observability katmanları açıldıkça zenginleşti.

**Hafta 3'ün asıl dersi — cache bedava değil.** Hafta 1'de "naive baseline kurdum" yetiyordu. Hafta 2'de "yeni layer baseline'ı geçti mi?" sorusu vardı. Hafta 3'te yeni bir boyut açıldı: **"bu optimizasyon kaliteyi düşürüyor mu?"** Cache tasarruf sağlıyor ama yanlış threshold'da yanlış cevap veriyor. Bu tradeoff'u sayısal olarak görmek — false hit forensics — bu haftanın zirvesiydi.

**En çok keyif aldığım kısım:** Perşembe B'de 0.85 koşumunda 32 hit görünce "harika!" deyip, sonra keyword coverage 0.634'ü görünce durmam. "32 hit ama 15 paraphrase var, 17 fazla nereden geldi?" sorusu beni false hit forensics'e götürdü. Retrieval misses listesinde legal-002...006'nın hepsinin GDPR cevabı aldığını görmek — **veri kendi hikayesini anlattı.**

**En çok zorlandığım kısım:** Retrieval metrics çöküşünü artifakt olarak tanımak. Bir an "tüm cache implementasyonum retrieval'ı bozdu mu?" paniği yaşadım. `retrieved: []` görene kadar. Soğukkanlı kalıp ölçüm mekaniğini kontrol etmek — panik yerine analiz — mühendisliğin gerçek hali.

**Hafta 4'e hazır mıyım?** Evet. Cache + observability altyapısı router için zemin. Router cache lookup'tan sonra devreye girecek (cache hit'te router atlanır). Model tiering ile maliyeti daha da düşüreceğiz. Risk: router'ın yanlış model seçmesi (kaliteyi düşürür) — yine bir tradeoff.

**Demo açısından Hafta 3 altın:** "İkinci kez ücretsiz" iddiası artık canlı. Üç komutluk şov (ask, ask aynı, ask paraphrase) + cost projection tablosu + quality-adjusted dürüstlük. CFO'ya gösterilecek tek slide hazır.

---

_Last updated: Week 3 Friday, by Latif Şimşek._