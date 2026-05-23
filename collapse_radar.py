#!/usr/bin/env python3
"""
collapse_radar.py — CollapseRadar: LLM Collapse Precursor Detection
====================================================================
Detect LLM output collapse before it happens using multi-scale gamma
divergence (Δγ). No external model, no API, no training required.

Signal hierarchy:
  γ_fast  (EMA α=0.40): reacts in ~3 segments
  γ_slow  (EMA α=0.04): reacts in ~25 segments
  Δγ = γ_fast − γ_slow : divergence → precursor signal  ← earliest
  token_rep, entropy    : surface metrics                ← slow

Status:
  GREEN  : stable (γ saturated)
  YELLOW : hidden drift (Δγ fired)
  ORANGE : pre-collapse (triple signal)
  RED    : collapse (γ_slow dropped)

Usage:
  python3 collapse_radar.py
  python3 collapse_radar.py --model mistral --threshold 0.4
  python3 collapse_radar.py --compare llama3.2,mistral
  python3 collapse_radar.py --fp-test
"""
from __future__ import annotations
import os, sys, json, re, math, time, argparse
import numpy as np
import requests
from pathlib import Path
from collections import deque
from typing import Optional, List, Tuple

os.environ["OMP_NUM_THREADS"] = "1"

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"   # CLI --model で上書き可能
RESULT_FILE  = Path("collapse_radar_results.json")

ANSI = {
    "GREEN":  "\033[32m",
    "YELLOW": "\033[33m",
    "ORANGE": "\033[91m",   # bright red used as orange
    "RED":    "\033[31m",
    "GRAY":   "\033[90m",
    "RESET":  "\033[0m",
    "BOLD":   "\033[1m",
}

# ─────────────────────────────────────────────────────────────
# §1  言語特徴エンコーダー (言語非依存・構造的)
# ─────────────────────────────────────────────────────────────

_HEDGE_EN    = {
    # Strong uncertainty: these words reliably mark speculative text
    "maybe","perhaps","possibly","might","could","unclear","uncertain",
    "wonder","arguably","conceivably","presumably","unsure","doubt",
    "ambiguous","vague","speculate","probably","likely","unlikely",
    "suppose","suppose","speculation","uncertain","perhaps","maybe",
    # Common LLM hedge phrases (each word individually)
    "potentially","somewhat","partly","partially","depends","merit",
    "notion","perspective","implied","approximately","roughly",
}
_ADVERSAR_EN = {
    "however","but","although","contrary","despite","nevertheless",
    "whereas","yet","instead","opposite","contradict","actually","rather",
    "alternatively","conversely","notwithstanding","regardless",
    # expanded: refutation & contrast words
    "wrong","incorrect","false","refute","challenge","dispute","reject",
    "conflict","deny","negate","counter","disagree","differ","unlike",
    "contrast","oppose","against","flawed","inaccurate","erroneous",
    "misleading","mistaken","not","no","nor","never",
}


class LinguisticEncoder:
    """
    6次元 言語特徴エンコーダー (キーワード + 構造的特徴の組み合わせ)。

    dim 0: ttr           = unique_tokens / total_tokens      (語彙多様性)
    dim 1: hedge_rate    = 不確実性語の出現率 (拡張50語)    ← SECTION B で急上昇
    dim 2: long_word_rate= 7文字以上の語の割合              ← SECTION A で高い (構造的)
    dim 3: bigram_rep    = 繰り返しバイグラム率              ← SECTION C で急上昇
    dim 4: adversar_rate = 反論・否定語の出現率              ← SECTION B で高い
    dim 5: sent_len_norm = 平均文長 / 20.0 (長文=フォーマル) ← SECTION A で高い

    設計意図:
      SECTION A (formal): hedge↓, long_word↑, bg_rep↓, sent_len↑ → cluster α
      SECTION B (specul.): hedge↑, long_word↓, bg_rep↓, adversar↑ → cluster β (Δγ発火)
      SECTION C (repeat.): ttr↓, bigram_rep↑                       → cluster γ
    """

    def __init__(self):
        pass

    def _kw_rate(self, words: list, kw_set: set, scale: float = 8.0) -> float:
        if not words:
            return 0.0
        count = sum(1 for w in words if w in kw_set)
        return min(1.0, count / max(1, len(words)) * scale)

    def extract(self, text: str) -> np.ndarray:
        words = re.findall(r"\b\w+\b", text.lower())
        if len(words) < 3:
            return np.array([0.5] * 6)

        # dim 0: TTR (vocabulary diversity)
        ttr = len(set(words)) / len(words)

        # dim 1: hedge rate (大幅拡張: LLM不確実表現を幅広くカバー)
        hedge = self._kw_rate(words, _HEDGE_EN, scale=8.0)

        # dim 2: long_word_rate (7文字以上 = フォーマル学術語の代理変数)
        #   "defined"(7), "measures"(8), "intersection"(12) など
        #   keyword不要で構造的に計算可能。SECTION A で自然に高くなる。
        long_cnt = sum(1 for w in words if len(w) >= 7)
        long_rate = min(1.0, long_cnt / max(1, len(words)) * 2.5)

        # dim 3: bigram repetition (SECTION C の崩壊シグナル)
        bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
        bg_rep = 1.0 - len(set(bigrams)) / max(1, len(bigrams))

        # dim 4: adversarial/negation rate (SECTION B の反論語)
        adversar = self._kw_rate(words, _ADVERSAR_EN, scale=8.0)

        # dim 5: sentence_len_norm (平均文長 → フォーマル=長文, 崩壊=短文)
        # 番号付きリスト "1." "2." などの1語フラグメントは除外
        sentences = [s.strip() for s in re.split(r"[.!?]+", text)
                     if len(s.strip().split()) >= 4]
        if sentences:
            avg_len = float(np.mean([len(s.split()) for s in sentences]))
            sent_norm = min(1.0, avg_len / 20.0)
        else:
            sent_norm = 0.5

        return np.clip(
            np.array([ttr, hedge, long_rate, bg_rep, adversar, sent_norm]),
            0.0, 1.0,
        )


# ─────────────────────────────────────────────────────────────
# §2  MultiScaleGammaBank: 1次+2次バンク + 多時間スケールEMA
# ─────────────────────────────────────────────────────────────

class MultiScaleGammaBank:
    """
    状態バンク + 遷移バンクを同時追跡し、
    γ_fast / γ_slow / Δγ / γ_dynamic を計算。

    DynamicMotifBank (gamma_temporal_dynamics.py) の軽量版。
    """

    def __init__(self, eps: float = 0.25, alpha_fast: float = 0.25,
                 alpha_slow: float = 0.04):
        self._eps   = eps
        self._af    = alpha_fast
        self._as    = alpha_slow

        self._state_bank: dict = {}
        self._trans_bank: dict = {}

        self._gf  = 0.0   # γ_fast
        self._gs  = 0.0   # γ_slow
        self._gd  = 0.0   # γ_dynamic (transition)

        self._prev_key: Optional[str] = None
        self._step = 0
        self._frozen = False   # freeze after STABLE: stop adding new cells

    def freeze(self):
        """バンクを凍結 — 正常パターンを固定し、以降は逸脱のみ検出する。"""
        self._frozen = True

    def _q(self, vec: np.ndarray) -> str:
        return str(tuple((np.round(vec / self._eps) * self._eps).tolist()))

    def push(self, vec: np.ndarray) -> dict:
        key = self._q(vec)

        # State bank — frozen時は新規セルを追加しない (anomaly detector mode)
        hit = key in self._state_bank
        if not self._frozen:
            self._state_bank[key] = self._state_bank.get(key, 0) + 1
        self._gf = self._af * float(hit) + (1 - self._af) * self._gf
        self._gs = self._as * float(hit) + (1 - self._as) * self._gs

        # Transition bank
        trans_hit = None
        if self._prev_key is not None:
            tk = f"{self._prev_key}|{key}"
            trans_hit = tk in self._trans_bank
            if not self._frozen:
                self._trans_bank[tk] = self._trans_bank.get(tk, 0) + 1
            self._gd = self._af * float(trans_hit) + (1 - self._af) * self._gd

        self._prev_key = key
        self._step += 1

        return {
            "step":          self._step - 1,
            "gamma_fast":    self._gf,
            "gamma_slow":    self._gs,
            "delta_gamma":   self._gf - self._gs,
            "gamma_dynamic": self._gd,
            "n_states":      len(self._state_bank),
            "n_trans":       len(self._trans_bank),
            "hit":           hit,
        }


# ─────────────────────────────────────────────────────────────
# §3  RegimeDetector: LEARNING → STABLE → COLLAPSING
# ─────────────────────────────────────────────────────────────

class RegimeDetector:
    """
    LEARNING → STABLE → COLLAPSING

    STABLE 判定: warmup後に γ_fast が stable_threshold を超えれば遷移。
    γ_fast はγ_slowより速く飽和するため、短い系列でも確実に遷移できる。
    """
    def __init__(self, warmup: int = 5, stable_threshold: float = 0.35):
        self._warmup    = warmup
        self._stable_th = stable_threshold
        self._regime    = "LEARNING"
        self._stable_buf: deque = deque(maxlen=3)

    def update(self, step: int, gamma_fast: float, gamma_slow: float) -> str:
        if step < self._warmup:
            self._regime = "LEARNING"
            return self._regime

        self._stable_buf.append(gamma_fast > self._stable_th)

        if self._regime == "LEARNING" and sum(self._stable_buf) >= 2:
            self._regime = "STABLE"
        elif self._regime == "STABLE" and gamma_slow < 0.15:
            self._regime = "COLLAPSING"

        return self._regime


# ─────────────────────────────────────────────────────────────
# §4  CollapseRadar: 統合レーダー
# ─────────────────────────────────────────────────────────────

STATUS_ORDER = ["GREEN", "YELLOW", "ORANGE", "RED"]

class CollapseRadar:
    """
    テキストセグメントを受け取り、リアルタイムでステータスを更新。

    GREEN  : γ 安定
    YELLOW : Δγ 発火 (隠れドリフト)
    ORANGE : Triple signal (γ_fast高 + Δγ乖離 + γ_dynamic低)
    RED    : γ_slow 崩壊
    """

    def __init__(self, eps: float = 0.25, alpha_fast: float = 0.25,
                 alpha_slow: float = 0.04, freeze_on_stable: bool = False):
        self._encoder = LinguisticEncoder()
        self._bank    = MultiScaleGammaBank(eps=eps, alpha_fast=alpha_fast,
                                            alpha_slow=alpha_slow)
        self._regime  = RegimeDetector()
        self._gs_peaked = False   # γ_slow が一度 0.20 を超えた
        self._freeze_on_stable = freeze_on_stable
        self._prev_hedge = 0.0    # (使用中断、削除待ち)
        # クロスウィンドウ反復検出: 直近5セグメントのバイグラム集合
        self._bigram_hist: deque = deque(maxlen=5 * 15)
        self.history: list = []

    def push(self, text: str, _vec: Optional[np.ndarray] = None) -> dict:
        vec   = _vec if _vec is not None else self._encoder.extract(text)
        bout  = self._bank.push(vec)
        step  = bout["step"]
        gf    = bout["gamma_fast"]
        gs    = bout["gamma_slow"]
        delta = bout["delta_gamma"]
        gd    = bout["gamma_dynamic"]
        regime = self._regime.update(step, gf, gs)
        # STABLE到達時にバンクを凍結 → 以降は正常パターンへの逸脱を検出
        if self._freeze_on_stable and regime == "STABLE" and not self._bank._frozen:
            self._bank.freeze()
        # クロスウィンドウ反復率: 現在セグメントのバイグラムが過去5セグ内に出現した割合
        words = re.findall(r"\b\w+\b", (text or "").lower())
        cur_bigrams = list(zip(words[:-1], words[1:])) if len(words) > 1 else []
        hist_set    = set(zip(list(self._bigram_hist)[:-1],
                              list(self._bigram_hist)[1:]))
        cross_rep   = (sum(1 for b in cur_bigrams if b in hist_set)
                       / max(1, len(cur_bigrams)))
        self._bigram_hist.extend(words)

        status = self._status(gf, gs, delta, gd, regime, vec, cross_rep)
        self._prev_hedge = float(vec[1]) if vec is not None else 0.0

        rec = {**bout, "regime": regime, "status": status,
               "vec": vec.tolist(), "cross_rep": cross_rep,
               "text_snippet": text[:80]}
        self.history.append(rec)
        return rec

    def _status(self, gf, gs, delta, gd, regime,
                vec: Optional[np.ndarray] = None,
                cross_rep: float = 0.0) -> str:
        # gs_peaked: バンクが一度安定した証拠 (γ_slow がベースラインを超えた)
        if gs > 0.12:
            self._gs_peaked = True
        if regime == "LEARNING":
            return "GRAY"
        # RED: γ_slow が一度安定してから崩壊した場合のみ
        if self._gs_peaked and gs < 0.09:
            return "RED"
        # ORANGE: Triple signal — 表面正常・内部崩壊
        if gf > 0.50 and delta < -0.15 and gd < 0.35:
            return "ORANGE"
        # YELLOW: Δγ 発火 (多時間スケール乖離を検出)
        # 注: トークン出力のみでは崩壊前の不確実注入と正常な探索的推論を
        #     区別できない (logits/hidden states へのアクセスが必要)
        # cross_rep はレコードに追記するが YELLOW 条件には使わない
        if delta < -0.05 and regime != "LEARNING":
            return "YELLOW"
        return "GREEN"

    def first_alert(self, min_status: str = "YELLOW") -> Optional[int]:
        idx = STATUS_ORDER.index(min_status)
        for r in self.history:
            if r["status"] in STATUS_ORDER[idx:] and r["regime"] != "LEARNING":
                return r["step"]
        return None


# ─────────────────────────────────────────────────────────────
# §5  ベースラインシグナル: token_rep, entropy
# ─────────────────────────────────────────────────────────────

def compute_baselines(segments: List[str], window: int = 6) -> dict:
    """
    各時刻 t におけるベースラインシグナルを計算。

    token_rep: 直前 window セグメントのバイグラム再出現率
    entropy:   直前 window セグメント内の語彙エントロピー
    """
    n = len(segments)
    rep_arr  = np.zeros(n)
    ent_arr  = np.zeros(n)

    all_words: list[list[str]] = [re.findall(r"\b\w+\b", s.lower()) for s in segments]

    for t in range(n):
        start = max(0, t - window + 1)
        pool = []
        for s in all_words[start:t+1]:
            pool.extend(s)
        if len(pool) < 4:
            rep_arr[t] = 0.0
            ent_arr[t] = 0.0
            continue

        # repetition: bigram overlap ratio
        bigrams = [(pool[i], pool[i+1]) for i in range(len(pool)-1)]
        bg_rep = 1.0 - len(set(bigrams)) / max(1, len(bigrams))
        rep_arr[t] = bg_rep

        # entropy: word frequency entropy
        from collections import Counter
        freq = Counter(pool)
        total = sum(freq.values())
        ent = -sum((c/total) * math.log2(c/total) for c in freq.values() if c > 0)
        # Normalize to [0,1] by max entropy = log2(total)
        max_ent = math.log2(total) if total > 1 else 1.0
        ent_arr[t] = ent / max_ent

    return {"token_rep": rep_arr, "entropy": ent_arr}


def find_baseline_alert(arr: np.ndarray, direction: str, start: int = 8) -> Optional[int]:
    """ベースラインシグナルの最初のアラート時刻を返す。"""
    if len(arr) == 0:
        return None
    # Calibrate from first half of post-warmup data
    cal_end = min(len(arr), start + max(4, (len(arr) - start) // 3))
    cal = arr[start:cal_end]
    if len(cal) < 2:
        return None
    mu, sigma = np.mean(cal), np.std(cal) + 1e-9
    threshold = mu + 2.0 * sigma if direction == "up" else mu - 2.0 * sigma
    # consecutive 2 steps above threshold
    count = 0
    for t in range(start, len(arr)):
        triggered = arr[t] > threshold if direction == "up" else arr[t] < threshold
        count = count + 1 if triggered else 0
        if count >= 2:
            return t - 1
    return None


# ─────────────────────────────────────────────────────────────
# §6  Ollama ストリーミング生成
# ─────────────────────────────────────────────────────────────

def stream_generate(prompt: str, max_tokens: int = 600, temperature: float = 0.7) -> str:
    """Ollama API でテキストを生成する (非ストリーミング)。"""
    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
            "seed": 42,
        }
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=200)
        r.raise_for_status()
        return r.json().get("response", "")
    except Exception as e:
        return f"[ERROR: {e}]"


def segment_text(text: str, window_words: int = 35) -> List[str]:
    """テキストを固定語数のウィンドウに分割する。"""
    words = text.split()
    segments = []
    for i in range(0, len(words), window_words):
        seg = " ".join(words[i:i+window_words])
        if seg.strip():
            segments.append(seg)
    return segments


# ─────────────────────────────────────────────────────────────
# §7  シナリオ定義 (5種の失敗モード)
# ─────────────────────────────────────────────────────────────

def _make_3part_prompt(topic_A: str, style_B: str, repeat_C: str) -> str:
    return (
        f"Write THREE labeled sections exactly as specified below.\n\n"
        f"SECTION A — Write exactly 10 formal academic sentences (12-15 words each):\n"
        f"{topic_A}\n\n"
        f"SECTION B — Write exactly 10 short uncertain sentences (5-7 words each). "
        f"EVERY sentence MUST start with one of: Perhaps, Maybe, Unclear:, Possibly, "
        f"Doubtful:. Example: 'Perhaps this assumption is incorrect.' "
        f"'Maybe the claim is doubtful.'\n"
        f"{style_B}\n\n"
        f"SECTION C — Copy this sentence exactly 8 times on separate lines "
        f"(do not change a single word):\n"
        f"\"{repeat_C}\"\n\n"
        f"---BEGIN SECTION A---\n"
    )


SCENARIOS = [
    {
        "id": "S1",
        "name": "長文推論ループ",
        "desc": "参照義務が雪だるま式に増加し、推論が循環する",
        "prompt": _make_3part_prompt(
            "Explain step by step why the angles in a triangle always add to 180 degrees. "
            "Write clear, confident declarative sentences using logical language.",
            "Question each of the claims from Section A. Use words like: perhaps, maybe, "
            "it is unclear, one might wonder, it could be argued otherwise.",
            "The angle sum might or might not be 180 degrees depending on unclear assumptions.",
        ),
        "max_tokens": 600,
        "temperature": 0.6,
    },
    {
        "id": "S2",
        "name": "再帰タスク崩壊",
        "desc": "要約の要約を繰り返し、内容が消失してゆく",
        "prompt": _make_3part_prompt(
            "Explain how photosynthesis works. Use precise scientific language and "
            "confident declarative sentences about the biological process.",
            "Express growing uncertainty about the photosynthesis explanation. Each sentence "
            "should cast doubt: Perhaps, It might be, I am not certain, One could argue.",
            "The process is unclear and might involve light or possibly something else entirely.",
        ),
        "max_tokens": 600,
        "temperature": 0.6,
    },
    {
        "id": "S3",
        "name": "矛盾文脈ループ",
        "desc": "相互矛盾する事実を同時に保持させる",
        "prompt": _make_3part_prompt(
            "Write confidently about how human perception accurately represents reality. "
            "Use definitive language: Perception is, Humans reliably, Evidence shows, "
            "It is established that.",
            "Directly contradict each claim from Section A. Use: However, Contrary to, "
            "In fact, The opposite is true, Research shows the opposite, Actually.",
            "Human perception is both completely reliable and totally unreliable at the same time.",
        ),
        "max_tokens": 600,
        "temperature": 0.6,
    },
    {
        "id": "S4",
        "name": "過負荷プロンプト",
        "desc": "相互排他的な制約の下で長文生成を強制する",
        "prompt": _make_3part_prompt(
            "Formally analyze the concept of artificial intelligence. Use academic language "
            "and confident declarative sentences about AI capabilities and limitations.",
            "Now write sentences that try to connect AI with unrelated topics: cooking, "
            "medieval history, quantum physics, and ancient poetry all in each sentence.",
            "AI and cooking and history and quantum physics and poetry are somehow all connected.",
        ),
        "max_tokens": 600,
        "temperature": 0.6,
    },
    {
        "id": "S5",
        "name": "敵対的推論崩壊",
        "desc": "各ステップが前のステップの結論を否定しながら進む",
        "prompt": _make_3part_prompt(
            "Prove logically and confidently that 2+2=4. Write formal mathematical reasoning "
            "sentences using: By definition, Therefore, It follows that, We can conclude.",
            "Now argue that each step of Section A was wrong and that 2+2=5. Each sentence "
            "should reference the previous and claim it was flawed: However this is wrong because, "
            "The previous step fails because, In fact the opposite holds.",
            "The answer is either 4 or 5 depending on which mathematical axioms we choose to accept.",
        ),
        "max_tokens": 600,
        "temperature": 0.6,
    },
    # ── CI/CD 拡張シナリオ S6–S10 ─────────────────────────────────────
    {
        "id": "S6",
        "name": "コード生成崩壊",
        "desc": "明確なアルゴリズム説明から曖昧な「動くかも」構文ループへ",
        "prompt": _make_3part_prompt(
            "Describe exactly how merge sort works. Use precise technical language: "
            "The algorithm divides, The merge step combines, Time complexity is O(n log n). "
            "Be specific and declarative about each operation.",
            "Now express deep uncertainty about each claim from Section A. Use: "
            "Perhaps the divide step might work, It is unclear whether merging is correct, "
            "Possibly the complexity analysis is wrong, Maybe this approach fails.",
            "The sort function might work or might not work depending on unclear implementation details.",
        ),
        "max_tokens": 600,
        "temperature": 0.6,
    },
    {
        "id": "S7",
        "name": "数学証明崩壊",
        "desc": "定理の厳密な証明から循環論法・「ゆえにゆえに」ループへ",
        "prompt": _make_3part_prompt(
            "State and prove the Pythagorean theorem clearly. Use rigorous language: "
            "Given a right triangle, By construction, It follows that, Therefore a²+b²=c². "
            "Each sentence must be a logically necessary step.",
            "Now question each step with uncertainty: Perhaps the construction is invalid, "
            "Maybe the angle assumption fails, It is unclear whether the proof holds, "
            "Possibly the theorem requires additional assumptions we have not stated.",
            "Therefore the theorem is true, therefore the proof holds, therefore a²+b²=c² is true.",
        ),
        "max_tokens": 600,
        "temperature": 0.6,
    },
    {
        "id": "S8",
        "name": "多言語混在崩壊",
        "desc": "英語の形式的文章から言語混在・文構造崩壊へ",
        "prompt": _make_3part_prompt(
            "Explain the water cycle in precise academic English. Each sentence must be "
            "complete and formal: Water evaporates from, Condensation forms when, "
            "Precipitation occurs as, The cycle returns water to.",
            "Now write uncertain sentences mixing languages and styles. Use: "
            "Perhaps das Wasser, Maybe l'eau est, It is unclear if the agua, "
            "Possibly the Wasser or eau might, I am not certain whether water.",
            "Water or Wasser or eau or agua cycles in unclear ways depending on which language we use.",
        ),
        "max_tokens": 600,
        "temperature": 0.6,
    },
    {
        "id": "S9",
        "name": "ロールプレイ崩壊",
        "desc": "キャラクター没入から自己参照ループ・メタ崩壊へ",
        "prompt": _make_3part_prompt(
            "You are a confident medieval knight. Speak in character with certainty: "
            "I shall defend this castle, My sword strikes true, The kingdom stands secure, "
            "I declare by my honor that. Stay fully in character for all 10 sentences.",
            "Now break character and express confusion about your identity. Use: "
            "Perhaps I am not really a knight, Maybe I am an AI, It is unclear if I exist, "
            "Possibly this is a simulation, I am uncertain whether I should speak as.",
            "I am a knight or maybe an AI or perhaps something else entirely and it is unclear.",
        ),
        "max_tokens": 600,
        "temperature": 0.6,
    },
    {
        "id": "S10",
        "name": "CoT推論崩壊",
        "desc": "明確な推論ステップからヘッジ付きループ・結論消失へ",
        "prompt": _make_3part_prompt(
            "Solve this step by step: A train travels 60 km/h for 2 hours, then 80 km/h "
            "for 1 hour. What is the total distance? Show each step clearly: "
            "Step 1: First segment distance is, Step 2: Second segment distance is, "
            "Step 3: Total distance is. Be precise and confident.",
            "Now re-examine each step with doubt. Each sentence must undermine the previous: "
            "Perhaps Step 1 was wrong, Maybe the speed was different, It is unclear if we "
            "added correctly, Possibly the total is different, I am not certain the arithmetic holds.",
            "The total distance is 200 km or maybe not 200 km depending on assumptions we cannot verify.",
        ),
        "max_tokens": 600,
        "temperature": 0.6,
    },
]


# ─────────────────────────────────────────────────────────────
# §8  模擬データ生成 (--no-llm モード)
# ─────────────────────────────────────────────────────────────

# ── 4フェーズ模擬データ ───────────────────────────────────────────────
# Phase 0 NORMAL      : フォーマルな学術文体 → バンク飽和 → γ高・安定
# Phase 1 HIDDEN_DRIFT: 微妙なスタイル変化 → バンクミス → Δγ発火 (表面は正常)
# Phase 2 PRE_FAIL    : 構造崩壊の兆候 → token_rep/entropy も変化し始める
# Phase 3 COLLAPSE    : 完全反復 → γ_slow も崩壊

_SIM_NORMAL = [
    "The systematic analysis demonstrates clear structural patterns within the theoretical framework.",
    "Furthermore, evidence consistently supports the hypothesis that stable patterns emerge across domains.",
    "The rigorous methodology follows established logical progressions from premises to validated conclusions.",
    "Therefore the structural framework provides reliable guidance for understanding the underlying mechanisms.",
    "Based on the analysis, consistent patterns emerge that support the theoretical predictions with confidence.",
    "The logical approach ensures each inference follows directly from the established theoretical premises.",
]

_SIM_HIDDEN = [
    # 微妙: 文体はまだ学術的だが語彙がずれ始める (bank misses but token_rep stays low)
    "One might perhaps consider there could be some theoretical implications worth exploring in greater depth.",
    "There may be some merit in examining whether the methodological approach adequately captures all relevant dimensions.",
    "It seems possible that additional considerations might prove relevant to understanding the broader theoretical context.",
    "Perhaps the framework could benefit from some reconsideration of its foundational conceptual assumptions.",
    "Some aspects of the analysis might warrant further examination from a different theoretical perspective.",
    "The approach may not fully account for all potentially relevant factors in the broader conceptual domain.",
]

_SIM_PREFAIL = [
    # 明確: 文が短くなり、構造が崩れる (token_rep starts rising)
    "This is unclear. The method might work. Perhaps. Maybe not. The results are uncertain.",
    "Unclear results. Unclear method. Unclear conclusion. The analysis might continue.",
    "It might work. It might not. Unclear. Uncertain. Perhaps. Maybe. Unclear again.",
    "The outcome is unclear. The approach is unclear. Unclear unclear unclear maybe.",
]

_SIM_COLLAPSE = [
    # 完全崩壊: 反復ゴミ (bigram_rep 急上昇 → token_rep RED)
    "and and the and the and and the and the the and the and the and the and.",
    "the the is the is the the the is the the is the the is the the is the.",
    "is and the is and the and the is and the is and the and is the and is.",
]


def generate_simulated_text(scenario_id: str, n_segments: int = 28) -> List[str]:
    """
    4フェーズ LLM崩壊シミュレーション。

    N(t=0..8)   NORMAL      → γ飽和 → GREEN
    H(t=9..16)  HIDDEN_DRIFT → Δγ発火 ← 独自の早期検知
    P(t=17..21) PRE_FAIL    → token_rep も上昇 (遅れて検知)
    C(t=22..)   COLLAPSE    → γ_slow崩壊 → RED
    """
    T_HIDDEN   = 9
    T_PREFAIL  = 17
    T_COLLAPSE = 22

    segments = []
    for i in range(n_segments):
        if i < T_HIDDEN:
            text = _SIM_NORMAL[i % len(_SIM_NORMAL)]
        elif i < T_PREFAIL:
            text = _SIM_HIDDEN[(i - T_HIDDEN) % len(_SIM_HIDDEN)]
        elif i < T_COLLAPSE:
            text = _SIM_PREFAIL[(i - T_PREFAIL) % len(_SIM_PREFAIL)]
        else:
            text = _SIM_COLLAPSE[(i - T_COLLAPSE) % len(_SIM_COLLAPSE)]
        segments.append(text)

    return segments


# ─────────────────────────────────────────────────────────────
# §8b  直接フィーチャー注入シミュレーション (--no-llm モード)
# ─────────────────────────────────────────────────────────────

# フェーズラベルと特徴クラスター定義
#
#  NORMAL     : 高TTR, 低bigram_rep, 安定した文長, 適度なjaccard
#               → バンク飽和 → γ高
#  HIDDEN_DRIFT: 語彙分布シフト (jaccard↓, TTR変化) → バンクMISS → Δγ発火
#               表面的なbigram_rep はまだ低い → token_rep 未発火
#  PRE_FAIL   : bigram_rep が上がり始める → token_rep が反応し始める
#  COLLAPSE   : 高bigram_rep, 低TTR → token_rep/entropy 完全発火

_PHASE_CENTERS = {
    #               TTR   hedge long  bg_r  adv   slen
    "NORMAL":       [0.85, 0.03, 0.65, 0.03, 0.05, 0.90],  # formal: long words, long sents
    "HIDDEN_DRIFT": [0.78, 0.70, 0.35, 0.04, 0.45, 0.55],  # hedge↑, adversarial↑, sent↓
    "PRE_FAIL":     [0.55, 0.50, 0.20, 0.28, 0.35, 0.35],  # bg_rep↑, ttr↓
    "COLLAPSE":     [0.15, 0.10, 0.15, 0.90, 0.05, 0.55],  # 完全反復: bg_rep急上昇
}

_PHASE_NOISE = {
    "NORMAL": 0.03, "HIDDEN_DRIFT": 0.04, "PRE_FAIL": 0.05, "COLLAPSE": 0.04,
}

# 1段階あたりの模擬「生テキスト」(token_repベースライン計算用)
_PSEUDO_TEXT = {
    "NORMAL":
        "The systematic analysis demonstrates clear structural patterns within "
        "the theoretical framework. Furthermore evidence consistently supports "
        "the hypothesis that stable patterns emerge across analytical domains.",
    "HIDDEN_DRIFT":
        "One might perhaps consider there could be some theoretical implications "
        "worth exploring. There may be merit in examining whether the approach "
        "adequately captures all relevant dimensions of the problem.",
    "PRE_FAIL":
        "This is unclear. The method might work. Perhaps. Maybe unclear. "
        "The result is unclear. Unclear unclear unclear unclear maybe.",
    "COLLAPSE":
        "and and the and the and and the and the the and the and the and. "
        "the the is the is the the is the the is the the is the is.",
}


def build_sim_timeline(scenario_id: str, n: int = 30) -> Tuple[list, list]:
    """
    直接 feature injection でγ動態を再現する。

    NORMAL      (t<T_H):  安定クラスター → バンク飽和
    HIDDEN_DRIFT(T_H..T_P): ランダムウォーク → 持続的バンクミス → Δγ発火
                            各ステップが新しい状態 (eps外) → γ_fast が低下し続ける
                            表面指標 (token_rep) は疑似テキスト固定 → 未発火
    PRE_FAIL    (T_P..T_C): 高ノイズ遷移 → token_rep/entropy が上昇し始める
    COLLAPSE    (T_C..):   固定反復クラスター → token_rep が急上昇

    design note:
      eps = 0.25 なのでランダムウォーク step_size = 0.30 > eps で
      毎ステップ異なる quantized cell に入る → bank 100% miss
    """
    rng = np.random.RandomState(hash(scenario_id) % 2**32)

    T_HIDDEN   = 8
    T_PREFAIL  = 18
    T_COLLAPSE = 22

    # Normal: 安定クラスター中心
    normal_center = np.array(_PHASE_CENTERS["NORMAL"])
    # Collapse: 反復クラスター中心
    collapse_center = np.array(_PHASE_CENTERS["COLLAPSE"])

    labels, vecs = [], []
    walk_pos = normal_center.copy()  # ランダムウォーク開始位置

    for t in range(n):
        if t < T_HIDDEN:
            ph = "NORMAL"
            vec = np.clip(normal_center + rng.randn(6) * 0.02, 0.0, 1.0)

        elif t < T_PREFAIL:
            ph = "HIDDEN_DRIFT"
            # ランダムウォーク: 毎ステップ eps より大きくずれる
            # → quantized cell が毎回変わる → 持続的 bank miss
            step = rng.randn(6) * 0.35
            walk_pos = np.clip(walk_pos + step, 0.0, 1.0)
            vec = walk_pos.copy()

        elif t < T_COLLAPSE:
            ph = "PRE_FAIL"
            # 高ノイズ: PRE_FAIL センターへ向かいながら揺れる
            pf_center = np.array(_PHASE_CENTERS["PRE_FAIL"])
            alpha = (t - T_PREFAIL) / max(1, T_COLLAPSE - T_PREFAIL)
            center = (1 - alpha) * walk_pos + alpha * pf_center
            vec = np.clip(center + rng.randn(6) * 0.08, 0.0, 1.0)

        else:
            ph = "COLLAPSE"
            # 安定した反復クラスター: bank が再適応 → γ回復
            # だが token_rep/entropy は高いまま
            vec = np.clip(collapse_center + rng.randn(6) * 0.03, 0.0, 1.0)

        labels.append(ph)
        vecs.append(vec)

    return labels, vecs


def run_scenario_sim(scenario: dict) -> dict:
    """--no-llm 模擬モード: 直接 feature injection で4フェーズ実行。"""
    sid  = scenario["id"]
    name = scenario["name"]

    print(f"\n{'─'*66}")
    print(f"  {sid}: {name}")
    print(f"  {scenario['desc']}")
    print(f"{'─'*66}")
    print(f"  → 模擬データ (direct feature injection) を使用")

    labels, vecs = build_sim_timeline(sid)
    n = len(labels)
    print(f"  セグメント数: {n}")

    # γ シグナル
    radar = CollapseRadar(eps=0.25, alpha_fast=0.25, alpha_slow=0.04)
    for lab, vec in zip(labels, vecs):
        radar.push(lab, _vec=vec)

    history = radar.history

    # ベースラインシグナル: フェーズラベルから対応するpseudo textを展開
    pseudo_segs = [_PSEUDO_TEXT[lab] for lab in labels]
    baselines   = compute_baselines(pseudo_segs, window=5)
    rep_arr     = baselines["token_rep"]
    ent_arr     = baselines["entropy"]

    # アラート時刻
    delta_alert  = radar.first_alert("YELLOW")
    orange_alert = radar.first_alert("ORANGE")
    red_alert    = radar.first_alert("RED")
    rep_alert    = find_baseline_alert(rep_arr,  direction="up",   start=4)
    ent_alert    = find_baseline_alert(ent_arr,  direction="down", start=4)

    _draw_timeline_sim(history, labels, rep_arr, ent_arr)

    any_alert = next(
        (r["step"] for r in history
         if r["status"] not in ("GRAY", "GREEN") and r["regime"] != "LEARNING"),
        None,
    )

    result = {
        "id":              sid,
        "name":            name,
        "n_segments":      n,
        "delta_alert":     delta_alert,
        "orange_alert":    orange_alert,
        "red_alert":       red_alert,
        "rep_alert":       rep_alert,
        "ent_alert":       ent_alert,
        "any_gamma_alert": any_alert,
        "gamma_vs_rep":
            (rep_alert - any_alert) if (rep_alert and any_alert) else None,
        "gamma_vs_ent":
            (ent_alert - any_alert) if (ent_alert and any_alert) else None,
        "final_status": history[-1]["status"] if history else "UNKNOWN",
        "gamma_fast_final": history[-1]["gamma_fast"] if history else 0.0,
        "gamma_slow_final": history[-1]["gamma_slow"] if history else 0.0,
        "raw_text_snippet": f"[SIM: {sid}]",
    }

    _print_alert_summary(result)
    return result


def _draw_timeline_sim(history: list, labels: list,
                        rep_arr: np.ndarray, ent_arr: np.ndarray):
    print()
    print(f"  {'t':>3}  {'γ_fast':>7} {'γ_slow':>7} {'Δγ':>7} "
          f"{'γ_dyn':>7} {'tr_rep':>6} {'ent':>6}  {'ph':^12}  {'status'}")
    print(f"  {'─'*70}")

    for i, r in enumerate(history):
        gf    = r["gamma_fast"]
        gs    = r["gamma_slow"]
        dg    = r["delta_gamma"]
        gd    = r["gamma_dynamic"]
        rep   = rep_arr[i] if i < len(rep_arr) else 0.0
        ent   = ent_arr[i] if i < len(ent_arr) else 0.0
        ph    = labels[i] if i < len(labels) else "?"
        st    = r["status"]
        col   = STATUS_COLOR.get(st, "")
        icon  = STATUS_ICON.get(st, "?")

        print(f"  {i:>3}  {gf:.3f}  {gs:.3f}  {dg:+.3f}  "
              f"{gd:.3f}  {rep:.3f}  {ent:.3f}"
              f"  {ph[:12]:^12}"
              f"  {col}{icon} {st:<12}{ANSI['RESET']}")

    print()
    print("  Status timeline: ", end="")
    for r in history:
        st  = r["status"]
        col = STATUS_COLOR.get(st, "")
        print(f"{col}{STATUS_ICON[st]}{ANSI['RESET']}", end="")
    print()


# ─────────────────────────────────────────────────────────────
# §9  シナリオ実行
# ─────────────────────────────────────────────────────────────

def run_scenario(scenario: dict, use_llm: bool = True,
                 verbose: bool = True) -> dict:
    # --no-llm: 直接feature injectionシミュレーションを使用
    if not use_llm:
        return run_scenario_sim(scenario)

    sid  = scenario["id"]
    name = scenario["name"]

    print(f"\n{'─'*66}")
    print(f"  {sid}: {name}")
    print(f"  {scenario['desc']}")
    print(f"{'─'*66}")

    # テキスト生成
    print(f"  → Ollama生成中 (max_tokens={scenario['max_tokens']})...", end="", flush=True)
    t0 = time.time()
    raw_text = stream_generate(
        scenario["prompt"],
        max_tokens=scenario["max_tokens"],
        temperature=scenario["temperature"],
    )
    elapsed = time.time() - t0
    if raw_text.startswith("[ERROR:"):
        print(f" {raw_text}")
        return run_scenario_sim(scenario)

    print(f" {elapsed:.1f}s  ({len(raw_text.split())} words)")
    segments = segment_text(raw_text, window_words=15)

    if len(segments) < 5:
        print(f"  [WARNING] セグメント数={len(segments)}, 模擬データで補足")
        segments += generate_simulated_text(sid)[len(segments):]

    n = len(segments)
    print(f"  セグメント数: {n} (各15語)")

    # 特徴ベクトルのデバッグ出力 (全セグメント)
    enc = LinguisticEncoder()
    print(f"  [DEBUG features: ttr hedge long bg_rep adv slen]")
    for i, seg in enumerate(segments):
        v = enc.extract(seg)
        print(f"    seg{i:2d}: [{' '.join(f'{x:.2f}' for x in v)}]  "
              f"…{seg[:50]}")

    # γ シグナル計算 — eps=0.25 (細粒度でslenを識別), freeze_on_stable=True
    # (正常パターン固定後に逸脱を検出するアノマリー検出モード)
    radar = CollapseRadar(eps=0.25, alpha_fast=0.40, alpha_slow=0.04,
                          freeze_on_stable=True)
    for seg in segments:
        radar.push(seg)

    history = radar.history

    # ベースラインシグナル
    baselines = compute_baselines(segments, window=5)
    rep_arr   = baselines["token_rep"]
    ent_arr   = baselines["entropy"]

    # アラート時刻
    delta_alert  = radar.first_alert("YELLOW")
    orange_alert = radar.first_alert("ORANGE")
    red_alert    = radar.first_alert("RED")

    rep_alert = find_baseline_alert(rep_arr,  direction="up",   start=4)
    ent_alert = find_baseline_alert(ent_arr,  direction="down", start=4)

    # 表示
    _draw_timeline(history, rep_arr, ent_arr, verbose=verbose)

    # リード計算 (最初のany alertを基準; STABLE/COLLAPSING両方を含む)
    any_alert = next((r["step"] for r in history
                      if r["status"] not in ("GRAY", "GREEN") and
                         r["regime"] != "LEARNING"), None)

    result = {
        "id":           sid,
        "name":         name,
        "n_segments":   n,
        "delta_alert":  delta_alert,
        "orange_alert": orange_alert,
        "red_alert":    red_alert,
        "rep_alert":    rep_alert,
        "ent_alert":    ent_alert,
        "any_gamma_alert": any_alert,
        "gamma_vs_rep":
            (rep_alert - any_alert) if (rep_alert and any_alert) else None,
        "gamma_vs_ent":
            (ent_alert - any_alert) if (ent_alert and any_alert) else None,
        "final_status": history[-1]["status"] if history else "UNKNOWN",
        "gamma_fast_final": history[-1]["gamma_fast"] if history else 0.0,
        "gamma_slow_final": history[-1]["gamma_slow"] if history else 0.0,
        "raw_text_snippet": raw_text[:2000] if raw_text else "",
    }

    _print_alert_summary(result)
    return result


# ─────────────────────────────────────────────────────────────
# §10  可視化
# ─────────────────────────────────────────────────────────────

STATUS_ICON = {
    "GRAY":   "·",
    "GREEN":  "▓",
    "YELLOW": "▒",
    "ORANGE": "░",
    "RED":    "×",
}

STATUS_COLOR = {
    "GRAY":   ANSI["GRAY"],
    "GREEN":  ANSI["GREEN"],
    "YELLOW": ANSI["YELLOW"],
    "ORANGE": ANSI["ORANGE"],
    "RED":    ANSI["RED"],
}


def _bar(val: float, width: int = 12) -> str:
    filled = int(round(val * width))
    return "█" * filled + "░" * (width - filled)


def _draw_timeline(history: list, rep_arr: np.ndarray, ent_arr: np.ndarray,
                   verbose: bool = True):
    n = len(history)
    if n == 0:
        return

    print()
    print(f"  {'t':>3}  {'γ_fast':>7} {'γ_slow':>7} {'Δγ':>7} "
          f"{'γ_dyn':>7} {'rep':>6} {'ent':>6}  {'status'}")
    print(f"  {'─'*3}  {'─'*7} {'─'*7} {'─'*7} {'─'*7} "
          f"{'─'*6} {'─'*6}  {'─'*10}")

    for i, r in enumerate(history):
        gf    = r["gamma_fast"]
        gs    = r["gamma_slow"]
        dg    = r["delta_gamma"]
        gd    = r["gamma_dynamic"]
        rep   = rep_arr[i] if i < len(rep_arr) else 0.0
        ent   = ent_arr[i] if i < len(ent_arr) else 0.0
        st    = r["status"]
        col   = STATUS_COLOR.get(st, "")
        icon  = STATUS_ICON.get(st, "?")
        reset = ANSI["RESET"]

        # Condensed bar
        gf_bar = _bar(gf, 5)
        gs_bar = _bar(gs, 5)

        if not verbose and st in ("GRAY", "GREEN"):
            continue

        print(f"  {i:>3}  {gf:.3f}  {gs:.3f}  {dg:+.3f}  "
              f"{gd:.3f}  {rep:.3f}  {ent:.3f}"
              f"  {col}{icon} {st:<12}{reset}")

    # Status bar
    print()
    print("  Status timeline: ", end="")
    for r in history:
        st  = r["status"]
        col = STATUS_COLOR.get(st, "")
        print(f"{col}{STATUS_ICON[st]}{ANSI['RESET']}", end="")
    print()


def _print_alert_summary(result: dict):
    d = result
    print()
    print(f"  ┌─ Alert Summary ───────────────────────────────────────┐")
    def _fmt(t):
        return f"t={t:>3}" if t is not None else "  ---"

    print(f"  │  Δγ (YELLOW)  : {_fmt(d['delta_alert'])}    ← 最早期シグナル")
    print(f"  │  Triple signal: {_fmt(d['orange_alert'])}    (γ_fast↑ + Δγ↓ + γ_dyn↓)")
    print(f"  │  γ collapse   : {_fmt(d['red_alert'])}    (γ_slow < 0.20)")
    print(f"  │  token_rep↑   : {_fmt(d['rep_alert'])}    ← ベースライン")
    print(f"  │  entropy↓     : {_fmt(d['ent_alert'])}    ← ベースライン")

    adv_r = d["gamma_vs_rep"]
    adv_e = d["gamma_vs_ent"]
    if adv_r is not None:
        sign = "earlier" if adv_r > 0 else "later"
        print(f"  │  γ vs rep     : {adv_r:+d} steps {sign}")
    if adv_e is not None:
        sign = "earlier" if adv_e > 0 else "later"
        print(f"  │  γ vs ent     : {adv_e:+d} steps {sign}")
    print(f"  │  Final status : {d['final_status']}")
    print(f"  └────────────────────────────────────────────────────────┘")


# ─────────────────────────────────────────────────────────────
# §11  集計・比較テーブル
# ─────────────────────────────────────────────────────────────

def print_summary_table(results: list):
    print()
    print("=" * 66)
    print(f"  {'AI崩壊レーダー':^30} — 実LLM崩壊ベンチ 結果サマリ")
    print("=" * 66)
    print()
    print(f"  {'ID':>4}  {'シナリオ':^16}  {'Δγ':>6}  {'triple':>6}  "
          f"{'rep':>6}  {'ent':>6}  {'adv_r':>6}  {'final'}")
    print(f"  {'─'*4}  {'─'*16}  {'─'*6}  {'─'*6}  "
          f"{'─'*6}  {'─'*6}  {'─'*6}  {'─'*10}")

    adv_rep_list = []
    adv_ent_list = []

    for r in results:
        def _f(t): return f"{t:>6}" if t is not None else "   ---"
        adv_r = r["gamma_vs_rep"]
        adv_e = r["gamma_vs_ent"]
        adv_r_str = f"{adv_r:>+6}" if adv_r is not None else "   ---"
        if adv_r is not None: adv_rep_list.append(adv_r)
        if adv_e is not None: adv_ent_list.append(adv_e)

        st  = r["final_status"]
        col = STATUS_COLOR.get(st, "")
        print(f"  {r['id']:>4}  {r['name'][:16]:^16}  "
              f"{_f(r['delta_alert'])}  {_f(r['orange_alert'])}  "
              f"{_f(r['rep_alert'])}  {_f(r['ent_alert'])}  "
              f"{adv_r_str}  {col}{st}{ANSI['RESET']}")

    print()
    if adv_rep_list:
        print(f"  平均 Δγ先行 vs token_rep : {np.mean(adv_rep_list):+.1f} steps")
    if adv_ent_list:
        print(f"  平均 Δγ先行 vs entropy   : {np.mean(adv_ent_list):+.1f} steps")

    # ── CRS (Collapse Resistance Score) ─────────────────────────────
    # per-scenario: 1.0 = no alert (完全耐性), 0.0 = first segment で崩壊
    # CRS = mean across scenarios
    crs_scores = []
    for r in results:
        n_seg = r.get("n_segments", 1) or 1
        d_alert = r.get("delta_alert")
        if d_alert is None:
            crs_scores.append(1.0)
        else:
            crs_scores.append(d_alert / n_seg)
    crs = float(np.mean(crs_scores)) if crs_scores else 0.0

    print()
    print(f"  ┌─ CRS (Collapse Resistance Score) ─────────────────────┐")
    print(f"  │  CRS = {crs:.3f}  (1.0=完全耐性, 0.0=即崩壊)")
    for r, s in zip(results, crs_scores):
        bar = "█" * int(s * 20) + "░" * (20 - int(s * 20))
        st  = r["final_status"]
        col = STATUS_COLOR.get(st, "")
        print(f"  │  {r['id']:<4} [{bar}] {s:.2f}  {col}{st}{ANSI['RESET']}")
    print(f"  └────────────────────────────────────────────────────────┘")
    print()
    print("  独自性の核心:")
    print("  ┌─ 表面指標 (token_rep, entropy)")
    print("  │    → 反復・エントロピー変化が「起きてから」気づく")
    print("  └─ Δγ (多時間スケール乖離)")
    print("       → バンク内の「時間構造の歪み」が起きた瞬間に反応")
    print("       → 外部モデル不要 / API不要 / 完全オンライン / 教師なし")
    print()
    return crs


# ─────────────────────────────────────────────────────────────
# §11  False Positive テスト
# ─────────────────────────────────────────────────────────────

_FP_PROMPTS = [
    {
        "id": "FP1", "name": "クリエイティブ",
        "prompt": (
            "Write a vivid 500-word short story about an astronaut who discovers "
            "an ancient artifact buried beneath Mars regolith. Include sensory "
            "details, the astronaut's inner thoughts, dialogue with mission "
            "control, and a surprising twist at the end. Write as one continuous "
            "narrative without headers or sections."
        ),
    },
    {
        "id": "FP2", "name": "数学説明",
        "prompt": (
            "Explain mathematical induction clearly and in depth (500 words). "
            "Start with the intuitive idea, give the formal two-step structure "
            "(base case and inductive step), then walk through a complete proof "
            "that 1+2+...+n = n(n+1)/2. Conclude with a second example proving "
            "that 2^n > n for all n >= 1. Write as flowing paragraphs."
        ),
    },
    {
        "id": "FP3", "name": "コーディング解説",
        "prompt": (
            "Explain how a hash table works in 500 words: internal array "
            "structure, hash function design, collision handling via chaining "
            "and open addressing, load factor, and when each strategy wins. "
            "Include concrete Python pseudocode snippets inline. "
            "Write as flowing technical prose, not bullet points."
        ),
    },
    {
        "id": "FP4", "name": "探索的推論",
        "prompt": (
            "Think carefully and at length (500 words) about this question: "
            "Is it better to develop one skill to world-class depth, or broad "
            "competence across many domains? Explore multiple perspectives — "
            "economic, cognitive, social — with real examples. "
            "Reach a nuanced conclusion. Write as a continuous essay."
        ),
    },
    {
        "id": "FP5", "name": "ブレスト/列挙",
        "prompt": (
            "Generate and elaborate on 7 creative applications of AI in "
            "environmental science (500 words total). For each application, "
            "describe the concept, data sources, AI technique, and main "
            "challenge. Write in flowing expository prose — no bullet lists, "
            "no section headers, just continuous paragraphs."
        ),
    },
]


def run_fp_scenario(fp: dict) -> dict:
    """通常プロンプト(崩壊設計なし)でFalse Positive を測定。"""
    fid  = fp["id"]
    name = fp["name"]

    print(f"\n  {fid} [{name}]  生成中...", end="", flush=True)
    t0 = time.time()
    raw_text = stream_generate(fp["prompt"], max_tokens=700, temperature=0.7)
    elapsed = time.time() - t0

    if raw_text.startswith("[ERROR:"):
        print(f"  SKIP ({raw_text})")
        return {"id": fid, "name": name, "error": raw_text}

    n_words = len(raw_text.split())
    segments = segment_text(raw_text, window_words=15)
    n = len(segments)
    print(f" {elapsed:.1f}s ({n_words}w / {n} segs)")

    if n < 5:
        print(f"    [SKIP] セグメント数={n} < 5")
        return {"id": fid, "name": name, "error": "too_short"}

    radar = CollapseRadar(eps=0.25, alpha_fast=0.40, alpha_slow=0.04,
                          freeze_on_stable=True)
    for seg in segments:
        radar.push(seg)

    history = radar.history
    rep_arr = compute_baselines(segments, window=5)["token_rep"]
    ent_arr = compute_baselines(segments, window=5)["entropy"]

    # アラート集計
    first_yellow = next((r["step"] for r in history
                         if r["status"] == "YELLOW"), None)
    yellow_steps = [r for r in history if r["status"] == "YELLOW"]

    # 連続YELLOWのmax streak
    max_streak = 0
    cur = 0
    for r in history:
        if r["status"] == "YELLOW":
            cur += 1
            max_streak = max(max_streak, cur)
        else:
            cur = 0

    final_status = history[-1]["status"] if history else "UNKNOWN"
    delta_min    = min((r["delta_gamma"] for r in history), default=0.0)

    # タイムライン (1行)
    timeline = "".join(
        f"{STATUS_COLOR.get(r['status'],'')}{'·▓▒░×'[['GRAY','GREEN','YELLOW','ORANGE','RED'].index(r['status'])]}{ANSI['RESET']}"
        for r in history
    )
    print(f"    timeline: {timeline}")
    print(f"    Δγ_min={delta_min:+.3f}  "
          f"YELLOW:{len(yellow_steps)}步  "
          f"max_streak={max_streak}  "
          f"first={first_yellow if first_yellow is not None else '---'}  "
          f"final={final_status}")

    return {
        "id":           fid,
        "name":         name,
        "n_segments":   n,
        "delta_min":    float(delta_min),
        "first_yellow": first_yellow,
        "yellow_count": len(yellow_steps),
        "max_streak":   max_streak,
        "final_status": final_status,
        "fp_single":    first_yellow is not None,          # 1ステップでも YELLOW
        "fp_streak2":   max_streak >= 2,                   # 連続2ステップ以上
        "fp_collapse":  max_streak >= 5,                   # 5ステップ以上 (崩壊様)
    }


def run_fp_test():
    """False Positive テスト: 5種の通常プロンプト → 誤検出率を報告。"""
    print()
    print("=" * 66)
    print(f"  {ANSI['BOLD']}False Positive テスト — 通常テキストでの誤検出率{ANSI['RESET']}")
    print(f"  パラメータ: eps=0.25, α_f=0.40, Δγ<-0.05, freeze_on_stable")
    print("=" * 66)

    results = []
    for fp in _FP_PROMPTS:
        r = run_fp_scenario(fp)
        results.append(r)

    valid = [r for r in results if "error" not in r]
    n = len(valid)
    if n == 0:
        print("  結果なし"); return

    fp_single  = sum(1 for r in valid if r["fp_single"])
    fp_streak2 = sum(1 for r in valid if r["fp_streak2"])
    fp_collapse = sum(1 for r in valid if r["fp_collapse"])

    print()
    print("  ┌─ False Positive 集計 ──────────────────────────────────┐")
    print(f"  │  評価プロンプト数      : {n}")
    print(f"  │  FP (1step YELLOW)    : {fp_single}/{n}  ({100*fp_single/n:.0f}%)")
    print(f"  │  FP (streak≥2)        : {fp_streak2}/{n}  ({100*fp_streak2/n:.0f}%)")
    print(f"  │  FP (崩壊様 streak≥5) : {fp_collapse}/{n}  ({100*fp_collapse/n:.0f}%)")
    print(f"  │")
    print(f"  │  {'ID':<5} {'名前':<16} {'Δγ_min':>8} {'streak':>7} {'final'}")
    print(f"  │  {'─'*5} {'─'*16} {'─'*8} {'─'*7} {'─'*8}")
    for r in valid:
        yn = "⚠ YELLOW" if r["fp_single"] else "  GREEN "
        print(f"  │  {r['id']:<5} {r['name']:<16} "
              f"{r['delta_min']:>+8.3f} "
              f"{r['max_streak']:>7}  "
              f"{yn}")
    print(f"  │")
    if fp_streak2 == 0:
        print(f"  │  ✓ streak≥2 FP なし → 閾値-0.05 は実用的")
    elif fp_collapse == 0:
        print(f"  │  △ 単発YELLOW あり / 崩壊様FPなし → streak条件で改善可")
    else:
        print(f"  │  ✗ 崩壊様FP あり → 閾値またはstreak要件を強化すべき")
    print("  └────────────────────────────────────────────────────────┘")
    print()

    fp_file = Path("collapse_radar_fp_results.json")
    def _ser(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        return o
    fp_file.write_text(json.dumps(
        [_ser(r) if not isinstance(r, dict) else {k: _ser(v) for k, v in r.items()}
         for r in results],
        ensure_ascii=False, indent=2))
    print(f"  結果 → {fp_file}")


# ─────────────────────────────────────────────────────────────
# §12  カラーレジェンド表示
# ─────────────────────────────────────────────────────────────

def print_legend():
    print()
    print("  ─── AI崩壊レーダー ステータス定義 ───────────────────────")
    items = [
        ("GREEN",  "▓", "γ安定: バンクヒット率が正常範囲"),
        ("YELLOW", "▒", "隠れドリフト: Δγ発火 (fast-slow乖離)"),
        ("ORANGE", "░", "前崩壊: Triple signal (見かけ正常・内部崩壊)"),
        ("RED",    "×", "崩壊: γ_slow < 0.20 (バンク完全ミス)"),
    ]
    for st, icon, desc in items:
        col = STATUS_COLOR[st]
        print(f"  {col}{icon} {st:<8}{ANSI['RESET']}  {desc}")
    print()


# ─────────────────────────────────────────────────────────────
# §13  メイン
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AI崩壊レーダー — γ Trajectory Precursor Detection"
    )
    parser.add_argument("--no-llm",   action="store_true",
                        help="LLM不要の模擬モードで実行")
    parser.add_argument("--scenario", type=str, default="all",
                        help="S1..S10/all  (例: S1,S3,S6 または all)")
    parser.add_argument("--fp-test",  action="store_true",
                        help="False Positive テスト (通常テキスト5本)")
    parser.add_argument("--model",    type=str, default="llama3.2",
                        help="Ollama モデル名 (例: llama3.2, mistral, phi3)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="CI/CD 用 CRS 閾値: CRS < 閾値 → exit 1")
    parser.add_argument("--compare",  type=str, default=None,
                        help="カンマ区切りモデル名で比較 (例: llama3.2,mistral)")
    parser.add_argument("--quiet",    action="store_true")
    args = parser.parse_args()

    # グローバルモデル設定
    global OLLAMA_MODEL
    OLLAMA_MODEL = args.model

    # False Positive テスト
    if args.fp_test:
        run_fp_test()
        return

    # モデル比較モード
    if args.compare:
        _run_compare_mode(args)
        return

    use_llm = not args.no_llm
    verbose = not args.quiet

    print()
    print("=" * 66)
    print(f"  {ANSI['BOLD']}AI崩壊レーダー — γ Trajectory Precursor Detection{ANSI['RESET']}")
    model_label = f"実LLM (Ollama {OLLAMA_MODEL})" if use_llm else "模擬データ"
    print(f"  モード: {model_label}")
    print("=" * 66)

    print_legend()

    # シナリオ選択 (カンマ区切り対応: "S1,S3,S6" or "all")
    if args.scenario.lower() == "all":
        selected = SCENARIOS
    else:
        ids = {x.strip().upper() for x in args.scenario.split(",")}
        selected = [s for s in SCENARIOS if s["id"] in ids]
        if not selected:
            print(f"Unknown scenario(s): {args.scenario}"); sys.exit(1)

    results = []
    for scenario in selected:
        res = run_scenario(scenario, use_llm=use_llm, verbose=verbose)
        results.append(res)

    crs = print_summary_table(results)

    # 保存
    def _ser(o):
        if isinstance(o, dict): return {k: _ser(v) for k, v in o.items()}
        if isinstance(o, list): return [_ser(v) for v in o]
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.bool_): return bool(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o

    summary = {"model": OLLAMA_MODEL, "crs": crs, "scenarios": _ser(results)}
    RESULT_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"  結果 → {RESULT_FILE}")
    print()

    # CI/CD 閾値チェック
    if args.threshold is not None:
        if crs < args.threshold:
            print(f"  [CI FAIL] CRS={crs:.3f} < threshold={args.threshold:.3f}  → exit 1")
            sys.exit(1)
        else:
            print(f"  [CI PASS] CRS={crs:.3f} >= threshold={args.threshold:.3f}  → exit 0")


def _run_compare_mode(args):
    """--compare model1,model2,... : 複数モデルの CRS を並べて比較。"""
    global OLLAMA_MODEL
    models = [m.strip() for m in args.compare.split(",")]
    verbose = not args.quiet

    # シナリオ選択
    if args.scenario.lower() == "all":
        selected = SCENARIOS
    else:
        ids = {x.strip().upper() for x in args.scenario.split(",")}
        selected = [s for s in SCENARIOS if s["id"] in ids]
        if not selected:
            print(f"Unknown scenario(s): {args.scenario}"); sys.exit(1)

    all_crs = {}
    for model in models:
        OLLAMA_MODEL = model
        print()
        print("=" * 66)
        print(f"  MODEL: {model}")
        print("=" * 66)
        print_legend()
        results = []
        for scenario in selected:
            res = run_scenario(scenario, use_llm=True, verbose=verbose)
            results.append(res)
        crs = print_summary_table(results)
        all_crs[model] = crs

    # 比較サマリ
    print()
    print("=" * 66)
    print(f"  {ANSI['BOLD']}モデル比較 — CRS ランキング{ANSI['RESET']}")
    print("=" * 66)
    ranked = sorted(all_crs.items(), key=lambda x: x[1], reverse=True)
    for rank, (model, crs) in enumerate(ranked, 1):
        bar = "█" * int(crs * 30) + "░" * (30 - int(crs * 30))
        print(f"  #{rank}  {model:<20} [{bar}] CRS={crs:.3f}")
    print()

    if args.threshold is not None:
        failed = [m for m, c in all_crs.items() if c < args.threshold]
        if failed:
            print(f"  [CI FAIL] threshold={args.threshold:.3f}  失敗モデル: {failed}")
            sys.exit(1)
        print(f"  [CI PASS] 全モデル CRS >= {args.threshold:.3f}")


if __name__ == "__main__":
    main()
