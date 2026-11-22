"""Chunk the typology documents and retrieve the ones that fit a case.

The index is small (8 documents, 56 chunks), so retrieval is a dense cosine
comparison against every chunk. The text encoder sits behind a two-method
interface (`fit`, `encode`). The default is TF-IDF from scikit-learn. A
sentence-transformers backend is included but is imported only when asked for.

The query is built from what the account did, never from the case label:
`describe_behaviour` renders the profile in plain words, and `build_query`
adds the fact lines, each weighted by its salience.

Retrieval answers two questions with the same vectors:

* which typology document does the case resemble most (`rank_typologies`),
* which individual chunks are closest (`search`, `indicators_for`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol, Sequence

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from .cases import Case, generate_cases
from .facts import compute_profile
from .typologies import Typology, load_typologies

# --------------------------------------------------------------------------
# Chunks
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    id: str  # "STR-00" for the description, "STR-01"... for indicators
    typology_id: str
    kind: str  # "description" or "indicator"
    title: str  # typology title
    text: str


def build_chunks(typologies: Sequence[Typology]) -> list[Chunk]:
    """One chunk for each description and one for each indicator."""
    chunks = []
    for typology in typologies:
        chunks.append(Chunk(f"{typology.id}-00", typology.id, "description", typology.title, typology.description))
        for indicator in typology.indicators:
            chunks.append(Chunk(indicator.id, typology.id, "indicator", typology.title, indicator.text))
    return chunks


# --------------------------------------------------------------------------
# Encoders
# --------------------------------------------------------------------------


class EncoderBackend(Protocol):
    """Anything that can turn texts into row vectors.

    `fit` receives the chunk texts and, optionally, a second collection of
    texts from which term statistics may be estimated. A pretrained encoder
    can ignore both.
    """

    def fit(self, texts: Sequence[str], idf_corpus: Sequence[str] | None = None) -> None: ...

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


_NUMBER_RE = re.compile(r"\$?(\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
_WORD_RE = re.compile(r"[a-z]+")


def _stem(token: str) -> str:
    """A very small suffix stripper so that 'deposits' meets 'deposited'.

    It is cruder than a real stemmer, but it has no dependency and its mistakes
    are the same on both sides of the comparison.
    """
    if len(token) < 4 or token in ENGLISH_STOP_WORDS:
        return token
    if token.endswith("ies"):
        token = token[:-3] + "y"
    elif token.endswith(("ches", "shes", "sses", "xes")):
        token = token[:-2]
    elif token.endswith("s") and not token.endswith(("ss", "us", "is")):
        token = token[:-1]
    for suffix in ("ing", "ed"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[: -len(suffix)]
            break
    if token.endswith("e") and len(token) >= 4:
        token = token[:-1]
    return token


def normalise(text: str, stem: bool = True) -> str:
    """Lowercase, write amounts as plain integers and optionally stem words."""
    text = text.lower()
    text = _NUMBER_RE.sub(lambda m: m.group(1).replace(",", ""), text)
    if stem:
        text = _WORD_RE.sub(lambda m: _stem(m.group(0)), text)
    return text


class TfidfBackend:
    """TF-IDF over word unigrams and bigrams.

    The vocabulary always comes from the chunk texts. The inverse document
    frequencies come from `idf_corpus` when one is given. With only 56 chunks
    to count over, a word such as "wire" looks rarer than it is, so I let the
    retriever pass in a few hundred unlabelled case descriptions as well: a
    behaviour that most alerts show then carries little weight, and one that
    few alerts show carries a lot.
    """

    name = "tfidf"

    def __init__(self, stem: bool = True):
        self.stem = stem
        self._vectorizer: TfidfVectorizer | None = None

    def _make(self, vocabulary=None) -> TfidfVectorizer:
        return TfidfVectorizer(
            preprocessor=lambda text: normalise(text, stem=self.stem),
            stop_words=sorted(ENGLISH_STOP_WORDS),
            ngram_range=(1, 2),
            sublinear_tf=True,
            vocabulary=vocabulary,
        )

    def fit(self, texts: Sequence[str], idf_corpus: Sequence[str] | None = None) -> None:
        vocabulary = self._make().fit(texts).vocabulary_
        self._vectorizer = self._make(vocabulary)
        self._vectorizer.fit(list(texts) + list(idf_corpus or []))

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._vectorizer is None:
            raise RuntimeError("call fit() before encode()")
        return self._vectorizer.transform(texts).toarray()


class SentenceTransformerBackend:
    """Optional neural encoder. Needs `pip install sentence-transformers`.

    It is not on the default path: it downloads a model on first use, and the
    main results table in the README was produced with TF-IDF.
    """

    name = "sentence-transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on an optional package
            raise ImportError(
                "the sentence-transformers backend needs the optional package: "
                "pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(model_name)

    def fit(self, texts: Sequence[str], idf_corpus: Sequence[str] | None = None) -> None:
        return None  # pretrained, nothing to fit

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self._model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False))


def make_backend(name: str = "tfidf", **kwargs) -> EncoderBackend:
    if name == "tfidf":
        return TfidfBackend(**kwargs)
    if name == "sentence-transformers":
        return SentenceTransformerBackend(**kwargs)
    raise ValueError(f"unknown backend {name!r}")


# --------------------------------------------------------------------------
# Retriever
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Query:
    """A query is a list of lines with one weight per line."""

    lines: tuple[str, ...]
    weights: tuple[float, ...]

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    @classmethod
    def from_text(cls, text: str) -> "Query":
        return cls((text,), (1.0,))


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float

    def to_dict(self) -> dict:
        return {"chunk_id": self.chunk.id, "typology_id": self.chunk.typology_id,
                "score": round(self.score, 4), "text": self.chunk.text}


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


RANKING_METHODS = ("document", "chunks")
BACKGROUND_SEED = 900_000  # far from any seed used for evaluation or examples
BACKGROUND_SIZE = 400


class Retriever:
    """Cosine similarity between a query and the typology chunks and documents."""

    TOP_CHUNKS = 3  # used by the "chunks" ranking method

    def __init__(self, chunks: Sequence[Chunk], backend: EncoderBackend | None = None,
                 idf_corpus: Sequence[str] | None = None, ranking: str = "document"):
        if ranking not in RANKING_METHODS:
            raise ValueError(f"ranking must be one of {RANKING_METHODS}")
        self.chunks = list(chunks)
        self.ranking = ranking
        self.backend = backend or TfidfBackend()
        texts = [c.text for c in self.chunks]
        self.backend.fit(texts, idf_corpus)
        self._chunk_matrix = _unit_rows(self.backend.encode(texts))
        self.typology_ids = list(dict.fromkeys(c.typology_id for c in self.chunks))
        documents = [" ".join(c.text for c in self.chunks if c.typology_id == tid) for tid in self.typology_ids]
        self._document_matrix = _unit_rows(self.backend.encode(documents))

    @classmethod
    def from_directory(cls, directory=None, backend: EncoderBackend | None = None,
                       background: bool = True, ranking: str = "document") -> "Retriever":
        """Build a retriever over the typology documents in `directory`."""
        chunks = build_chunks(load_typologies(directory))
        return cls(chunks, backend, idf_corpus=background_corpus() if background else None, ranking=ranking)

    def query_vector(self, query: Query | str) -> np.ndarray:
        """Weighted sum of unit line vectors, so one long line cannot dominate."""
        if isinstance(query, str):
            query = Query.from_text(query)
        if not query.lines:
            return np.zeros((1, self._chunk_matrix.shape[1]))
        lines = _unit_rows(self.backend.encode(list(query.lines)))
        weights = np.asarray(query.weights, dtype=float)[:, None]
        return _unit_rows((weights * lines).sum(axis=0, keepdims=True))

    def chunk_scores(self, query: Query | str) -> np.ndarray:
        return (self._chunk_matrix @ self.query_vector(query).T).ravel()

    def search(self, query: Query | str, k: int = 5) -> list[Hit]:
        """The `k` best chunks, best first. Ties keep document order."""
        scores = self.chunk_scores(query)
        order = np.argsort(-scores, kind="stable")[:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in order]

    def rank_typologies(self, query: Query | str, method: str | None = None) -> list[tuple[str, float]]:
        """Rank the typologies for a query, best first.

        `document` compares the query with each whole typology document.
        `chunks` sums each typology's three best chunk scores. I kept both
        because the comparison between them is part of the evaluation.
        """
        method = method or self.ranking
        if method == "document":
            scores = (self._document_matrix @ self.query_vector(query).T).ravel()
            totals = dict(zip(self.typology_ids, (float(s) for s in scores)))
        elif method == "chunks":
            chunk_scores = self.chunk_scores(query)
            totals = {}
            for tid in self.typology_ids:
                own = sorted((float(s) for c, s in zip(self.chunks, chunk_scores) if c.typology_id == tid),
                             reverse=True)
                totals[tid] = sum(own[: self.TOP_CHUNKS])
        else:
            raise ValueError(f"method must be one of {RANKING_METHODS}")
        position = {tid: i for i, tid in enumerate(self.typology_ids)}
        return sorted(totals.items(), key=lambda kv: (-kv[1], position[kv[0]]))

    def indicators_for(self, query: Query | str, typology_id: str, k: int = 3) -> list[Hit]:
        """The best `k` indicator chunks within one typology."""
        scores = self.chunk_scores(query)
        hits = [Hit(c, float(s)) for c, s in zip(self.chunks, scores)
                if c.typology_id == typology_id and c.kind == "indicator"]
        return sorted(hits, key=lambda h: -h.score)[:k]

    def chunk(self, chunk_id: str) -> Chunk | None:
        for chunk in self.chunks:
            if chunk.id == chunk_id:
                return chunk
        return None


@lru_cache(maxsize=4)
def background_corpus(n: int = BACKGROUND_SIZE, seed: int = BACKGROUND_SEED) -> tuple[str, ...]:
    """Query texts of `n` unlabelled synthetic cases, used only for term statistics.

    The cases are drawn with random typologies and their labels are discarded.
    In a bank the equivalent would be last quarter's alert descriptions.
    """
    return tuple(build_query(case).text for case in generate_cases(n, seed=seed, balanced=False))


@lru_cache(maxsize=2)
def default_retriever() -> Retriever:
    """The retriever used by the CLI: TF-IDF, background IDF, document ranking."""
    return Retriever.from_directory()


# --------------------------------------------------------------------------
# Query construction
# --------------------------------------------------------------------------


def describe_behaviour(case: Case) -> list[str]:
    """Describe what the account did in plain words.

    Each sentence is switched on by a threshold on the profile (for example
    "at least half of the cash deposits were made outside the home state").
    The wording uses ordinary banking nouns such as cash, branch, wire and
    invoice, because there is no other honest way to say these things, but it
    does not reuse the phrasing of the typology indicators.
    """
    p = compute_profile(case)
    subject = case.subject
    out: list[str] = []

    if subject["kind"] == "individual":
        out.append(f"The account holder is a private individual who works as a {subject.get('occupation')}."
                   if not str(subject.get("occupation", "")).startswith(("retired", "unemployed", "student"))
                   else f"The account holder is a private individual, {subject.get('occupation')}.")
        if subject.get("age", 0) >= 65:
            out.append("The customer is an older person.")
        elif subject.get("age", 99) <= 25:
            out.append("The customer is a young adult.")
    else:
        out.append("The account belongs to a company.")
        if p["company_age_days"] is not None and p["company_age_days"] < 365:
            out.append("The company was registered only a few months ago.")
        if p["everyday_n"] == 0:
            out.append("The account shows no payroll, rent, utility or card spending.")
    if p["account_age_days"] < 120:
        out.append("The account was opened shortly before the activity began.")
    if p["expected_monthly"] and p["total_in"] > 5 * p["expected_monthly"] * p["months"]:
        out.append("Turnover was many times higher than the customer told the bank to expect.")

    n = p["cash_in_n"]
    if n >= 4:
        out.append("Cash was paid in at the counter many times.")
        if p["cash_in_band_n"] / n >= 0.5:
            out.append("Most of the cash deposits were for a little under $10,000 each.")
        if p["cash_multi_days"] >= 1:
            out.append("On some days cash was deposited more than once.")
        if len(p["cash_in_sites"]) >= 3:
            out.append("Several different branches were used.")
        if p["cash_in_away_n"] / n >= 0.5:
            out.append("Most of the cash was deposited in other states, a long way from the customer's home branch.")
        if p["cash_in_third_party_n"] / n >= 0.5:
            out.append("Much of the cash was paid in by people who are not the account holder.")
    if p["cash_out_n"] >= 3:
        out.append("Cash was withdrawn many times.")
        if p["cash_out_band_n"] / p["cash_out_n"] >= 0.5:
            out.append("Most of the withdrawals were for a little under $10,000 each.")

    if p["person_in_parties"] >= 5:
        out.append("Many different individuals sent money to the account, and nothing links them to each other "
                   "or to the customer.")
    if p["exchange_in_n"] >= 2:
        out.append("Money arrived from cryptocurrency exchanges.")
    if p["exchange_out_n"] >= 1:
        out.append("Money was sent to a cryptocurrency exchange.")
    if p["wires_in_n"] >= 3 and p["wires_out_n"] >= 3:
        out.append("Large wires came in and large wires went out.")
    if p["foreign_places"]:
        out.append("Some wires went to or came from other countries.")
    if p["wires_n"] >= 4 and p["round_wires_n"] / p["wires_n"] >= 0.6:
        out.append("Most of the wire amounts were round numbers.")

    flow = p["flow_through_pct"]
    if p["lag_n"] >= 3 and p["lag_median"] <= 1 and flow is not None and flow >= 85:
        out.append("Money left the account within a day or so of arriving, and very little stayed behind.")
    elif flow is not None and flow >= 85:
        out.append("Nearly all the money that came in was paid out again during the period.")

    if p["new_payees"] and p["prior_avg_debits"] and p["total_out"] > 3 * p["prior_avg_debits"] * p["months"]:
        out.append("Spending rose far above the customer's usual level and went to payees the customer had "
                   "never paid before.")
    if p["gift_cards_n"] >= 2:
        out.append("The customer bought gift cards several times.")
    if p["savings_in_n"]:
        out.append("Savings were moved into the account to fund the payments.")

    if p["invoices_n"]:
        out.append("The customer holds invoices for goods sold to buyers in other countries.")
        if p["third_party_payments_n"]:
            out.append("Some invoice payments came from companies that do not appear on any invoice.")
        if p["goods_mismatch"]:
            out.append("The goods on the invoices are not what this company normally sells.")
        if p["invoices_total"] and abs(p["invoice_payments_total"] - p["invoices_total"]) / p["invoices_total"] > 0.1:
            out.append("The payments received do not match the invoice totals.")
    if p["shared_address"]:
        out.append("Some counterparties are registered at the same address as the customer.")
    if p["returned_n"]:
        out.append("Some money was wired back to a company that had earlier sent money in.")
    return out


QUERY_MODES = ("both", "facts", "behaviour", "naive")


def build_query(case: Case, mode: str = "both") -> Query:
    """Build the retrieval query for a case. The case label is never used.

    * `behaviour`: the plain-language description, one line per sentence.
    * `facts`: the fact lines, each weighted by its salience; facts with zero
      weight (who, which account, which dates) are left out.
    * `both` (default): the two together.
    * `naive`: every fact line and the description as one unweighted string.
      It exists as a baseline for the evaluation.
    """
    if mode not in QUERY_MODES:
        raise ValueError(f"mode must be one of {QUERY_MODES}")
    behaviour = describe_behaviour(case)
    if mode == "naive":
        return Query.from_text("\n".join(behaviour + [fact.text() for fact in case.facts]))
    lines: list[str] = []
    weights: list[float] = []
    if mode in ("both", "behaviour"):
        lines.extend(behaviour)
        weights.extend([1.0] * len(behaviour))
    if mode in ("both", "facts"):
        for fact in case.facts:
            if fact.weight > 0:
                lines.append(fact.text())
                weights.append(fact.weight)
    return Query(tuple(lines), tuple(weights))
