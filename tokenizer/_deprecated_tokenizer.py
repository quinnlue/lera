
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import re
from typing import List, Tuple, Union
from transformers import AutoTokenizer, PreTrainedTokenizerBase

MATH_CHARS = set("0123456789+-*/=() ")
MIN_INT = -100
MAX_INT = 500

base = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0125-SFT", use_fast=True)

# Precompile regex for splitting math/text spans - much faster than char-by-char
_MATH_TEXT_SPLIT_RE = re.compile(r'([0-9+\-*/=() ]+|[^0-9+\-*/=() ]+)')
_DIGIT_CHECK_RE = re.compile(r'\d')

def split_math_text_spans(text: str) -> List[Tuple[bool, str]]:
    """Split text into math and non-math spans using regex (much faster)."""
    if not text:
        return []

    spans = []
    for span in _MATH_TEXT_SPLIT_RE.findall(text):
        if not span:
            continue
        # Check if first char suggests math span
        if span[0] in MATH_CHARS:
            # Validate: must contain a digit to be math
            if _DIGIT_CHECK_RE.search(span):
                spans.append((True, span))
            else:
                spans.append((False, span))
        else:
            spans.append((False, span))
    
    return spans


# Optimized: use findall instead of repeated match() in a loop
_MATH_TOKEN_RE = re.compile(r'-?\d+|[+\-*/=()]')

def tokenize_math_expr(expr: str) -> List[str]:
    """
    Turn a math expression string like '-47 * -2 - 35 * -19 = 759'
    into ['-47', '*', '-2', '-', '35', '*', '-19', '=', '759'].
    Optimized to use findall for ~10x speed improvement.
    
    Note: Normalizes integers to canonical form (e.g., '05' -> '5').
    """
    # Quick checks
    if "--" in expr:
        raise ValueError("Invalid unary sequence '--'")
    
    # Extract all tokens at once (much faster than loop)
    raw_tokens = _MATH_TOKEN_RE.findall(expr)
    
    # Normalize and validate integers
    normalized_tokens = []
    for token in raw_tokens:
        if token[0].isdigit() or (token[0] == '-' and len(token) > 1):
            # Integer token: validate range and normalize (remove leading zeros)
            val = int(token)
            if val < MIN_INT or val > MAX_INT:
                raise ValueError("Integer out of allowed range")
            # Use canonical form (e.g., '05' becomes '5')
            normalized_tokens.append(str(val))
        else:
            # Operator/paren: keep as-is
            normalized_tokens.append(token)
    
    return normalized_tokens


class Tokenizer:
    """
    Hybrid tokenizer:
      - Uses a base HF tokenizer for natural language.
      - Uses custom integer/operator tokens for math spans.

    Math spans are detected heuristically via MATH_CHARS.
    """

    def __init__(
        self,
        base_tokenizer: PreTrainedTokenizerBase = base,
        min_int: int = -500,
        max_int: int = 500,
        add_expr_markers: bool = False,
        extra_special_tokens: List[str] = None,
    ):
        """
        base_tokenizer: HF tokenizer name or instance (e.g. "meta-llama/Llama-3-8B-Instruct").
        min_int, max_int: inclusive integer range for atomic number tokens.
        add_expr_markers: if True, adds <EXPR_START> and <EXPR_END> tokens around math spans.
        extra_special_tokens: optional list of extra tokens (e.g. <FACT_1>, <READY_1>).
        """
        if isinstance(base_tokenizer, str):
            if AutoTokenizer is None:
                raise ImportError("transformers is required to load a tokenizer by name")
            self.base = AutoTokenizer.from_pretrained(base_tokenizer, use_fast=True)
        else:
            self.base = base_tokenizer

        self.min_int = min_int
        self.max_int = max_int
        self.add_expr_markers = add_expr_markers

        # Build list of tokens to add to the base tokenizer
        new_tokens = []

        # Operators / parens
        self.op_tokens = ["+", "-", "*", "/", "=", "(", ")"]
        new_tokens.extend(self.op_tokens)

        # Integers as atomic tokens
        self.int_tokens = [str(i) for i in range(min_int, max_int + 1)]
        new_tokens.extend(self.int_tokens)

        # Optional expression markers and extra specials
        self.expr_start = "<EXPR_START>"
        self.expr_end = "<EXPR_END>"
        self.special_extra = extra_special_tokens or []

        if add_expr_markers:
            new_tokens.extend([self.expr_start, self.expr_end])

        new_tokens.extend(self.special_extra)

        # Add to base tokenizer vocab
        # NOTE: add_tokens will only add tokens that aren't already in vocab
        self.base.add_tokens(new_tokens)

        # For convenience, keep the token->id mapping
        self._update_maps()

    def _update_maps(self):
        self.token_to_id = self.base.get_vocab()
        # Reverse map
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        
        # Cache math token IDs for faster lookup
        self._math_token_ids = {}
        for token in self.int_tokens + self.op_tokens:
            if token in self.token_to_id:
                self._math_token_ids[token] = self.token_to_id[token]
        if self.add_expr_markers:
            if self.expr_start in self.token_to_id:
                self._math_token_ids[self.expr_start] = self.token_to_id[self.expr_start]
            if self.expr_end in self.token_to_id:
                self._math_token_ids[self.expr_end] = self.token_to_id[self.expr_end]

    # ------------- public API -------------

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text into a single list of token ids, with math spans
        tokenized using atomic integer/operator tokens.
        """
        spans = split_math_text_spans(text)
        all_ids: List[int] = []

        for is_math, span in spans:
            if not span:
                continue

            if is_math:
                # math span: tokenize via our math rules
                try:
                    math_tokens = tokenize_math_expr(span)
                except ValueError:
                    # fallback: this isn't valid math for our domain → treat as text
                    ids = self.base.encode(span, add_special_tokens=False)
                    all_ids.extend(ids)
                    continue

                if self.add_expr_markers:
                    math_tokens = [self.expr_start] + math_tokens + [self.expr_end]
                
                # Use cached IDs for faster lookup (avoids dict lookup overhead)
                ids = [self._math_token_ids[tok] for tok in math_tokens]
                all_ids.extend(ids)
            else:
                # normal text span: delegate to base tokenizer
                ids = self.base.encode(
                    span,
                    add_special_tokens=False,  # we handle global special tokens outside
                )
                all_ids.extend(ids)

        if add_special_tokens and hasattr(self.base, "bos_token_id") and hasattr(self.base, "eos_token_id"):
            # Very simple: wrap with BOS/EOS if they exist
            bos = [] if self.base.bos_token_id is None else [self.base.bos_token_id]
            eos = [] if self.base.eos_token_id is None else [self.base.eos_token_id]
            return bos + all_ids + eos

        return all_ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token ids back to text via the base tokenizer.
        For math tokens, since we used actual string forms as tokens,
        this round-trips cleanly.
        """
        return self.base.decode(ids, skip_special_tokens=skip_special_tokens)

    # Convenience helpers if you want to bypass span splitting:

    def encode_math_only(self, expr: str) -> List[int]:
        """Encode a pure math expression string."""
        tokens = tokenize_math_expr(expr)
        if self.add_expr_markers:
            tokens = [self.expr_start] + tokens + [self.expr_end]
        return [self._math_token_ids[tok] for tok in tokens]

    def encode_text_only(self, text: str) -> List[int]:
        """Encode pure natural language through the base tokenizer."""
        return self.base.encode(text, add_special_tokens=False)




if __name__ == "__main__":
    tok = Tokenizer(base_tokenizer=base, min_int=MIN_INT, max_int=MAX_INT, add_expr_markers=True)
    expr = "-47 * -2 = 79"
    ids = tok.encode(expr)
    print(ids)
    print(tok.decode(ids))



