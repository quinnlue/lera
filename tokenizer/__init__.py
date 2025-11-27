from tokenizer._deprecated_tokenizer import Tokenizer, base, MIN_INT, MAX_INT

# Create the tokenizer instance
tok = Tokenizer(base_tokenizer=base, min_int=MIN_INT, max_int=MAX_INT, add_expr_markers=True)

__all__ = ["tok"]