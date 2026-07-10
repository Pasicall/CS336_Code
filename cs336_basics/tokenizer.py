import regex as re # type: ignore
from collections.abc import Iterable

class BPETokenizer:
    def __init__(self, vocab: dict[int,bytes], merges: list[tuple[bytes,bytes]], special_tokens: list[str] | None = None):
        # 1.建立双向映射表
        self.vocab = vocab
        self.id_to_byte = vocab
        self.byte_to_id = {v: k for k , v in vocab.items()}

        # 2.将合并规则转化为Rank字典
        self.merge = {pair:i for i,pair in enumerate(merges)}

        self.special_tokens = special_tokens or []

        # 3.特殊token的正则表达
        if self.special_tokens:
            # 将长度从长到短排列，以便于优先匹配最长的特殊标记
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            special_pattern = "|".join(re.escape(t) for t in sorted_special)
            self.special_regex = re.compile(special_pattern)

        else:
            self.special_regex = None 
        
        self.gpt2_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
    
    def encode(self, text: str) -> list[int]:
        if not text:
            return None
        
