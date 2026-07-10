import os
from collections import defaultdict, Counter
import regex as re  # type: ignore
import json

def train_bpe(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
) -> tuple[dict[int,bytes],list[tuple[bytes,bytes]]]:
# -- 1.初始化词表
    vocab = {i: bytes([i]) for i in range(256)}

    # 计算需要的合并次数
    num_merges = vocab_size - 256 - len(special_tokens)

#       --- 2. 读取语料，划分token ---
    with open(input_path,"r",encoding="utf-8") as f:
        text = f.read()
    """
    For special_tokens:
    在训练时，必须保证特殊 Token 不参与频率统计。
    代码逻辑：
        切割语料：在开始统计词频之前，利用正则将语料库在特殊 Token 处切开。
        独立统计：只对切分出来的普通文本片段进行 BPE 统计。
        最后加入：训练结束后，强制将特殊 Token 加入词表（通常放在最后），确保它们有 ID。
    """
    if special_tokens:
        special_regex = "|".join(re.escape(t) for t in special_tokens)
        # 使用split分割特殊token
        parts = re.split(f"({special_regex})",text)
        train_segments = [p for p in parts if p not in special_tokens]
    else:
        train_segments = [text]
    # print(train_segments)

    # --- 3. 预分词（Pre-tokenization）并统计词频 ---
    # 使用 GPT-2 的 BPE 预分词正则表达式。
    # GPT-2 正则表达式的作用是执行“预分词（Pre-tokenization）”。 它的规则是：
    #   (1)不允许跨越类型合并：比如它会把字母和标点符号分开。
    #   (2)保护空格：它通常会把单词前面的空格和单词连在一起，作为一个整体。
    # text = "Hello World test! ..."
    # 分割后 words = ['Hello', ' World', ' test', '!', ' ...']   
    gpt2_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    # raw_counts创建一个计数器计算各个token的出现频率
    # 单词被表现为字节元组： "hello" --> b'h',b'e',b'l',b'l',b'o' 
    raw_counts = Counter()
    for segment in train_segments:
        words = gpt2_pat.findall(segment)
        for word in words:
            word_list = tuple(bytes([b]) for b in word.encode("utf-8"))
            raw_counts[word_list] += 1

    """
    优化策略：通过构建高效的数据结构用于快速合并
    word_list: 存储每个单词的字节列表，
    counts_list: 存储对应单词的频率
    """
    words_list = []
    counts_list = []
    for word_tuple,freq in raw_counts.items():
        words_list.append(list(word_tuple))
        counts_list.append(freq)
    
    # defaultdict(int)创建一个带默认值0的字典，当访问字典中一个不存的值时不会报错
    # stats存储所有可能的相邻字节对（key），以及其出现频率（value）
    stats = defaultdict(int)

    # 倒排索引，存储 pair -> {包含该 pair的单词在words_list中的下标集和}
    indices = defaultdict(set)
    #  --- 初始化 stats 和 indices
    for idx,word in enumerate(words_list):
        freq = counts_list[idx]
        # print(word)
        for i in range(len(word)-1):
            pair = (word[i],word[i+1])
            stats[pair] += freq
            indices[pair].add(idx)

    merges = []

# --- 4.执行迭代流程
    # 执行num_merges次，每次找到并且应用一个最佳流程
    for _ in range(num_merges):
        if not stats:
            break
        # --- 4a.寻找最佳'pair'
        best_pair = max(stats.items(),key = lambda x : (x[1],x[0]))[0]
        if stats[best_pair] <= 0:
            break
        
        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]

        # ---4b.获取需要更新的单词，遍历并更新所有收到影响的单词
        # 使用倒排索引indices获取到所有包含'best_pair'的单词下标
        relevant_indices = list(indices[best_pair])

        # 遍历并更新所有受影响的单词、统计信息和倒排索引 
        for idx in relevant_indices:
            word = words_list[idx]
            freq = counts_list[idx]
            # 扫描当前的单词，找到所有 'best_pair' 的位置
            i = 0
            while i < len(word) - 1:
                if word[i] == best_pair[0] and word[i+1] == best_pair[1]:
                # 成功匹配 'best_pair' 执行合并

                # 1.更新旧邻居 Pair 的频率
                #  -- 左邻居 ：
                    if i > 0:
                        prev_pair = (word[i-1],word[i])
                        stats[prev_pair] -= freq
                        if stats[prev_pair] == 0:
                            del stats[prev_pair]
                
                #  -- 右邻居
                    if i < len(word) - 2:
                        next_pair = (word[i+1],word[i+2])
                        stats[next_pair] -= freq
                        if stats[next_pair] == 0:
                            del stats[next_pair]
                
                    # 2.更新单词 ，(word[i],word[i+1])替换成 new_token
                    word[i] = new_token
                    del word[i+1]

                    # 3.添加新产生的邻居 Pair 的频率和索引
                    if i > 0:
                        new_prev = (word[i-1],word[i])
                        stats[new_prev] += freq
                        indices[new_prev].add(idx) #添加到新pair的倒排索引

                    if i < len(word) - 1:
                        new_next = (word[i],word[i+1])
                        stats[new_next] += freq
                        indices[new_next].add(idx)
                
                else:
                    i += 1
        # 4d.清除已经完全合并的 'best_pair'
        if best_pair in stats: del stats[best_pair]
        if best_pair in indices: del indices[best_pair]

    #  --- 5.构建最终词条
    for pair in merges:
        new_id = len(vocab)
        vocab[new_id] = pair[0] + pair[1]
    # 添加特殊token
    for s_tok in special_tokens:
        s_bytes = s_tok.encode("utf-8")
        vocab[len(vocab)] = s_bytes
    return vocab,merges

def byte_to_unicode():
    """
    创建一组映射，将0-255字节映射为一组可见的unicode字符
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs,cs))

def save_tokenizer_files(vocab,merges,out_dir):
    os.makedirs(out_dir,exist_ok=True)

    # 初始化映射表
    byte_encoder = byte_to_unicode()

    # 词表保存，使用 byte_encoder 将 bytes 转换为可见字符串
    json_vocab={
        k: "".join(byte_encoder[b] for b in v)
        for k,v in vocab.items()
    }
    with open(os.path.join(out_dir,"vocab.json"),"w",encoding="utf-8") as f:
        json.dump(json_vocab,f,indent=4)
    
    with open(os.path.join(out_dir,"merges.json"),"w",encoding="utf-8") as f:
        for p1,p2 in merges:
            s1= "".join(byte_encoder[b] for b in p1)
            s2= "".join(byte_encoder[b] for b in p2)
            f.write(f"{s1} {s2}\n")
    
def main():
    input_path = "/home/pasical/projects/dl/LLM/CS336/assignment1-basics/data/demo2.txt"
    vocab_size = 10000

    special_tokens = ["<|endoftext|>"]
    output_dir = "/home/pasical/projects/dl/LLM/CS336/assignment1-basics/data/output"

    print(f"开始训练BPE分词器，目标词表大小{vocab_size}...")
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    save_tokenizer_files(vocab,merges,output_dir)
    
if __name__ == "__main__":
    main()