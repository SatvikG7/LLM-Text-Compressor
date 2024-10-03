import math
from typing import List, Tuple
import struct

class ArithmeticCoder:
    def __init__(self, precision=32):
        self.precision = precision
        self.one = 1 << precision
        self.quarter = self.one // 4
        self.half = self.one // 2
        self.threequarters = self.one - self.quarter

    def get_frequency(self, ranks):
        freq = {}
        for rank in ranks:
            freq[rank] = freq.get(rank, 0) + 1
        return freq

    def encode(self, ranks: List[int]) -> Tuple[int, dict]:
        freq = self.get_frequency(ranks)
        cumulative_freq = {}
        total = 0
        for rank in sorted(freq.keys()):
            cumulative_freq[rank] = total
            total += freq[rank]

        low, high = 0, self.one
        for rank in ranks:
            range_size = high - low
            high = low + range_size * (cumulative_freq[rank] + freq[rank]) // total
            low = low + range_size * cumulative_freq[rank] // total

            while True:
                if high < self.half:
                    pass
                elif low >= self.half:
                    low -= self.half
                    high -= self.half
                elif low >= self.quarter and high < self.threequarters:
                    low -= self.quarter
                    high -= self.quarter
                else:
                    break
                low *= 2
                high *= 2

        return low, freq

    def decode(self, encoded: int, total_ranks: int, freq: dict) -> List[int]:
        cumulative_freq = {}
        total = sum(freq.values())
        current_total = 0
        for rank in sorted(freq.keys()):
            cumulative_freq[rank] = current_total
            current_total += freq[rank]

        decoded_ranks = []
        value = encoded
        low, high = 0, self.one

        for _ in range(total_ranks):
            range_size = high - low
            scaled_value = ((value - low + 1) * total - 1) // range_size

            rank = None
            for r in sorted(cumulative_freq.keys()):
                if cumulative_freq[r] <= scaled_value < cumulative_freq[r] + freq[r]:
                    rank = r
                    break

            decoded_ranks.append(rank)

            high = low + range_size * (cumulative_freq[rank] + freq[rank]) // total
            low = low + range_size * cumulative_freq[rank] // total

            while True:
                if high < self.half:
                    pass
                elif low >= self.half:
                    low -= self.half
                    high -= self.half
                    value -= self.half
                elif low >= self.quarter and high < self.threequarters:
                    low -= self.quarter
                    high -= self.quarter
                    value -= self.quarter
                else:
                    break
                low *= 2
                high *= 2
                value *= 2

        return decoded_ranks

def encode_and_store(ranks: List[int], filename: str):
    coder = ArithmeticCoder()
    encoded, freq = coder.encode(ranks)
    with open(filename, 'wb') as f:
        # Store the total number of ranks, the encoded value, and the frequency dictionary
        f.write(struct.pack('!QQ', len(ranks), encoded))
        f.write(struct.pack('!I', len(freq)))
        for rank, count in freq.items():
            f.write(struct.pack('!II', rank, count))

def read_and_decode(filename: str) -> List[int]:
    with open(filename, 'rb') as f:
        total_ranks, encoded = struct.unpack('!QQ', f.read(16))
        freq_len = struct.unpack('!I', f.read(4))[0]
        freq = {}
        for _ in range(freq_len):
            rank, count = struct.unpack('!II', f.read(8))
            freq[rank] = count
    
    coder = ArithmeticCoder()
    return coder.decode(encoded, total_ranks, freq)
