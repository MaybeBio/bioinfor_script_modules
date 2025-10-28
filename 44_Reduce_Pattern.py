# 将一个冗余的长序列/符号字符串, reduce简并为简短的字符串
# 例, +++++++------++++, reduce成+-+(同类连续字符归并)

def reduce_pattern(s: str) -> str:
            if not s:
                return s
            out = [s[0]]
            for ch in s[1:]:
                if ch != out[-1]:
                    out.append(ch)
            return "".join(out)
