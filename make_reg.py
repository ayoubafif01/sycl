import re
p = "GEMM_reg/gemm.cu"
s = open(p).read()
pat = re.compile(r"c\[i \* NJ \+ j\] \*= beta;\s*int k;\s*for\s*\(k=0; k < _PB_NK; k\+\+\)\s*\{\s*c\[i \* NJ \+ j\] \+= alpha \* a\[i \* NK \+ k\] \* b\[k \* NJ \+j\];\s*\}")
assert len(pat.findall(s)) == 1, "boucle du kernel introuvable"
new = ("DATA_TYPE acc = c[i * NJ + j] * beta;\n"
       "                for (int k = 0; k < _PB_NK; k++)\n"
       "                {\n"
       "                        acc += alpha * a[i * NK + k] * b[k * NJ + j];\n"
       "                }\n"
       "                c[i * NJ + j] = acc;")
open(p, "w").write(pat.sub(lambda m: new, s))
