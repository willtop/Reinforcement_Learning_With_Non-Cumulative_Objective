import pstats
from pstats import SortKey
p = pstats.Stats('../timefile')
p.strip_dirs().sort_stats('tottime').print_stats(20)