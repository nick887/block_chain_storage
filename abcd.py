import pstats

p = pstats.Stats('output_file')

p.strip_dirs().sort_stats('cumulative').print_stats(10)
