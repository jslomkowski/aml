
klas = []
block = []
for p in pipeline.steps:
    klas.append(p[0])
    block.append(''.join([i for i in p[0] if not i.isdigit()]))
