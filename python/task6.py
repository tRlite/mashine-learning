def check(s, filename):
    d = {}
    for word in s.lower().split():
      if word in d:
        d[word] += 1
      else:
        d[word] = 1
    dct = dict(sorted(d.items()))
    with open(filename, 'w') as file:
      keys = list(dct.keys())
      if (len(keys) == 1):
        file.write(f'{keys[0]} {d[keys[0]]}')
        return
      for key in keys:
        file.write(f'{key} {d[key]}\n')
