from typing import List


def hello(s = None) -> str:
    if not s:
      return "Hello!"
    else:
      return f'Hello, {s}!'

    
def int_to_roman(x: int) -> str:
  res = ""
  for i in range (x // 1000):
    res += 'M'
  x %= 1000
  if x // 100 == 9:
    res += 'C'
    res += 'M'
    x -= 900
  elif x >= 500:
    res += 'D'
    x -= 500
  elif x // 100 == 4:
    res += 'C'
    res += 'D'
    x -= 400
  for i in range(x // 100):
    res += 'C'    
  x %= 100
  if x // 10 == 9:
    res += 'X'
    res += 'C'
    x -= 90
  elif x >= 50:
    res += 'L'
    x -= 50
  elif x // 10 == 4:
    res += 'X'
    res += 'L'
    x -= 40;
  for i in range(x // 10):
    res += 'X'
  x %= 10
  if x == 9:
    res += 'I'
    res += 'X'
    return res
  if x >= 5:
    res += 'V'
    x -= 5
  elif x == 4:
    res += 'I'
    res += 'V'
    return res
  for i in range(x):
    res += 'I'
  return res 


def longest_common_prefix(x: List[str]) -> str:
  if len(x) == 0:
    return ''
  X = [elem.strip() for elem in x]
  prefix = X[0]
  for i in range(len(prefix)):
    for j in range(1, len(x)):
      if i >= len(X[j]) or X[j][i] != prefix[i]:
        return prefix[:i]
  return prefix


class BankCard:
  def __init__(self, total_sum, balance_limit = -1):
    self.total_sum  = total_sum
    self.balance_limit = balance_limit
  def __call__(self, sum_spent):
    if sum_spent > self.total_sum:
      print(f"Not enough money to spent {sum_spent} dollars.")
      raise ValueError
    else:
      self.total_sum -= sum_spent
      print(f'You spent {sum_spent} dollars.')
  def __repr__(self):
    return "To learn the balance call balance."
  def __getattr__(self, balance): 
    if self.balance_limit == 0:
      print("Balance check limits exceeded.")
      raise ValueError
    self.balance_limit -= 1
    return self.total_sum
  def put(self, sum_put):
    self.total_sum += sum_put
    print(f'You put {sum_put} dollars.')
  def __add__(self, x):
    self.total_sum += x.total_sum
    self.balance_limit = max (self.balance_limit, x.balance_limit)
    return self


def primes() -> int:
  yield 2
  x = 3 
  while True: 
    i = 2
    while i <= int(x**0.5):
        if x % i == 0:
            x += 2
            i = 1
        i += 1
    yield x 
    x += 2
