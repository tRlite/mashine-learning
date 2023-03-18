def find_shortest(l):
    import re
    return min(map(len,re.findall("[A-Za-z]+",l)),default=0)
