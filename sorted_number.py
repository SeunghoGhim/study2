"""number = list(map(int, input().split( )))

number.sort()

print(number[3])
"""

a, b = map(int, input().split())

if a < b and  0 < b < 11:
    print(b*"*")
elif b < a and 0 < a < 11:
    print(a*"*")
else:
    print("1이상 10이하의 숫자를 입력해주세요.")
