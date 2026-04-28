def f(x):
    return x ** 2 - 4


def bisection(left, right, eps):
    if f(left) * f(right) > 0:
        print("Bisection method cannot be used - same signs at interval ends.")
        return None

    iter = 0

    while right - left > eps:
        iter += 1
        mid = (left + right) / 2

        print(f"Iteration {iter}: current root approximation = {mid}")

        if f(mid) == 0:
            return mid

        if f(left) * f(mid) < 0:
            right = mid
        else:
            left = mid

    return (left + right) / 2


slop = bisection(1, 5, 0.01)

print("Root for this function:", slop)
print("Value of the function at root:", f(slop))
#The theoretical worst-case number of iterations is 9. 
#In this code, the algorithm stops earlier because the midpoint becomes exactly equal to the root.