import numpy as np
import numba as nb


@nb.jit()
def random_func(random_list):
    split_list = []
    for idx in random_list:
        if idx > 1 and (idx < len(random_list)-1):
            if idx % 2 == 0:  # even
                if idx not in split_list and idx - 1 not in split_list:
                    split_list.append(idx-1)
                    split_list.append(idx)

            else:  # odd
                if idx not in split_list and idx + 1 not in split_list:
                    split_list.append(idx)
                    split_list.append(idx+1)
    return split_list


random_list = np.random.choice(range(100000), 100000, replace=False)
out = random_func(random_list)

# print(random_list)
# print(split_list)
print(random_list)
print(out)
