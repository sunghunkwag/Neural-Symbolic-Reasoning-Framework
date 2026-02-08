from aria.logic.dsl import filter_list, is_even, is_positive, logical_and, logical_or
'\nAlgorithmic Evolution Target.\nThis file contains a broken algorithm that the RSI engine should fix structurally.\n'

def process_data(data: list) -> list:
    return filter_list(filter_list(filter_list(data, lambda var_3_0: is_even(-4)), lambda var_2_0: is_even(var_2_0)), lambda var_1_0: logical_and(is_even(var_1_0), is_positive(var_1_0)))
if __name__ == '__main__':
    test_data = [1, -2, 3, 4, 0, 6, -8, 10]
    result = process_data(test_data)
    print(f'Result: {result}')