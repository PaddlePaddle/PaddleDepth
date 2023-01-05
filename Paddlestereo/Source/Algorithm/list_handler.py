# -*- coding: utf-8 -*-
class ListHandler(object):
    """docstring for ListHandler"""

    def __init__(self):
        super().__init__()

    @staticmethod
    def list_add(list_a: list, list_b: list) -> list:
        assert len(list_a) == len(list_b)
        return [item + list_b[i] for i, item in enumerate(list_a)]

    @staticmethod
    def list_div(list_a: list, num: float) -> list:
        return [item / num for _, item in enumerate(list_a)]

    @staticmethod
    def list_mean(list_a: list) -> list:
        return [item for _, item in enumerate(list_a)]

    @staticmethod
    def double_list_add(list_a: list, list_b: list = None) -> list:
        assert isinstance(list_a, list) and isinstance(list_a[0], list)
        if list_b is None:
            return list_a
        for i, item in enumerate(list_a):
            list_a[i] = ListHandler.list_add(item, list_b[i])
        return list_a

    @staticmethod
    def double_list_div(list_a: list, num: float) -> list:
        res = []
        for item in list_a:
            tem_res = ListHandler.list_div(item, num)
            res.append(tem_res)

        return res
