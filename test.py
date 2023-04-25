import paddle


# 测试linear原理

# def main():
#     t = paddle.randn([1, 3])
#     w = paddle.randn([3, 1])
#
#     print(w.shape)
#
#     print(t.shape)
#
#     s = paddle.nn.functional.linear(t, w)
#     print(s)


# 测试flatten原理

# def main():
#     t = paddle.randn([1, 2, 3])
#     print(t.shape)
#     t = t.flatten(2)
#     print(t.shape)


if __name__ == '__main__':
    main()
