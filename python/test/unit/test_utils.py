import pytest

import triton
from triton._utils import is_power_of_two, validate_block_shape


def test_is_power_of_two():
    assert is_power_of_two(1)
    assert is_power_of_two(2)
    assert is_power_of_two(8)
    assert is_power_of_two(1024)
    # 0 is not a power of two; x & (x - 1) == 0 alone wrongly accepts it.
    assert not is_power_of_two(0)
    assert not is_power_of_two(3)
    assert not is_power_of_two(6)
    assert not is_power_of_two(-4)


def test_next_power_of_2_contract():
    # Docstring: "smallest power of 2 >= n" — must always be a power of 2 and >= n.
    for n, expected in [(1, 1), (2, 2), (3, 4), (5, 8), (8, 8), (17, 32), (100, 128), (1000, 1024)]:
        assert triton.next_power_of_2(n) == expected
    # n <= 0 has no positive lower bound, so the smallest power of 2 >= n is 1
    # (used to return 0, which is neither a power of 2 nor >= n).
    for n in [0, -1, -5, -100]:
        assert triton.next_power_of_2(n) == 1
    for n in range(-8, 65):
        r = triton.next_power_of_2(n)
        assert is_power_of_two(r) and r >= n


def test_validate_block_shape_rejects_zero():
    # validate_block_shape promises every element is a power of 2, but a 0
    # element used to slip through because is_power_of_two(0) returned True.
    with pytest.raises(ValueError, match="must be a power of 2"):
        validate_block_shape([0])
    with pytest.raises(ValueError, match="must be a power of 2"):
        validate_block_shape([8, 0])


def test_validate_block_shape_accepts_powers_of_two():
    assert validate_block_shape([8, 16]) == 128
