from srctools import Angle
from srctools.test import *


VALID_NUMS = [
    # Make sure we use double-precision with the precise small value.
    30, 1.5, 0.2827, 282.34645456362782677821,
]
VALID_NUMS += [-x for x in VALID_NUMS]

VALID_ZERONUMS = VALID_NUMS + [0, -0]


def test_construction(py_c_vec):
    """Check various parts of the constructor - Vec(), Vec.from_str()."""
    Vec, Angle, Matrix, parse_vec_str = py_c_vec
    
    for pit, yaw, rol in iter_vec(VALID_ZERONUMS):
        assert_ang(Angle(pit, yaw, rol), pit, yaw, rol)
        assert_ang(Angle(pit, yaw), pit, yaw, 0)
        assert_ang(Angle(pit), pit, 0, 0)
        assert_ang(Angle(), 0, 0, 0)

        assert_ang(Angle([pit, yaw, rol]), pit, yaw, rol)
        assert_ang(Angle([pit, yaw], roll=rol), pit, yaw, rol)
        assert_ang(Angle([pit], yaw=yaw, roll=rol), pit, yaw, rol)
        assert_ang(Angle([pit]), pit, 0, 0)
        assert_ang(Angle([pit, yaw]), pit, yaw, 0)
        assert_ang(Angle([pit, yaw, rol]), pit, yaw, rol)

        # Test this does nothing (except copy).
        ang = Angle(pit, yaw, rol)
        ang2 = Angle(ang)
        assert_ang(ang2, pit, yaw, rol)
        assert ang is not ang2

        ang3 = Angle.copy(ang)
        assert_ang(ang3, pit, yaw, rol)
        assert ang is not ang3

        # Test Angle.from_str()
        assert_ang(Angle.from_str('{} {} {}'.format(pit, yaw, rol)), pit, yaw, rol)
        assert_ang(Angle.from_str('<{} {} {}>'.format(pit, yaw, rol)), pit, yaw, rol)
        # {x y z}
        assert_ang(Angle.from_str('{{{} {} {}}}'.format(pit, yaw, rol)), pit, yaw, rol)
        assert_ang(Angle.from_str('({} {} {})'.format(pit, yaw, rol)), pit, yaw, rol)
        assert_ang(Angle.from_str('[{} {} {}]'.format(pit, yaw, rol)), pit, yaw, rol)

        # Test converting a converted Angle
        orig = Angle(pit, yaw, rol)
        new = Angle.from_str(Angle(pit, yaw, rol))
        assert_ang(new, pit, yaw, rol)
        assert orig is not new  # It must be a copy

        # Check as_tuple() makes an equivalent tuple
        tup = orig.as_tuple()

        # Flip to work arond the coercion.
        pit %= 360.0
        yaw %= 360.0
        rol %= 360.0

        assert isinstance(tup, tuple)
        assert (pit, yaw, rol) == tup
        assert hash((pit, yaw, rol)) == hash(tup)
        # Bypass subclass functions.
        assert tuple.__getitem__(tup, 0) == pit
        assert tuple.__getitem__(tup, 1) == yaw
        assert tuple.__getitem__(tup, 2) == rol

    # Check failures in Angle.from_str()
    # Note - does not pass through unchanged, they're converted to floats!
    for val in VALID_ZERONUMS:
        test_val = val % 360.0
        assert test_val == Angle.from_str('', pitch=val).pitch
        assert test_val == Angle.from_str('blah 4 2', yaw=val).yaw
        assert test_val == Angle.from_str('2 hi 2', pitch=val).pitch
        assert test_val == Angle.from_str('2 6 gh', roll=val).roll
        assert test_val == Angle.from_str('1.2 3.4', pitch=val).pitch
        assert test_val == Angle.from_str('34.5 38.4 -23 -38', roll=val).roll


def test_with_axes(py_c_vec):
    """Test the with_axes() constructor."""
    Vec, Angle, Matrix, parse_vec_str = py_c_vec

    for axis, u, v in [
        ('pitch', 'yaw', 'roll'),
        ('yaw', 'pitch', 'roll'),
        ('roll', 'pitch', 'yaw'),
    ]:
        for num in VALID_ZERONUMS:
            test_num = num % 360

            ang = Angle.with_axes(axis, num)
            assert ang[axis] == test_num
            assert getattr(ang, axis) == test_num
            # Other axes are zero.
            assert ang[u] == 0
            assert ang[v] == 0
            assert getattr(ang, u) == 0
            assert getattr(ang, v) == 0

    for a, b, c in iter_vec(('pitch', 'yaw', 'roll')):
        if a == b or b == c or a == c:
            continue
        for x, y, z in iter_vec(VALID_ZERONUMS):
            ang = Angle.with_axes(a, x, b, y, c, z)

            x %= 360
            y %= 360
            z %= 360

            assert ang[a] == x
            assert ang[b] == y
            assert ang[c] == z

            assert getattr(ang, a) == x
            assert getattr(ang, b) == y
            assert getattr(ang, c) == z


@pytest.mark.parametrize('axis, index, u, v, u_ax, v_ax', [
    ('pitch', 0, 'yaw', 'roll', 1, 2), ('yaw', 1, 'pitch', 'roll', 0, 2), ('roll', 2, 'pitch', 'yaw', 0, 1),
], ids=['pitch', 'yaw', 'roll'])
def test_attrs(py_c_vec, axis: str, index: int, u: str, v: str, u_ax: int, v_ax: int) -> None:
    """Test the pitch/yaw/roll attributes and item access."""
    Vec, Angle, Matrix, parse_vec_str = py_c_vec
    ang = Angle()

    def check(targ: float, other: float):
        """Check all the indexes are correct."""
        assert math.isclose(getattr(ang, axis), targ), f'ang.{axis} != {targ}, other: {other}'
        assert math.isclose(getattr(ang, u), other), f'ang.{u} != {other}, targ={targ}'
        assert math.isclose(getattr(ang, v), other), f'ang.{v} != {other}, targ={targ}'

        assert math.isclose(ang[index], targ),  f'ang[{index}] != {targ}, other: {other}'
        assert math.isclose(ang[axis], targ), f'ang[{axis!r}] != {targ}, other: {other}'
        assert math.isclose(ang[u_ax], other),  f'ang[{u_ax!r}] != {other}, targ={targ}'
        assert math.isclose(ang[v_ax], other), f'ang[{v_ax!r}] != {other}, targ={targ}'
        assert math.isclose(ang[u], other),  f'ang[{u!r}] != {other}, targ={targ}'
        assert math.isclose(ang[v], other), f'[{v!r}] != {other}, targ={targ}'

    nums = [
        (0, 0.0),
        (38.29, 38.29),
        (-89.0, 271.0),
        (360.0, 0.0),
        (361.49, 1.49),
        (-725.87, 354.13),
    ]

    for oth_set, oth_read in nums:
        ang.pitch = ang.yaw = ang.roll = oth_set
        check(oth_read, oth_read)
        for x_set, x_read in nums:
            setattr(ang, axis, x_set)
            check(x_read, oth_read)
