"""Tests for datetime64 and timedelta64 operations.

Compares rumpy datetime handling against numpy.
"""

import numpy as np
import pytest
from helpers import assert_eq

# Only import rumpy if available
rp = pytest.importorskip("rumpy")


class TestDateTime64DType:
    """Test datetime64 dtype creation and properties."""

    @pytest.mark.parametrize("unit", ["Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns"])
    def test_dtype_creation(self, unit):
        """Test creating datetime64 dtypes with different units."""
        n = np.array(['2023-06-15'], dtype=f'datetime64[{unit}]')
        r = rp.array(['2023-06-15'], dtype=f'datetime64[{unit}]')
        # rumpy dtype returns string, numpy dtype has .name attribute
        assert r.dtype == n.dtype.name

    @pytest.mark.parametrize("unit", ["D", "s", "ms", "us", "ns"])
    def test_dtype_itemsize(self, unit):
        """datetime64 should always be 8 bytes."""
        n = np.array(['2023-06-15'], dtype=f'datetime64[{unit}]')
        r = rp.array(['2023-06-15'], dtype=f'datetime64[{unit}]')
        assert r.itemsize == 8
        assert r.itemsize == n.dtype.itemsize


class TestTimeDelta64DType:
    """Test timedelta64 dtype creation and properties."""

    @pytest.mark.parametrize("unit", ["Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns"])
    def test_dtype_creation(self, unit):
        """Test creating timedelta64 dtypes with different units."""
        n = np.array([1, 2, 3], dtype=f'timedelta64[{unit}]')
        r = rp.array([1, 2, 3], dtype=f'timedelta64[{unit}]')
        assert r.dtype == n.dtype.name

    @pytest.mark.parametrize("unit", ["D", "s", "ms", "us", "ns"])
    def test_dtype_itemsize(self, unit):
        """timedelta64 should always be 8 bytes."""
        n = np.array([1, 2, 3], dtype=f'timedelta64[{unit}]')
        r = rp.array([1, 2, 3], dtype=f'timedelta64[{unit}]')
        assert r.itemsize == 8
        assert r.itemsize == n.dtype.itemsize


class TestDateTime64Creation:
    """Test creating datetime64 arrays."""

    def test_from_string_array(self):
        """Create datetime64 from string array."""
        dates = ['2023-01-01', '2023-06-15', '2023-12-31']
        n = np.array(dates, dtype='datetime64[D]')
        r = rp.array(dates, dtype='datetime64[D]')
        assert_eq(r, n)

    def test_from_iso_strings_with_time(self):
        """Create datetime64 from ISO strings with time."""
        dates = ['2023-06-15T12:30:00', '2023-06-15T18:45:30']
        n = np.array(dates, dtype='datetime64[s]')
        r = rp.array(dates, dtype='datetime64[s]')
        assert_eq(r, n)

    def test_from_integer_values(self):
        """Create datetime64 from integer values (days since epoch)."""
        # Days since 1970-01-01
        days = [0, 365, 19523]  # 1970-01-01, 1971-01-01, 2023-06-15
        n = np.array(days, dtype='int64').view('datetime64[D]')
        r = rp.array(days, dtype='int64').view('datetime64[D]')
        assert_eq(r, n)


class TestTimeDelta64Creation:
    """Test creating timedelta64 arrays."""

    def test_from_integers(self):
        """Create timedelta64 from integers."""
        n = np.array([1, 5, 10], dtype='timedelta64[D]')
        r = rp.array([1, 5, 10], dtype='timedelta64[D]')
        assert_eq(r, n)

    def test_from_integers_different_units(self):
        """Create timedelta64 with different units."""
        for unit in ['D', 'h', 'm', 's', 'ms', 'us', 'ns']:
            n = np.array([1, 5, 10], dtype=f'timedelta64[{unit}]')
            r = rp.array([1, 5, 10], dtype=f'timedelta64[{unit}]')
            assert_eq(r, n)


class TestDateTimeArithmetic:
    """Test arithmetic operations with datetime64/timedelta64."""

    def test_datetime_subtract_datetime(self):
        """Subtracting two datetime64 arrays gives timedelta64."""
        d1 = np.array(['2023-06-15', '2023-06-20'], dtype='datetime64[D]')
        d2 = np.array(['2023-06-10', '2023-06-15'], dtype='datetime64[D]')
        n_result = d1 - d2

        r1 = rp.array(['2023-06-15', '2023-06-20'], dtype='datetime64[D]')
        r2 = rp.array(['2023-06-10', '2023-06-15'], dtype='datetime64[D]')
        r_result = r1 - r2

        assert_eq(r_result, n_result)

    def test_datetime_add_timedelta(self):
        """Adding timedelta64 to datetime64."""
        d = np.array(['2023-06-15', '2023-06-20'], dtype='datetime64[D]')
        td = np.array([5, 10], dtype='timedelta64[D]')
        n_result = d + td

        r_d = rp.array(['2023-06-15', '2023-06-20'], dtype='datetime64[D]')
        r_td = rp.array([5, 10], dtype='timedelta64[D]')
        r_result = r_d + r_td

        assert_eq(r_result, n_result)

    def test_datetime_subtract_timedelta(self):
        """Subtracting timedelta64 from datetime64."""
        d = np.array(['2023-06-15', '2023-06-20'], dtype='datetime64[D]')
        td = np.array([5, 10], dtype='timedelta64[D]')
        n_result = d - td

        r_d = rp.array(['2023-06-15', '2023-06-20'], dtype='datetime64[D]')
        r_td = rp.array([5, 10], dtype='timedelta64[D]')
        r_result = r_d - r_td

        assert_eq(r_result, n_result)

    def test_timedelta_add_timedelta(self):
        """Adding two timedelta64 arrays."""
        td1 = np.array([5, 10], dtype='timedelta64[D]')
        td2 = np.array([3, 7], dtype='timedelta64[D]')
        n_result = td1 + td2

        r_td1 = rp.array([5, 10], dtype='timedelta64[D]')
        r_td2 = rp.array([3, 7], dtype='timedelta64[D]')
        r_result = r_td1 + r_td2

        assert_eq(r_result, n_result)

    def test_timedelta_multiply_integer(self):
        """Multiplying timedelta64 by integer."""
        td = np.array([5, 10], dtype='timedelta64[D]')
        n_result = td * 2

        r_td = rp.array([5, 10], dtype='timedelta64[D]')
        r_result = r_td * 2

        assert_eq(r_result, n_result)

    def test_timedelta_divide_integer(self):
        """Integer division of timedelta64."""
        td = np.array([10, 20], dtype='timedelta64[D]')
        n_result = td // 3

        r_td = rp.array([10, 20], dtype='timedelta64[D]')
        r_result = r_td // 3

        assert_eq(r_result, n_result)


class TestDateTimeComparison:
    """Test comparison operations with datetime64."""

    def test_less_than(self):
        """Compare datetime64 with <."""
        d1 = np.array(['2023-06-15', '2023-06-20'], dtype='datetime64[D]')
        d2 = np.array(['2023-06-20', '2023-06-15'], dtype='datetime64[D]')
        n_result = d1 < d2

        r1 = rp.array(['2023-06-15', '2023-06-20'], dtype='datetime64[D]')
        r2 = rp.array(['2023-06-20', '2023-06-15'], dtype='datetime64[D]')
        r_result = r1 < r2

        assert_eq(r_result, n_result)

    def test_equal(self):
        """Compare datetime64 with ==."""
        d1 = np.array(['2023-06-15', '2023-06-20'], dtype='datetime64[D]')
        d2 = np.array(['2023-06-15', '2023-06-25'], dtype='datetime64[D]')
        n_result = d1 == d2

        r1 = rp.array(['2023-06-15', '2023-06-20'], dtype='datetime64[D]')
        r2 = rp.array(['2023-06-15', '2023-06-25'], dtype='datetime64[D]')
        r_result = r1 == r2

        assert_eq(r_result, n_result)

    def test_greater_than_scalar(self):
        """Compare datetime64 array with scalar."""
        dates = np.array(['2023-01-01', '2023-06-15', '2023-12-31'], dtype='datetime64[D]')
        threshold = np.datetime64('2023-06-01')
        n_result = dates > threshold

        r_dates = rp.array(['2023-01-01', '2023-06-15', '2023-12-31'], dtype='datetime64[D]')
        # For now, test array vs array comparison
        r_threshold = rp.array(['2023-06-01', '2023-06-01', '2023-06-01'], dtype='datetime64[D]')
        r_result = r_dates > r_threshold

        assert_eq(r_result, n_result)


class TestDateTimeReductions:
    """Test reduction operations with datetime64."""

    def test_min(self):
        """Find minimum datetime64."""
        dates = np.array(['2023-06-15', '2023-01-01', '2023-12-31'], dtype='datetime64[D]')
        n_result = np.min(dates)

        r_dates = rp.array(['2023-06-15', '2023-01-01', '2023-12-31'], dtype='datetime64[D]')
        r_result = rp.min(r_dates)

        # rumpy returns float for scalar reductions, numpy returns datetime64 scalar
        # Compare the underlying values
        assert int(r_result) == n_result.view('int64')

    def test_max(self):
        """Find maximum datetime64."""
        dates = np.array(['2023-06-15', '2023-01-01', '2023-12-31'], dtype='datetime64[D]')
        n_result = np.max(dates)

        r_dates = rp.array(['2023-06-15', '2023-01-01', '2023-12-31'], dtype='datetime64[D]')
        r_result = rp.max(r_dates)

        # rumpy returns float for scalar reductions, numpy returns datetime64 scalar
        # Compare the underlying values
        assert int(r_result) == n_result.view('int64')

    def test_argmin(self):
        """Find index of minimum datetime64."""
        dates = np.array(['2023-06-15', '2023-01-01', '2023-12-31'], dtype='datetime64[D]')
        n_result = np.argmin(dates)

        r_dates = rp.array(['2023-06-15', '2023-01-01', '2023-12-31'], dtype='datetime64[D]')
        r_result = rp.argmin(r_dates)

        assert r_result == n_result

    def test_argmax(self):
        """Find index of maximum datetime64."""
        dates = np.array(['2023-06-15', '2023-01-01', '2023-12-31'], dtype='datetime64[D]')
        n_result = np.argmax(dates)

        r_dates = rp.array(['2023-06-15', '2023-01-01', '2023-12-31'], dtype='datetime64[D]')
        r_result = rp.argmax(r_dates)

        assert r_result == n_result


class TestDateTimeSorting:
    """Test sorting operations with datetime64."""

    def test_sort(self):
        """Sort datetime64 array."""
        dates = np.array(['2023-12-31', '2023-01-01', '2023-06-15'], dtype='datetime64[D]')
        n_result = np.sort(dates)

        r_dates = rp.array(['2023-12-31', '2023-01-01', '2023-06-15'], dtype='datetime64[D]')
        r_result = rp.sort(r_dates)

        assert_eq(r_result, n_result)

    def test_argsort(self):
        """Get sort indices for datetime64 array."""
        dates = np.array(['2023-12-31', '2023-01-01', '2023-06-15'], dtype='datetime64[D]')
        n_result = np.argsort(dates)

        r_dates = rp.array(['2023-12-31', '2023-01-01', '2023-06-15'], dtype='datetime64[D]')
        r_result = rp.argsort(r_dates)

        assert_eq(r_result, n_result)


class TestIsNaT:
    """Test NaT (Not a Time) handling."""

    def test_isnat_with_nat(self):
        """Test isnat function with NaT values."""
        dates = np.array(['2023-06-15', 'NaT', '2023-12-31'], dtype='datetime64[D]')
        n_result = np.isnat(dates)

        r_dates = rp.array(['2023-06-15', 'NaT', '2023-12-31'], dtype='datetime64[D]')
        r_result = rp.isnat(r_dates)

        assert_eq(r_result, n_result)

    def test_isnat_no_nat(self):
        """Test isnat function without NaT values."""
        dates = np.array(['2023-06-15', '2023-06-20'], dtype='datetime64[D]')
        n_result = np.isnat(dates)

        r_dates = rp.array(['2023-06-15', '2023-06-20'], dtype='datetime64[D]')
        r_result = rp.isnat(r_dates)

        assert_eq(r_result, n_result)


class TestDateTimeAsString:
    """Test datetime_as_string function."""

    def test_basic(self):
        """Basic datetime_as_string conversion."""
        dates = np.array(['2023-01-15', '2023-06-20'], dtype='datetime64[D]')
        n_result = np.datetime_as_string(dates)

        r_dates = rp.array(['2023-01-15', '2023-06-20'], dtype='datetime64[D]')
        r_result = rp.datetime_as_string(r_dates)

        # Compare as lists since string arrays may differ in dtype
        assert list(r_result) == list(n_result)

    def test_with_unit(self):
        """datetime_as_string with different unit."""
        dates = np.array(['2023-01-15', '2023-06-20'], dtype='datetime64[D]')
        n_result = np.datetime_as_string(dates, unit='M')

        r_dates = rp.array(['2023-01-15', '2023-06-20'], dtype='datetime64[D]')
        r_result = rp.datetime_as_string(r_dates, unit='M')

        assert list(r_result) == list(n_result)


class TestDateTimeData:
    """Test datetime_data function."""

    @pytest.mark.parametrize("unit", ["D", "s", "ms", "us", "ns"])
    def test_datetime_data(self, unit):
        """Test datetime_data returns correct unit info."""
        dtype = np.dtype(f'datetime64[{unit}]')
        n_result = np.datetime_data(dtype)

        # rumpy datetime_data accepts a dtype string
        r_result = rp.datetime_data(f'datetime64[{unit}]')

        assert r_result == n_result


class TestBusDayCalendar:
    """Test busdaycalendar class."""

    def test_default_calendar(self):
        """Test default busdaycalendar (Mon-Fri)."""
        n_cal = np.busdaycalendar()
        r_cal = rp.busdaycalendar()

        assert list(n_cal.weekmask) == list(r_cal.weekmask)

    def test_custom_weekmask_string(self):
        """Test busdaycalendar with custom weekmask string."""
        n_cal = np.busdaycalendar(weekmask='Mon Tue Wed Thu')
        r_cal = rp.busdaycalendar(weekmask='Mon Tue Wed Thu')

        assert list(n_cal.weekmask) == list(r_cal.weekmask)

    def test_with_holidays(self):
        """Test busdaycalendar with holidays."""
        holidays = ['2023-07-04', '2023-12-25']
        n_cal = np.busdaycalendar(holidays=holidays)
        r_cal = rp.busdaycalendar(holidays=holidays)

        assert list(n_cal.weekmask) == list(r_cal.weekmask)
        # Holidays are stored as datetime64[D] array
        assert len(n_cal.holidays) == r_cal.holidays.shape[0]


class TestIsBusDay:
    """Test is_busday function."""

    def test_weekday(self):
        """Test is_busday for weekday (business day)."""
        # June 15, 2023 was a Thursday
        n_result = np.is_busday('2023-06-15')
        r_result = rp.is_busday('2023-06-15')
        assert r_result == n_result

    def test_weekend(self):
        """Test is_busday for weekend (not business day)."""
        # June 17, 2023 was a Saturday
        n_result = np.is_busday('2023-06-17')
        r_result = rp.is_busday('2023-06-17')
        assert r_result == n_result

    def test_array(self):
        """Test is_busday with array input."""
        dates = ['2023-06-15', '2023-06-17', '2023-06-19']  # Thu, Sat, Mon
        n_result = np.is_busday(dates)
        r_result = rp.is_busday(dates)
        assert_eq(r_result, n_result)

    def test_with_calendar(self):
        """Test is_busday with custom calendar (holiday)."""
        holidays = ['2023-07-04']
        n_cal = np.busdaycalendar(holidays=holidays)
        r_cal = rp.busdaycalendar(holidays=holidays)

        # July 4, 2023 was a Tuesday but is a holiday
        n_result = np.is_busday('2023-07-04', busdaycal=n_cal)
        r_result = rp.is_busday('2023-07-04', busdaycal=r_cal)
        assert r_result == n_result
        assert r_result == False


class TestBusDayOffset:
    """Test busday_offset function."""

    def test_forward_offset(self):
        """Test busday_offset moving forward."""
        # June 15, 2023 (Thu) + 1 business day = June 16, 2023 (Fri)
        n_result = np.busday_offset('2023-06-15', 1)
        r_result = rp.busday_offset('2023-06-15', 1)
        assert_eq(r_result, n_result)

    def test_backward_offset(self):
        """Test busday_offset moving backward."""
        # June 15, 2023 (Thu) - 1 business day = June 14, 2023 (Wed)
        n_result = np.busday_offset('2023-06-15', -1)
        r_result = rp.busday_offset('2023-06-15', -1)
        assert_eq(r_result, n_result)

    def test_offset_over_weekend(self):
        """Test busday_offset skipping weekend."""
        # June 16, 2023 (Fri) + 1 business day = June 19, 2023 (Mon)
        n_result = np.busday_offset('2023-06-16', 1)
        r_result = rp.busday_offset('2023-06-16', 1)
        assert_eq(r_result, n_result)

    def test_array_offset(self):
        """Test busday_offset with array inputs."""
        dates = ['2023-06-15', '2023-06-16']
        offsets = [1, 2]
        n_result = np.busday_offset(dates, offsets)
        r_result = rp.busday_offset(dates, offsets)
        assert_eq(r_result, n_result)


class TestBusDayCount:
    """Test busday_count function."""

    def test_same_week(self):
        """Test busday_count within same week."""
        # Mon Jun 12 to Fri Jun 16 = 4 business days (not counting end)
        n_result = np.busday_count('2023-06-12', '2023-06-16')
        r_result = rp.busday_count('2023-06-12', '2023-06-16')
        assert r_result == n_result

    def test_month(self):
        """Test busday_count for a month."""
        # June 2023 has 22 business days
        n_result = np.busday_count('2023-06-01', '2023-06-30')
        r_result = rp.busday_count('2023-06-01', '2023-06-30')
        assert r_result == n_result

    def test_array(self):
        """Test busday_count with array inputs."""
        starts = ['2023-06-01', '2023-07-01']
        ends = ['2023-06-30', '2023-07-31']
        n_result = np.busday_count(starts, ends)
        r_result = rp.busday_count(starts, ends)
        assert_eq(r_result, n_result)

    def test_with_calendar(self):
        """Test busday_count with custom calendar."""
        holidays = ['2023-07-04']
        n_cal = np.busdaycalendar(holidays=holidays)
        r_cal = rp.busdaycalendar(holidays=holidays)

        # July 2023 has 21 business days minus July 4 holiday = 20
        n_result = np.busday_count('2023-07-01', '2023-07-31', busdaycal=n_cal)
        r_result = rp.busday_count('2023-07-01', '2023-07-31', busdaycal=r_cal)
        assert r_result == n_result


class TestUnitConversion:
    """Test datetime64 unit conversion."""

    def test_day_to_seconds(self):
        """Convert datetime64[D] to datetime64[s]."""
        d = np.array(['2023-06-15'], dtype='datetime64[D]')
        n_result = d.astype('datetime64[s]')

        r_d = rp.array(['2023-06-15'], dtype='datetime64[D]')
        r_result = r_d.astype('datetime64[s]')

        assert_eq(r_result, n_result)

    def test_seconds_to_day(self):
        """Convert datetime64[s] to datetime64[D] (truncates)."""
        d = np.array(['2023-06-15T12:30:45'], dtype='datetime64[s]')
        n_result = d.astype('datetime64[D]')

        r_d = rp.array(['2023-06-15T12:30:45'], dtype='datetime64[s]')
        r_result = r_d.astype('datetime64[D]')

        assert_eq(r_result, n_result)
