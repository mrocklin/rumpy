//! DateTime64 and TimeDelta64 dtype implementations - parametric dtypes with time units.

use super::{BinaryOp, DTypeKind, DTypeOps, ReduceOp, UnaryOp};
use std::cmp::Ordering;

/// Time units for datetime64 and timedelta64.
/// All units store values as i64 relative to the Unix epoch (1970-01-01).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TimeUnit {
    Years,        // Y - years since epoch
    Months,       // M - months since epoch
    Weeks,        // W - weeks since epoch
    Days,         // D - days since epoch
    Hours,        // h - hours since epoch
    Minutes,      // m - minutes since epoch
    Seconds,      // s - seconds since epoch
    Milliseconds, // ms - milliseconds since epoch
    Microseconds, // us - microseconds since epoch
    Nanoseconds,  // ns - nanoseconds since epoch
}

impl TimeUnit {
    /// Parse unit from string (e.g., "D", "ns")
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "Y" => Some(TimeUnit::Years),
            "M" => Some(TimeUnit::Months),
            "W" => Some(TimeUnit::Weeks),
            "D" => Some(TimeUnit::Days),
            "h" => Some(TimeUnit::Hours),
            "m" => Some(TimeUnit::Minutes),
            "s" => Some(TimeUnit::Seconds),
            "ms" => Some(TimeUnit::Milliseconds),
            "us" => Some(TimeUnit::Microseconds),
            "ns" => Some(TimeUnit::Nanoseconds),
            _ => None,
        }
    }

    /// Get the string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            TimeUnit::Years => "Y",
            TimeUnit::Months => "M",
            TimeUnit::Weeks => "W",
            TimeUnit::Days => "D",
            TimeUnit::Hours => "h",
            TimeUnit::Minutes => "m",
            TimeUnit::Seconds => "s",
            TimeUnit::Milliseconds => "ms",
            TimeUnit::Microseconds => "us",
            TimeUnit::Nanoseconds => "ns",
        }
    }

    /// Get the conversion factor to convert FROM this unit TO nanoseconds.
    /// Returns None for variable-length units (Years, Months).
    pub fn to_ns_factor(&self) -> Option<i64> {
        match self {
            TimeUnit::Years | TimeUnit::Months => None, // Variable length
            TimeUnit::Weeks => Some(7 * 24 * 60 * 60 * 1_000_000_000),
            TimeUnit::Days => Some(24 * 60 * 60 * 1_000_000_000),
            TimeUnit::Hours => Some(60 * 60 * 1_000_000_000),
            TimeUnit::Minutes => Some(60 * 1_000_000_000),
            TimeUnit::Seconds => Some(1_000_000_000),
            TimeUnit::Milliseconds => Some(1_000_000),
            TimeUnit::Microseconds => Some(1_000),
            TimeUnit::Nanoseconds => Some(1),
        }
    }

    /// Get the conversion factor to convert FROM this unit TO seconds.
    pub fn to_seconds_factor(&self) -> Option<i64> {
        match self {
            TimeUnit::Years | TimeUnit::Months => None, // Variable length
            TimeUnit::Weeks => Some(7 * 24 * 60 * 60),
            TimeUnit::Days => Some(24 * 60 * 60),
            TimeUnit::Hours => Some(60 * 60),
            TimeUnit::Minutes => Some(60),
            TimeUnit::Seconds => Some(1),
            TimeUnit::Milliseconds | TimeUnit::Microseconds | TimeUnit::Nanoseconds => Some(0), // Fractional
        }
    }

    /// Get the conversion factor to convert FROM this unit TO days.
    pub fn to_days_factor(&self) -> Option<i64> {
        match self {
            TimeUnit::Years | TimeUnit::Months => None, // Variable length
            TimeUnit::Weeks => Some(7),
            TimeUnit::Days => Some(1),
            _ => Some(0), // Sub-day units
        }
    }
}

/// NaT (Not a Time) value - same as NumPy's representation
pub const NAT: i64 = i64::MIN;

/// DateTime64 dtype operations.
pub(super) struct DateTime64Ops {
    pub(super) unit: TimeUnit,
}

impl DateTime64Ops {
    #[inline]
    unsafe fn read(ptr: *const u8, byte_offset: isize) -> i64 {
        *(ptr.offset(byte_offset) as *const i64)
    }

    #[inline]
    unsafe fn write(ptr: *mut u8, idx: usize, val: i64) {
        *(ptr as *mut i64).add(idx) = val;
    }
}

impl DTypeOps for DateTime64Ops {
    fn kind(&self) -> DTypeKind { DTypeKind::DateTime64(self.unit) }
    fn itemsize(&self) -> usize { 8 }

    fn typestr(&self) -> &'static str {
        match self.unit {
            TimeUnit::Years => "<M8[Y]",
            TimeUnit::Months => "<M8[M]",
            TimeUnit::Weeks => "<M8[W]",
            TimeUnit::Days => "<M8[D]",
            TimeUnit::Hours => "<M8[h]",
            TimeUnit::Minutes => "<M8[m]",
            TimeUnit::Seconds => "<M8[s]",
            TimeUnit::Milliseconds => "<M8[ms]",
            TimeUnit::Microseconds => "<M8[us]",
            TimeUnit::Nanoseconds => "<M8[ns]",
        }
    }

    fn format_char(&self) -> &'static str { "q" } // int64 underlying

    fn name(&self) -> &'static str {
        match self.unit {
            TimeUnit::Years => "datetime64[Y]",
            TimeUnit::Months => "datetime64[M]",
            TimeUnit::Weeks => "datetime64[W]",
            TimeUnit::Days => "datetime64[D]",
            TimeUnit::Hours => "datetime64[h]",
            TimeUnit::Minutes => "datetime64[m]",
            TimeUnit::Seconds => "datetime64[s]",
            TimeUnit::Milliseconds => "datetime64[ms]",
            TimeUnit::Microseconds => "datetime64[us]",
            TimeUnit::Nanoseconds => "datetime64[ns]",
        }
    }

    fn promotion_priority(&self) -> u8 { 50 } // Doesn't promote with numerics

    unsafe fn write_zero(&self, ptr: *mut u8, idx: usize) {
        Self::write(ptr, idx, 0);
    }

    unsafe fn write_one(&self, ptr: *mut u8, idx: usize) {
        Self::write(ptr, idx, 1);
    }

    unsafe fn copy_element(&self, src: *const u8, byte_offset: isize, dst: *mut u8, idx: usize) {
        Self::write(dst, idx, Self::read(src, byte_offset));
    }

    unsafe fn unary_op(&self, op: UnaryOp, src: *const u8, byte_offset: isize, out: *mut u8, idx: usize) {
        let v = Self::read(src, byte_offset);
        // Most unary ops don't make sense for datetime, just copy
        let result = match op {
            UnaryOp::Neg => -v,
            UnaryOp::Abs => v.abs(),
            _ => v,
        };
        Self::write(out, idx, result);
    }

    unsafe fn binary_op(&self, op: BinaryOp, a: *const u8, a_offset: isize, b: *const u8, b_offset: isize, out: *mut u8, idx: usize) {
        let av = Self::read(a, a_offset);
        let bv = Self::read(b, b_offset);
        // Handle NaT propagation
        if av == NAT || bv == NAT {
            Self::write(out, idx, NAT);
            return;
        }
        // Only add/sub make sense for datetime (as timedelta ops)
        let result = match op {
            BinaryOp::Add => av.wrapping_add(bv),
            BinaryOp::Sub => av.wrapping_sub(bv),
            BinaryOp::Mul => av.wrapping_mul(bv),
            BinaryOp::Div => if bv != 0 { av / bv } else { 0 },
            BinaryOp::Pow => if bv >= 0 { av.wrapping_pow(bv as u32) } else { 0 },
            BinaryOp::Mod => if bv != 0 { av % bv } else { 0 },
            BinaryOp::FloorDiv => if bv != 0 { av.div_euclid(bv) } else { 0 },
            BinaryOp::Maximum | BinaryOp::FMax => av.max(bv),
            BinaryOp::Minimum | BinaryOp::FMin => av.min(bv),
            // Float ops don't make sense for datetime; just return av
            BinaryOp::Arctan2 | BinaryOp::Hypot | BinaryOp::Copysign
            | BinaryOp::Logaddexp | BinaryOp::Logaddexp2 => av,
            BinaryOp::Nextafter => if av < bv { av.wrapping_add(1) } else if av > bv { av.wrapping_sub(1) } else { bv },
        };
        Self::write(out, idx, result);
    }

    unsafe fn reduce_init(&self, op: ReduceOp, out: *mut u8, idx: usize) {
        let v = match op {
            ReduceOp::Sum => 0,
            ReduceOp::Prod => 1,
            ReduceOp::Max => i64::MIN,
            ReduceOp::Min => i64::MAX,
        };
        Self::write(out, idx, v);
    }

    unsafe fn reduce_acc(&self, op: ReduceOp, acc: *mut u8, idx: usize, val: *const u8, byte_offset: isize) {
        let a = Self::read(acc as *const u8, (idx * 8) as isize);
        let v = Self::read(val, byte_offset);
        let result = match op {
            ReduceOp::Sum => a.wrapping_add(v),
            ReduceOp::Prod => a.wrapping_mul(v),
            ReduceOp::Max => a.max(v),
            ReduceOp::Min => a.min(v),
        };
        Self::write(acc, idx, result);
    }

    unsafe fn format_element(&self, ptr: *const u8, byte_offset: isize) -> String {
        let v = Self::read(ptr, byte_offset);
        if v == NAT {
            return "NaT".to_string();
        }
        format_datetime64(v, self.unit)
    }

    unsafe fn compare_elements(&self, a: *const u8, a_offset: isize, b: *const u8, b_offset: isize) -> Ordering {
        Self::read(a, a_offset).cmp(&Self::read(b, b_offset))
    }

    unsafe fn is_truthy(&self, ptr: *const u8, byte_offset: isize) -> bool {
        let v = Self::read(ptr, byte_offset);
        v != 0 && v != NAT
    }

    unsafe fn write_f64(&self, ptr: *mut u8, idx: usize, val: f64) -> bool {
        Self::write(ptr, idx, val as i64);
        true
    }

    unsafe fn read_f64(&self, ptr: *const u8, byte_offset: isize) -> Option<f64> {
        Some(Self::read(ptr, byte_offset) as f64)
    }

    unsafe fn read_i64(&self, ptr: *const u8, byte_offset: isize) -> Option<i64> {
        Some(Self::read(ptr, byte_offset))
    }

    unsafe fn write_i64(&self, ptr: *mut u8, idx: usize, val: i64) -> bool {
        Self::write(ptr, idx, val);
        true
    }

    unsafe fn write_f64_at_byte_offset(&self, ptr: *mut u8, byte_offset: isize, val: f64) -> bool {
        *(ptr.offset(byte_offset) as *mut i64) = val as i64;
        true
    }

    unsafe fn write_complex(&self, ptr: *mut u8, idx: usize, real: f64, _imag: f64) -> bool {
        Self::write(ptr, idx, real as i64);
        true
    }

    unsafe fn read_complex(&self, ptr: *const u8, byte_offset: isize) -> Option<(f64, f64)> {
        Some((Self::read(ptr, byte_offset) as f64, 0.0))
    }
}

/// TimeDelta64 dtype operations.
pub(super) struct TimeDelta64Ops {
    pub(super) unit: TimeUnit,
}

impl TimeDelta64Ops {
    #[inline]
    unsafe fn read(ptr: *const u8, byte_offset: isize) -> i64 {
        *(ptr.offset(byte_offset) as *const i64)
    }

    #[inline]
    unsafe fn write(ptr: *mut u8, idx: usize, val: i64) {
        *(ptr as *mut i64).add(idx) = val;
    }
}

impl DTypeOps for TimeDelta64Ops {
    fn kind(&self) -> DTypeKind { DTypeKind::TimeDelta64(self.unit) }
    fn itemsize(&self) -> usize { 8 }

    fn typestr(&self) -> &'static str {
        match self.unit {
            TimeUnit::Years => "<m8[Y]",
            TimeUnit::Months => "<m8[M]",
            TimeUnit::Weeks => "<m8[W]",
            TimeUnit::Days => "<m8[D]",
            TimeUnit::Hours => "<m8[h]",
            TimeUnit::Minutes => "<m8[m]",
            TimeUnit::Seconds => "<m8[s]",
            TimeUnit::Milliseconds => "<m8[ms]",
            TimeUnit::Microseconds => "<m8[us]",
            TimeUnit::Nanoseconds => "<m8[ns]",
        }
    }

    fn format_char(&self) -> &'static str { "q" } // int64 underlying

    fn name(&self) -> &'static str {
        match self.unit {
            TimeUnit::Years => "timedelta64[Y]",
            TimeUnit::Months => "timedelta64[M]",
            TimeUnit::Weeks => "timedelta64[W]",
            TimeUnit::Days => "timedelta64[D]",
            TimeUnit::Hours => "timedelta64[h]",
            TimeUnit::Minutes => "timedelta64[m]",
            TimeUnit::Seconds => "timedelta64[s]",
            TimeUnit::Milliseconds => "timedelta64[ms]",
            TimeUnit::Microseconds => "timedelta64[us]",
            TimeUnit::Nanoseconds => "timedelta64[ns]",
        }
    }

    fn promotion_priority(&self) -> u8 { 50 } // Doesn't promote with numerics

    unsafe fn write_zero(&self, ptr: *mut u8, idx: usize) {
        Self::write(ptr, idx, 0);
    }

    unsafe fn write_one(&self, ptr: *mut u8, idx: usize) {
        Self::write(ptr, idx, 1);
    }

    unsafe fn copy_element(&self, src: *const u8, byte_offset: isize, dst: *mut u8, idx: usize) {
        Self::write(dst, idx, Self::read(src, byte_offset));
    }

    unsafe fn unary_op(&self, op: UnaryOp, src: *const u8, byte_offset: isize, out: *mut u8, idx: usize) {
        let v = Self::read(src, byte_offset);
        let result = match op {
            UnaryOp::Neg => -v,
            UnaryOp::Abs => v.abs(),
            _ => v,
        };
        Self::write(out, idx, result);
    }

    unsafe fn binary_op(&self, op: BinaryOp, a: *const u8, a_offset: isize, b: *const u8, b_offset: isize, out: *mut u8, idx: usize) {
        let av = Self::read(a, a_offset);
        let bv = Self::read(b, b_offset);
        // Handle NaT propagation
        if av == NAT || bv == NAT {
            Self::write(out, idx, NAT);
            return;
        }
        let result = match op {
            BinaryOp::Add => av.wrapping_add(bv),
            BinaryOp::Sub => av.wrapping_sub(bv),
            BinaryOp::Mul => av.wrapping_mul(bv),
            BinaryOp::Div => if bv != 0 { av / bv } else { 0 },
            BinaryOp::Pow => if bv >= 0 { av.wrapping_pow(bv as u32) } else { 0 },
            BinaryOp::Mod => if bv != 0 { av % bv } else { 0 },
            BinaryOp::FloorDiv => if bv != 0 { av.div_euclid(bv) } else { 0 },
            BinaryOp::Maximum | BinaryOp::FMax => av.max(bv),
            BinaryOp::Minimum | BinaryOp::FMin => av.min(bv),
            BinaryOp::Arctan2 | BinaryOp::Hypot | BinaryOp::Copysign
            | BinaryOp::Logaddexp | BinaryOp::Logaddexp2 => av,
            BinaryOp::Nextafter => if av < bv { av.wrapping_add(1) } else if av > bv { av.wrapping_sub(1) } else { bv },
        };
        Self::write(out, idx, result);
    }

    unsafe fn reduce_init(&self, op: ReduceOp, out: *mut u8, idx: usize) {
        let v = match op {
            ReduceOp::Sum => 0,
            ReduceOp::Prod => 1,
            ReduceOp::Max => i64::MIN,
            ReduceOp::Min => i64::MAX,
        };
        Self::write(out, idx, v);
    }

    unsafe fn reduce_acc(&self, op: ReduceOp, acc: *mut u8, idx: usize, val: *const u8, byte_offset: isize) {
        let a = Self::read(acc as *const u8, (idx * 8) as isize);
        let v = Self::read(val, byte_offset);
        let result = match op {
            ReduceOp::Sum => a.wrapping_add(v),
            ReduceOp::Prod => a.wrapping_mul(v),
            ReduceOp::Max => a.max(v),
            ReduceOp::Min => a.min(v),
        };
        Self::write(acc, idx, result);
    }

    unsafe fn format_element(&self, ptr: *const u8, byte_offset: isize) -> String {
        let v = Self::read(ptr, byte_offset);
        if v == NAT {
            return "NaT".to_string();
        }
        format_timedelta64(v, self.unit)
    }

    unsafe fn compare_elements(&self, a: *const u8, a_offset: isize, b: *const u8, b_offset: isize) -> Ordering {
        Self::read(a, a_offset).cmp(&Self::read(b, b_offset))
    }

    unsafe fn is_truthy(&self, ptr: *const u8, byte_offset: isize) -> bool {
        let v = Self::read(ptr, byte_offset);
        v != 0 && v != NAT
    }

    unsafe fn write_f64(&self, ptr: *mut u8, idx: usize, val: f64) -> bool {
        Self::write(ptr, idx, val as i64);
        true
    }

    unsafe fn read_f64(&self, ptr: *const u8, byte_offset: isize) -> Option<f64> {
        Some(Self::read(ptr, byte_offset) as f64)
    }

    unsafe fn read_i64(&self, ptr: *const u8, byte_offset: isize) -> Option<i64> {
        Some(Self::read(ptr, byte_offset))
    }

    unsafe fn write_i64(&self, ptr: *mut u8, idx: usize, val: i64) -> bool {
        Self::write(ptr, idx, val);
        true
    }

    unsafe fn write_f64_at_byte_offset(&self, ptr: *mut u8, byte_offset: isize, val: f64) -> bool {
        *(ptr.offset(byte_offset) as *mut i64) = val as i64;
        true
    }

    unsafe fn write_complex(&self, ptr: *mut u8, idx: usize, real: f64, _imag: f64) -> bool {
        Self::write(ptr, idx, real as i64);
        true
    }

    unsafe fn read_complex(&self, ptr: *const u8, byte_offset: isize) -> Option<(f64, f64)> {
        Some((Self::read(ptr, byte_offset) as f64, 0.0))
    }
}

// ============ Date/Time Formatting and Parsing ============

/// Format a datetime64 value to a string based on unit.
pub fn format_datetime64(value: i64, unit: TimeUnit) -> String {
    match unit {
        TimeUnit::Years => {
            // value is years since 1970
            let year = 1970 + value;
            format!("{:04}", year)
        }
        TimeUnit::Months => {
            // value is months since 1970-01
            let total_months = 1970 * 12 + value;
            let year = total_months / 12;
            let month = (total_months % 12) + 1;
            format!("{:04}-{:02}", year, month)
        }
        TimeUnit::Weeks => {
            // value is weeks since 1970-01-01 (Thursday)
            // NumPy week epoch starts at Thursday 1970-01-01
            days_to_date_string(value * 7)
        }
        TimeUnit::Days => {
            days_to_date_string(value)
        }
        TimeUnit::Hours => {
            let days = value.div_euclid(24);
            let hours = value.rem_euclid(24);
            format!("{}T{:02}", days_to_date_string(days), hours)
        }
        TimeUnit::Minutes => {
            let days = value.div_euclid(24 * 60);
            let rem = value.rem_euclid(24 * 60);
            let hours = rem / 60;
            let minutes = rem % 60;
            format!("{}T{:02}:{:02}", days_to_date_string(days), hours, minutes)
        }
        TimeUnit::Seconds => {
            let days = value.div_euclid(24 * 60 * 60);
            let rem = value.rem_euclid(24 * 60 * 60);
            let hours = rem / 3600;
            let minutes = (rem % 3600) / 60;
            let seconds = rem % 60;
            format!("{}T{:02}:{:02}:{:02}", days_to_date_string(days), hours, minutes, seconds)
        }
        TimeUnit::Milliseconds => {
            let days = value.div_euclid(24 * 60 * 60 * 1000);
            let rem = value.rem_euclid(24 * 60 * 60 * 1000);
            let hours = rem / 3_600_000;
            let minutes = (rem % 3_600_000) / 60_000;
            let seconds = (rem % 60_000) / 1000;
            let ms = rem % 1000;
            format!("{}T{:02}:{:02}:{:02}.{:03}", days_to_date_string(days), hours, minutes, seconds, ms)
        }
        TimeUnit::Microseconds => {
            let days = value.div_euclid(24 * 60 * 60 * 1_000_000);
            let rem = value.rem_euclid(24 * 60 * 60 * 1_000_000);
            let hours = rem / 3_600_000_000;
            let minutes = (rem % 3_600_000_000) / 60_000_000;
            let seconds = (rem % 60_000_000) / 1_000_000;
            let us = rem % 1_000_000;
            format!("{}T{:02}:{:02}:{:02}.{:06}", days_to_date_string(days), hours, minutes, seconds, us)
        }
        TimeUnit::Nanoseconds => {
            let days = value.div_euclid(24 * 60 * 60 * 1_000_000_000);
            let rem = value.rem_euclid(24 * 60 * 60 * 1_000_000_000);
            let hours = rem / 3_600_000_000_000;
            let minutes = (rem % 3_600_000_000_000) / 60_000_000_000;
            let seconds = (rem % 60_000_000_000) / 1_000_000_000;
            let ns = rem % 1_000_000_000;
            format!("{}T{:02}:{:02}:{:02}.{:09}", days_to_date_string(days), hours, minutes, seconds, ns)
        }
    }
}

/// Format a timedelta64 value to a string based on unit.
fn format_timedelta64(value: i64, unit: TimeUnit) -> String {
    let unit_str = match unit {
        TimeUnit::Years => "years",
        TimeUnit::Months => "months",
        TimeUnit::Weeks => "weeks",
        TimeUnit::Days => "days",
        TimeUnit::Hours => "hours",
        TimeUnit::Minutes => "minutes",
        TimeUnit::Seconds => "seconds",
        TimeUnit::Milliseconds => "milliseconds",
        TimeUnit::Microseconds => "microseconds",
        TimeUnit::Nanoseconds => "nanoseconds",
    };
    // NumPy uses singular form for 1 and -1
    if value == 1 || value == -1 {
        let singular = match unit {
            TimeUnit::Years => "year",
            TimeUnit::Months => "month",
            TimeUnit::Weeks => "week",
            TimeUnit::Days => "day",
            TimeUnit::Hours => "hour",
            TimeUnit::Minutes => "minute",
            TimeUnit::Seconds => "second",
            TimeUnit::Milliseconds => "millisecond",
            TimeUnit::Microseconds => "microsecond",
            TimeUnit::Nanoseconds => "nanosecond",
        };
        format!("{} {}", value, singular)
    } else {
        format!("{} {}", value, unit_str)
    }
}

/// Convert days since epoch (1970-01-01) to YYYY-MM-DD string.
fn days_to_date_string(days: i64) -> String {
    // Handle negative days (before 1970)
    let (year, month, day) = days_to_ymd(days);
    format!("{:04}-{:02}-{:02}", year, month, day)
}

/// Convert days since epoch to (year, month, day).
fn days_to_ymd(days: i64) -> (i64, i64, i64) {
    // Algorithm from Howard Hinnant's date algorithms
    // http://howardhinnant.github.io/date_algorithms.html#civil_from_days
    let z = days + 719468; // Shift epoch from 1970-01-01 to 0000-03-01
    let era = if z >= 0 { z / 146097 } else { (z - 146096) / 146097 };
    let doe = (z - era * 146097) as i64; // day of era [0, 146096]
    let yoe = (doe - doe/1460 + doe/36524 - doe/146096) / 365; // year of era [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365*yoe + yoe/4 - yoe/100); // day of year [0, 365]
    let mp = (5*doy + 2)/153; // month index [0, 11]
    let d = doy - (153*mp + 2)/5 + 1; // day [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // month [1, 12]
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

/// Convert (year, month, day) to days since epoch.
pub fn ymd_to_days(year: i64, month: i64, day: i64) -> i64 {
    // Algorithm from Howard Hinnant's date algorithms
    let y = if month <= 2 { year - 1 } else { year };
    let m = if month <= 2 { month + 12 } else { month };
    let era = if y >= 0 { y / 400 } else { (y - 399) / 400 };
    let yoe = (y - era * 400) as i64; // year of era [0, 399]
    let doy = (153 * (m - 3) + 2) / 5 + day - 1; // day of year [0, 365]
    let doe = yoe * 365 + yoe/4 - yoe/100 + doy; // day of era [0, 146096]
    era * 146097 + doe - 719468 // shift epoch from 0000-03-01 to 1970-01-01
}

/// Parse an ISO 8601 datetime string to a value in the given unit.
/// Returns NAT on parse error.
pub fn parse_datetime64(s: &str, unit: TimeUnit) -> i64 {
    if s == "NaT" || s == "nat" {
        return NAT;
    }

    // Parse YYYY-MM-DDTHH:MM:SS.sss format
    // Minimum is YYYY
    let parts: Vec<&str> = s.split('T').collect();
    let date_part = parts[0];
    let time_part = parts.get(1).copied();

    // Parse date: YYYY or YYYY-MM or YYYY-MM-DD
    let date_parts: Vec<&str> = date_part.split('-').collect();
    let year: i64 = match date_parts.first().and_then(|s| s.parse().ok()) {
        Some(y) => y,
        None => return NAT,
    };
    let month: i64 = date_parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);
    let day: i64 = date_parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);

    // Calculate days since epoch
    let days = ymd_to_days(year, month, day);

    // Parse time if present
    let (hours, minutes, seconds, subseconds) = if let Some(time_str) = time_part {
        parse_time_part(time_str)
    } else {
        (0, 0, 0, 0.0)
    };

    // Convert to target unit
    match unit {
        TimeUnit::Years => year - 1970,
        TimeUnit::Months => (year - 1970) * 12 + (month - 1),
        TimeUnit::Weeks => days / 7,
        TimeUnit::Days => days,
        TimeUnit::Hours => days * 24 + hours,
        TimeUnit::Minutes => days * 24 * 60 + hours * 60 + minutes,
        TimeUnit::Seconds => days * 86400 + hours * 3600 + minutes * 60 + seconds,
        TimeUnit::Milliseconds => {
            let secs = days * 86400 + hours * 3600 + minutes * 60 + seconds;
            secs * 1000 + (subseconds * 1000.0) as i64
        }
        TimeUnit::Microseconds => {
            let secs = days * 86400 + hours * 3600 + minutes * 60 + seconds;
            secs * 1_000_000 + (subseconds * 1_000_000.0) as i64
        }
        TimeUnit::Nanoseconds => {
            let secs = days * 86400 + hours * 3600 + minutes * 60 + seconds;
            secs * 1_000_000_000 + (subseconds * 1_000_000_000.0) as i64
        }
    }
}

/// Parse time part HH:MM:SS.sss returning (hours, minutes, seconds, subseconds as fraction).
fn parse_time_part(s: &str) -> (i64, i64, i64, f64) {
    let parts: Vec<&str> = s.split(':').collect();
    let hours: i64 = parts.first().and_then(|s| s.parse().ok()).unwrap_or(0);
    let minutes: i64 = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);

    // Seconds may have fractional part
    let (seconds, subseconds) = if let Some(sec_str) = parts.get(2) {
        if let Some(dot_pos) = sec_str.find('.') {
            let secs: i64 = sec_str[..dot_pos].parse().unwrap_or(0);
            let frac: f64 = format!("0.{}", &sec_str[dot_pos+1..]).parse().unwrap_or(0.0);
            (secs, frac)
        } else {
            (sec_str.parse().unwrap_or(0), 0.0)
        }
    } else {
        (0, 0.0)
    };

    (hours, minutes, seconds, subseconds)
}

/// Convert datetime64 value from one unit to another.
/// Returns NAT if conversion would overflow or for variable-length units.
pub fn convert_datetime64(value: i64, from_unit: TimeUnit, to_unit: TimeUnit) -> i64 {
    if value == NAT {
        return NAT;
    }
    if from_unit == to_unit {
        return value;
    }

    // For fixed-length units, we can convert via common unit (nanoseconds)
    // For variable-length units (Y, M), we need special handling
    match (from_unit, to_unit) {
        // Variable to variable
        (TimeUnit::Years, TimeUnit::Months) => value * 12,
        (TimeUnit::Months, TimeUnit::Years) => value / 12,

        // Variable to fixed - convert via date calculation
        (TimeUnit::Years, to) => {
            let year = 1970 + value;
            let days = ymd_to_days(year, 1, 1);
            days_to_unit(days, to)
        }
        (TimeUnit::Months, to) => {
            let total_months = 1970 * 12 + value;
            let year = total_months / 12;
            let month = (total_months % 12) + 1;
            let days = ymd_to_days(year, month, 1);
            days_to_unit(days, to)
        }

        // Fixed to variable - approximate conversion
        (from, TimeUnit::Years) => {
            let days = unit_to_days(value, from);
            let (year, _, _) = days_to_ymd(days);
            year - 1970
        }
        (from, TimeUnit::Months) => {
            let days = unit_to_days(value, from);
            let (year, month, _) = days_to_ymd(days);
            (year - 1970) * 12 + (month - 1)
        }

        // Fixed to fixed - use factors
        (from, to) => {
            let from_factor = from.to_ns_factor().unwrap();
            let to_factor = to.to_ns_factor().unwrap();
            if from_factor >= to_factor {
                value * (from_factor / to_factor)
            } else {
                value / (to_factor / from_factor)
            }
        }
    }
}

fn days_to_unit(days: i64, unit: TimeUnit) -> i64 {
    match unit {
        TimeUnit::Years | TimeUnit::Months => {
            let (year, month, _) = days_to_ymd(days);
            if matches!(unit, TimeUnit::Years) {
                year - 1970
            } else {
                (year - 1970) * 12 + (month - 1)
            }
        }
        TimeUnit::Weeks => days / 7,
        TimeUnit::Days => days,
        TimeUnit::Hours => days * 24,
        TimeUnit::Minutes => days * 24 * 60,
        TimeUnit::Seconds => days * 86400,
        TimeUnit::Milliseconds => days * 86400 * 1000,
        TimeUnit::Microseconds => days * 86400 * 1_000_000,
        TimeUnit::Nanoseconds => days * 86400 * 1_000_000_000,
    }
}

fn unit_to_days(value: i64, unit: TimeUnit) -> i64 {
    match unit {
        TimeUnit::Years | TimeUnit::Months => {
            // These should be handled separately
            0
        }
        TimeUnit::Weeks => value * 7,
        TimeUnit::Days => value,
        TimeUnit::Hours => value / 24,
        TimeUnit::Minutes => value / (24 * 60),
        TimeUnit::Seconds => value / 86400,
        TimeUnit::Milliseconds => value / (86400 * 1000),
        TimeUnit::Microseconds => value / (86400 * 1_000_000),
        TimeUnit::Nanoseconds => value / (86400 * 1_000_000_000),
    }
}
