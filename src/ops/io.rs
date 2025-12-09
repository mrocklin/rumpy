//! I/O operations for reading and writing arrays to files.

use crate::array::{ArrayBuffer, ArrayFlags, DType, RumpyArray};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::Arc;

/// Load data from a text file.
///
/// Each row in the file must have the same number of columns.
/// Lines starting with `comments` are skipped.
pub fn loadtxt(
    path: &Path,
    dtype: DType,
    comments: &str,
    delimiter: Option<&str>,
    skiprows: usize,
    usecols: Option<&[usize]>,
    max_rows: Option<usize>,
) -> Result<RumpyArray, String> {
    let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
    let reader = BufReader::new(file);

    let mut data: Vec<f64> = Vec::new();
    let mut ncols: Option<usize> = None;
    let mut nrows = 0usize;
    let delimiter = delimiter.unwrap_or_default();

    for (line_num, line_result) in reader.lines().enumerate() {
        // Skip rows
        if line_num < skiprows {
            continue;
        }

        // Check max_rows
        if let Some(max) = max_rows {
            if nrows >= max {
                break;
            }
        }

        let line = line_result.map_err(|e| format!("Failed to read line: {}", e))?;
        let trimmed = line.trim();

        // Skip empty lines and comments
        if trimmed.is_empty() || (!comments.is_empty() && trimmed.starts_with(comments)) {
            continue;
        }

        // Parse values
        let values: Vec<&str> = if delimiter.is_empty() {
            trimmed.split_whitespace().collect()
        } else {
            trimmed.split(delimiter).map(|s| s.trim()).collect()
        };

        // Select columns
        let selected: Vec<&str> = if let Some(cols) = usecols {
            cols.iter()
                .filter_map(|&i| values.get(i).copied())
                .collect()
        } else {
            values
        };

        // Validate column count
        match ncols {
            Some(n) if n != selected.len() => {
                return Err(format!(
                    "Line {} has {} columns, expected {}",
                    line_num + 1,
                    selected.len(),
                    n
                ));
            }
            None => ncols = Some(selected.len()),
            _ => {}
        }

        // Parse and collect values
        for s in selected {
            let val: f64 = s
                .parse()
                .map_err(|_| format!("Cannot parse '{}' as number at line {}", s, line_num + 1))?;
            data.push(val);
        }

        nrows += 1;
    }

    if nrows == 0 {
        return Ok(RumpyArray::zeros(vec![0, 0], dtype));
    }

    let ncols = ncols.unwrap_or(0);
    let shape = if ncols == 1 {
        vec![nrows]
    } else {
        vec![nrows, ncols]
    };

    Ok(RumpyArray::from_vec_with_shape(data, shape, dtype))
}

/// Save array to a text file.
pub fn savetxt(
    path: &Path,
    arr: &RumpyArray,
    delimiter: &str,
    newline: &str,
    header: &str,
    footer: &str,
    fmt: &str,
    comments: &str,
) -> Result<(), String> {
    let file = File::create(path).map_err(|e| format!("Failed to create file: {}", e))?;
    let mut writer = BufWriter::new(file);

    // Write header
    if !header.is_empty() {
        for line in header.lines() {
            writer
                .write_all(comments.as_bytes())
                .map_err(|e| e.to_string())?;
            writer
                .write_all(line.as_bytes())
                .map_err(|e| e.to_string())?;
            writer
                .write_all(newline.as_bytes())
                .map_err(|e| e.to_string())?;
        }
    }

    let ptr = arr.data_ptr();
    let dtype = arr.dtype();
    let ops = dtype.ops();
    let shape = arr.shape();
    let ndim = arr.ndim();

    if ndim == 0 {
        // Scalar
        let val = arr.get_element(&[]);
        let formatted = format_value(val, fmt);
        writer
            .write_all(formatted.as_bytes())
            .map_err(|e| e.to_string())?;
        writer
            .write_all(newline.as_bytes())
            .map_err(|e| e.to_string())?;
    } else if ndim == 1 {
        // 1D array - write as column or row based on convention
        // NumPy writes 1D as a single column
        for offset in arr.iter_offsets() {
            let val = unsafe { ops.read_f64(ptr, offset) }.unwrap_or(0.0);
            let formatted = format_value(val, fmt);
            writer
                .write_all(formatted.as_bytes())
                .map_err(|e| e.to_string())?;
            writer
                .write_all(newline.as_bytes())
                .map_err(|e| e.to_string())?;
        }
    } else {
        // 2D+ array - write rows
        let nrows = shape[0];
        let ncols: usize = shape[1..].iter().product();

        // Collect all offsets once to avoid O(n²) iterator recreation
        let offsets: Vec<_> = arr.iter_offsets().collect();

        for row in 0..nrows {
            for col in 0..ncols {
                if col > 0 {
                    writer
                        .write_all(delimiter.as_bytes())
                        .map_err(|e| e.to_string())?;
                }

                let flat_idx = row * ncols + col;
                let offset = offsets[flat_idx];
                let val = unsafe { ops.read_f64(ptr, offset) }.unwrap_or(0.0);
                let formatted = format_value(val, fmt);
                writer
                    .write_all(formatted.as_bytes())
                    .map_err(|e| e.to_string())?;
            }
            writer
                .write_all(newline.as_bytes())
                .map_err(|e| e.to_string())?;
        }
    }

    // Write footer
    if !footer.is_empty() {
        for line in footer.lines() {
            writer
                .write_all(comments.as_bytes())
                .map_err(|e| e.to_string())?;
            writer
                .write_all(line.as_bytes())
                .map_err(|e| e.to_string())?;
            writer
                .write_all(newline.as_bytes())
                .map_err(|e| e.to_string())?;
        }
    }

    writer.flush().map_err(|e| e.to_string())?;
    Ok(())
}

/// Format a value using a format string.
fn format_value(val: f64, fmt: &str) -> String {
    if fmt.is_empty() || fmt == "%.18e" {
        // Default scientific notation
        format!("{:.18e}", val)
    } else if fmt.starts_with("%.") && fmt.ends_with('e') {
        // Scientific notation with precision
        if let Ok(precision) = fmt[2..fmt.len() - 1].parse::<usize>() {
            format!("{:.prec$e}", val, prec = precision)
        } else {
            format!("{:.18e}", val)
        }
    } else if fmt.starts_with("%.") && fmt.ends_with('f') {
        // Fixed-point with precision
        if let Ok(precision) = fmt[2..fmt.len() - 1].parse::<usize>() {
            format!("{:.prec$}", val, prec = precision)
        } else {
            format!("{}", val)
        }
    } else if fmt.starts_with("%.") && fmt.ends_with('g') {
        // General format
        if let Ok(precision) = fmt[2..fmt.len() - 1].parse::<usize>() {
            // Rust doesn't have %g, approximate it
            if val.abs() < 1e-4 || val.abs() >= 10f64.powi(precision as i32) {
                format!("{:.prec$e}", val, prec = precision)
            } else {
                format!("{:.prec$}", val, prec = precision)
            }
        } else {
            format!("{}", val)
        }
    } else if fmt.starts_with('%') && fmt.ends_with('d') {
        // Integer format
        format!("{}", val as i64)
    } else {
        format!("{}", val)
    }
}

/// Load data from a text file with missing value handling.
pub fn genfromtxt(
    path: &Path,
    dtype: DType,
    comments: &str,
    delimiter: Option<&str>,
    skip_header: usize,
    skip_footer: usize,
    usecols: Option<&[usize]>,
    missing_values: &str,
    filling_values: f64,
    max_rows: Option<usize>,
) -> Result<RumpyArray, String> {
    let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
    let reader = BufReader::new(file);

    // Read all lines first to handle skip_footer
    let lines: Vec<String> = reader
        .lines()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("Failed to read file: {}", e))?;

    let end_idx = lines.len().saturating_sub(skip_footer);
    let delimiter = delimiter.unwrap_or_default();

    let mut data: Vec<f64> = Vec::new();
    let mut ncols: Option<usize> = None;
    let mut nrows = 0usize;
    let mut processed_rows = 0usize;

    for (line_num, line) in lines.iter().enumerate() {
        // Skip header rows
        if line_num < skip_header {
            continue;
        }

        // Skip footer rows
        if line_num >= end_idx {
            break;
        }

        // Check max_rows
        if let Some(max) = max_rows {
            if processed_rows >= max {
                break;
            }
        }

        let trimmed = line.trim();

        // Skip empty lines and comments
        if trimmed.is_empty() || (!comments.is_empty() && trimmed.starts_with(comments)) {
            continue;
        }

        // Parse values
        let values: Vec<&str> = if delimiter.is_empty() {
            trimmed.split_whitespace().collect()
        } else {
            trimmed.split(delimiter).map(|s| s.trim()).collect()
        };

        // Select columns
        let selected: Vec<&str> = if let Some(cols) = usecols {
            cols.iter()
                .filter_map(|&i| values.get(i).copied())
                .collect()
        } else {
            values
        };

        // Validate column count (first non-empty row sets it)
        match ncols {
            Some(n) if n != selected.len() => {
                return Err(format!(
                    "Line {} has {} columns, expected {}",
                    line_num + 1,
                    selected.len(),
                    n
                ));
            }
            None => ncols = Some(selected.len()),
            _ => {}
        }

        // Parse values with missing value handling
        for s in selected {
            let val = if s.is_empty() || s == missing_values {
                filling_values
            } else {
                s.parse().unwrap_or(filling_values)
            };
            data.push(val);
        }

        nrows += 1;
        processed_rows += 1;
    }

    if nrows == 0 {
        return Ok(RumpyArray::zeros(vec![0, 0], dtype));
    }

    let ncols = ncols.unwrap_or(0);
    let shape = if ncols == 1 {
        vec![nrows]
    } else {
        vec![nrows, ncols]
    };

    Ok(RumpyArray::from_vec_with_shape(data, shape, dtype))
}

// NPY format constants
const NPY_MAGIC: &[u8] = b"\x93NUMPY";
const NPY_VERSION_1_0: [u8; 2] = [1, 0];
const NPY_VERSION_2_0: [u8; 2] = [2, 0];

/// Save array to .npy format.
pub fn save_npy(path: &Path, arr: &RumpyArray) -> Result<(), String> {
    let file = File::create(path).map_err(|e| format!("Failed to create file: {}", e))?;
    let mut writer = BufWriter::new(file);
    write_npy(&mut writer, arr)?;
    writer.flush().map_err(|e| e.to_string())?;
    Ok(())
}

/// Write array in NPY format to a writer.
pub fn write_npy<W: Write>(writer: &mut W, arr: &RumpyArray) -> Result<(), String> {
    // Build header dictionary
    let descr = arr.dtype().ops().typestr();
    let fortran_order = arr.is_f_contiguous() && !arr.is_c_contiguous();
    let shape_str = format!(
        "({}{})",
        arr.shape()
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", "),
        if arr.ndim() == 1 { "," } else { "" }
    );

    let header = format!(
        "{{'descr': '{}', 'fortran_order': {}, 'shape': {}}}",
        descr,
        if fortran_order { "True" } else { "False" },
        shape_str
    );

    // Calculate padding for 64-byte alignment (version 1.0)
    let header_len = header.len();
    let total_header_len = 10 + header_len; // magic(6) + version(2) + header_len(2)
    let padding_needed = (64 - (total_header_len % 64)) % 64;
    let padded_header_len = header_len + padding_needed;

    // Check if we need version 2.0 (header > 65535 bytes)
    if padded_header_len > 65535 {
        // Version 2.0: 4-byte header length
        writer.write_all(NPY_MAGIC).map_err(|e| e.to_string())?;
        writer
            .write_all(&NPY_VERSION_2_0)
            .map_err(|e| e.to_string())?;
        let len_bytes = (padded_header_len as u32).to_le_bytes();
        writer.write_all(&len_bytes).map_err(|e| e.to_string())?;
    } else {
        // Version 1.0: 2-byte header length
        writer.write_all(NPY_MAGIC).map_err(|e| e.to_string())?;
        writer
            .write_all(&NPY_VERSION_1_0)
            .map_err(|e| e.to_string())?;
        let len_bytes = (padded_header_len as u16).to_le_bytes();
        writer.write_all(&len_bytes).map_err(|e| e.to_string())?;
    }

    // Write header
    writer
        .write_all(header.as_bytes())
        .map_err(|e| e.to_string())?;

    // Write padding (spaces, ending with newline)
    if padding_needed > 0 {
        let mut padding = vec![b' '; padding_needed];
        padding[padding_needed - 1] = b'\n';
        writer.write_all(&padding).map_err(|e| e.to_string())?;
    }

    // Write data - need contiguous array
    let contiguous = if arr.is_c_contiguous() {
        arr.clone()
    } else {
        arr.copy()
    };

    let data = unsafe { std::slice::from_raw_parts(contiguous.data_ptr(), contiguous.nbytes()) };
    writer.write_all(data).map_err(|e| e.to_string())?;

    Ok(())
}

/// Load array from .npy format.
pub fn load_npy(path: &Path) -> Result<RumpyArray, String> {
    let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
    let mut reader = BufReader::new(file);
    read_npy(&mut reader)
}

/// Read array in NPY format from a reader.
pub fn read_npy<R: Read>(reader: &mut R) -> Result<RumpyArray, String> {
    // Read magic
    let mut magic = [0u8; 6];
    reader
        .read_exact(&mut magic)
        .map_err(|e| format!("Failed to read magic: {}", e))?;
    if magic != NPY_MAGIC {
        return Err("Invalid NPY file magic".to_string());
    }

    // Read version
    let mut version = [0u8; 2];
    reader
        .read_exact(&mut version)
        .map_err(|e| format!("Failed to read version: {}", e))?;

    // Read header length
    let header_len = if version[0] == 1 {
        let mut len_bytes = [0u8; 2];
        reader
            .read_exact(&mut len_bytes)
            .map_err(|e| format!("Failed to read header length: {}", e))?;
        u16::from_le_bytes(len_bytes) as usize
    } else if version[0] >= 2 {
        let mut len_bytes = [0u8; 4];
        reader
            .read_exact(&mut len_bytes)
            .map_err(|e| format!("Failed to read header length: {}", e))?;
        u32::from_le_bytes(len_bytes) as usize
    } else {
        return Err(format!("Unsupported NPY version: {}.{}", version[0], version[1]));
    };

    // Read header
    let mut header_bytes = vec![0u8; header_len];
    reader
        .read_exact(&mut header_bytes)
        .map_err(|e| format!("Failed to read header: {}", e))?;

    let header =
        String::from_utf8(header_bytes).map_err(|_| "Header is not valid UTF-8".to_string())?;

    // Parse header (simplified parsing of Python dict literal)
    let (dtype, shape, fortran_order) = parse_npy_header(&header)?;

    // Create array and read data
    let size: usize = shape.iter().product();
    let nbytes = size * dtype.itemsize();

    let mut buffer = ArrayBuffer::new(nbytes);
    let buf_ptr = buffer.as_mut_ptr();
    let buf_slice = unsafe { std::slice::from_raw_parts_mut(buf_ptr, nbytes) };
    reader
        .read_exact(buf_slice)
        .map_err(|e| format!("Failed to read data: {}", e))?;

    // Compute strides based on fortran_order
    let strides = if fortran_order {
        compute_f_strides(&shape, dtype.itemsize())
    } else {
        compute_c_strides(&shape, dtype.itemsize())
    };

    let mut flags = ArrayFlags::WRITEABLE;
    if !fortran_order {
        flags |= ArrayFlags::C_CONTIGUOUS;
    }
    if fortran_order || shape.iter().filter(|&&d| d > 1).count() <= 1 {
        flags |= ArrayFlags::F_CONTIGUOUS;
    }

    Ok(RumpyArray {
        buffer: Arc::new(buffer),
        offset: 0,
        shape,
        strides,
        dtype,
        flags,
    })
}

/// Parse NPY header dictionary.
fn parse_npy_header(header: &str) -> Result<(DType, Vec<usize>, bool), String> {
    let header = header.trim();

    // Extract descr
    let descr = extract_string_value(header, "descr")?;
    let dtype = descr
        .parse::<DType>()
        .map_err(|_| format!("Unsupported dtype: {}", descr))?;

    // Extract fortran_order
    let fortran_order = extract_bool_value(header, "fortran_order")?;

    // Extract shape
    let shape = extract_shape(header)?;

    Ok((dtype, shape, fortran_order))
}

/// Extract a string value from header dict.
fn extract_string_value(header: &str, key: &str) -> Result<String, String> {
    // Look for 'key': 'value' pattern
    let patterns = [
        format!("'{}': '", key),
        format!("\"{}': '", key),
        format!("'{}': \"", key),
        format!("\"{}': \"", key),
    ];

    for pattern in &patterns {
        if let Some(start) = header.find(pattern) {
            let value_start = start + pattern.len();
            let rest = &header[value_start..];
            // Find closing quote
            for (i, c) in rest.char_indices() {
                if c == '\'' || c == '"' {
                    return Ok(rest[..i].to_string());
                }
            }
        }
    }

    Err(format!("Key '{}' not found in header", key))
}

/// Extract a boolean value from header dict.
fn extract_bool_value(header: &str, key: &str) -> Result<bool, String> {
    let patterns = [format!("'{}': ", key), format!("\"{}': ", key)];

    for pattern in &patterns {
        if let Some(start) = header.find(pattern) {
            let value_start = start + pattern.len();
            let rest = &header[value_start..];
            if rest.starts_with("True") {
                return Ok(true);
            } else if rest.starts_with("False") {
                return Ok(false);
            }
        }
    }

    Err(format!("Key '{}' not found in header", key))
}

/// Extract shape tuple from header dict.
fn extract_shape(header: &str) -> Result<Vec<usize>, String> {
    // Look for 'shape': (...)
    let patterns = ["'shape': (", "\"shape': (", "'shape':("];

    for pattern in &patterns {
        if let Some(start) = header.find(pattern) {
            let value_start = start + pattern.len();
            let rest = &header[value_start..];

            // Find closing paren
            if let Some(end) = rest.find(')') {
                let shape_str = &rest[..end];

                // Parse shape values
                let shape: Vec<usize> = shape_str
                    .split(',')
                    .filter_map(|s| {
                        let trimmed = s.trim();
                        if trimmed.is_empty() {
                            None
                        } else {
                            trimmed.parse().ok()
                        }
                    })
                    .collect();

                return Ok(shape);
            }
        }
    }

    Err("Shape not found in header".to_string())
}

/// Compute C-contiguous strides.
fn compute_c_strides(shape: &[usize], itemsize: usize) -> Vec<isize> {
    let mut strides = vec![0isize; shape.len()];
    let mut stride = itemsize as isize;
    for i in (0..shape.len()).rev() {
        strides[i] = stride;
        stride *= shape[i] as isize;
    }
    strides
}

/// Compute F-contiguous strides.
fn compute_f_strides(shape: &[usize], itemsize: usize) -> Vec<isize> {
    let mut strides = vec![0isize; shape.len()];
    let mut stride = itemsize as isize;
    for i in 0..shape.len() {
        strides[i] = stride;
        stride *= shape[i] as isize;
    }
    strides
}

/// Create array from raw bytes buffer.
pub fn frombuffer(data: &[u8], dtype: DType, count: isize, offset: usize) -> Result<RumpyArray, String> {
    let itemsize = dtype.itemsize();
    let available_bytes = data.len().saturating_sub(offset);
    let max_elements = available_bytes / itemsize;

    let n = if count < 0 {
        max_elements
    } else {
        let requested = count as usize;
        if requested > max_elements {
            return Err(format!(
                "Not enough data: requested {} elements but only {} available",
                requested, max_elements
            ));
        }
        requested
    };

    let nbytes = n * itemsize;
    let mut buffer = ArrayBuffer::new(nbytes);
    let buf_ptr = buffer.as_mut_ptr();

    // Copy data
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr().add(offset), buf_ptr, nbytes);
    }

    let shape = vec![n];
    let strides = vec![itemsize as isize];

    Ok(RumpyArray {
        buffer: Arc::new(buffer),
        offset: 0,
        shape,
        strides,
        dtype,
        flags: ArrayFlags::default(),
    })
}

/// Read array data from a binary file.
pub fn fromfile(
    path: &Path,
    dtype: DType,
    count: isize,
    offset: usize,
    sep: &str,
) -> Result<RumpyArray, String> {
    if !sep.is_empty() {
        // Text mode - read as delimited text
        return fromfile_text(path, dtype, count, sep);
    }

    // Binary mode
    let mut file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;

    // Seek to offset
    if offset > 0 {
        file.seek_relative(offset as i64)
            .map_err(|e| format!("Failed to seek: {}", e))?;
    }

    let itemsize = dtype.itemsize();

    // Determine count
    let file_size = file
        .metadata()
        .map_err(|e| format!("Failed to get file size: {}", e))?
        .len() as usize;
    let available_bytes = file_size.saturating_sub(offset);
    let max_elements = available_bytes / itemsize;

    let n = if count < 0 {
        max_elements
    } else {
        let requested = count as usize;
        if requested > max_elements {
            max_elements // NumPy behavior: read as much as available
        } else {
            requested
        }
    };

    let nbytes = n * itemsize;
    let mut buffer = ArrayBuffer::new(nbytes);
    let buf_ptr = buffer.as_mut_ptr();

    let buf_slice = unsafe { std::slice::from_raw_parts_mut(buf_ptr, nbytes) };
    file.read_exact(buf_slice)
        .map_err(|e| format!("Failed to read data: {}", e))?;

    let shape = vec![n];
    let strides = vec![itemsize as isize];

    Ok(RumpyArray {
        buffer: Arc::new(buffer),
        offset: 0,
        shape,
        strides,
        dtype,
        flags: ArrayFlags::default(),
    })
}

/// Read array from text file with separator.
fn fromfile_text(path: &Path, dtype: DType, count: isize, sep: &str) -> Result<RumpyArray, String> {
    let content =
        std::fs::read_to_string(path).map_err(|e| format!("Failed to read file: {}", e))?;

    let values: Vec<f64> = content
        .split(sep)
        .filter_map(|s| {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                None
            } else {
                trimmed.parse().ok()
            }
        })
        .collect();

    let n = if count < 0 {
        values.len()
    } else {
        (count as usize).min(values.len())
    };

    let data = values[..n].to_vec();
    Ok(RumpyArray::from_vec(data, dtype))
}

/// Write array to binary file.
pub fn tofile(arr: &RumpyArray, path: &Path, sep: &str, format: &str) -> Result<(), String> {
    let file = File::create(path).map_err(|e| format!("Failed to create file: {}", e))?;
    let mut writer = BufWriter::new(file);

    if sep.is_empty() {
        // Binary mode
        let contiguous = if arr.is_c_contiguous() {
            arr.clone()
        } else {
            arr.copy()
        };

        let data = unsafe { std::slice::from_raw_parts(contiguous.data_ptr(), contiguous.nbytes()) };
        writer.write_all(data).map_err(|e| e.to_string())?;
    } else {
        // Text mode
        let ptr = arr.data_ptr();
        let dtype = arr.dtype();
        let ops = dtype.ops();
        let mut first = true;

        for offset in arr.iter_offsets() {
            if !first {
                writer.write_all(sep.as_bytes()).map_err(|e| e.to_string())?;
            }
            first = false;

            let val = unsafe { ops.read_f64(ptr, offset) }.unwrap_or(0.0);
            let formatted = format_value(val, format);
            writer.write_all(formatted.as_bytes()).map_err(|e| e.to_string())?;
        }
    }

    writer.flush().map_err(|e| e.to_string())?;
    Ok(())
}

// Re-export helper for seek
trait SeekRelative {
    fn seek_relative(&mut self, offset: i64) -> std::io::Result<()>;
}

impl SeekRelative for File {
    fn seek_relative(&mut self, offset: i64) -> std::io::Result<()> {
        use std::io::{Seek, SeekFrom};
        self.seek(SeekFrom::Current(offset))?;
        Ok(())
    }
}

/// Save multiple arrays to a single .npz file (uncompressed).
pub fn savez(path: &Path, arrays: &[(&str, &RumpyArray)]) -> Result<(), String> {
    savez_impl(path, arrays, false)
}

/// Save multiple arrays to a single .npz file (compressed).
pub fn savez_compressed(path: &Path, arrays: &[(&str, &RumpyArray)]) -> Result<(), String> {
    savez_impl(path, arrays, true)
}

fn savez_impl(path: &Path, arrays: &[(&str, &RumpyArray)], compressed: bool) -> Result<(), String> {
    let file = File::create(path).map_err(|e| format!("Failed to create file: {}", e))?;
    let mut zip = zip::ZipWriter::new(file);

    let options = if compressed {
        zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated)
    } else {
        zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored)
    };

    for (name, arr) in arrays {
        let file_name = format!("{}.npy", name);
        zip.start_file(&file_name, options)
            .map_err(|e| format!("Failed to start file {}: {}", file_name, e))?;

        write_npy(&mut zip, arr)?;
    }

    zip.finish().map_err(|e| format!("Failed to finish zip: {}", e))?;
    Ok(())
}

/// Load arrays from a .npz file.
/// Returns a vector of (name, array) pairs.
pub fn load_npz(path: &Path) -> Result<Vec<(String, RumpyArray)>, String> {
    let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|e| format!("Failed to open zip archive: {}", e))?;

    let mut results = Vec::new();

    for i in 0..archive.len() {
        let mut file = archive.by_index(i)
            .map_err(|e| format!("Failed to read archive entry: {}", e))?;

        let name = file.name().to_string();
        if !name.ends_with(".npy") {
            continue;
        }

        // Read all data into memory first
        let mut data = Vec::new();
        file.read_to_end(&mut data)
            .map_err(|e| format!("Failed to read {}: {}", name, e))?;

        // Parse the npy data
        let mut cursor = std::io::Cursor::new(data);
        let arr = read_npy(&mut cursor)?;

        // Extract name without .npy extension
        let array_name = name.trim_end_matches(".npy").to_string();
        results.push((array_name, arr));
    }

    Ok(results)
}
