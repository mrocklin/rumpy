//! Array formatting for repr and str display.

use super::dtype::DTypeKind;
use super::RumpyArray;

impl RumpyArray {
    /// Format array as repr string: `array([...])`.
    pub fn format_repr(&self) -> String {
        let threshold = 1000;
        let edgeitems = 3;
        let truncate = self.size() > threshold;

        let data = self.format_data(", ", truncate, edgeitems, 7); // 7 = len("array([")

        // Determine if dtype suffix is needed
        let needs_dtype =
            !matches!(self.dtype.kind(), DTypeKind::Int64 | DTypeKind::Float64 | DTypeKind::Bool);

        // Show shape for truncated arrays (numpy shows shape for any truncated array)
        let show_shape = truncate;

        // Format shape as tuple: (3, 4) not [3, 4]
        let shape_str = if self.ndim() == 1 {
            format!("({},)", self.shape[0])
        } else {
            format!(
                "({})",
                self.shape
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };

        // Empty arrays always show dtype
        let suffix = if self.size() == 0 {
            format!(", dtype={}", self.dtype.ops().name())
        } else if show_shape && needs_dtype {
            format!(
                ", shape={}, dtype={}",
                shape_str,
                self.dtype.ops().name()
            )
        } else if show_shape {
            format!(", shape={}", shape_str)
        } else if needs_dtype {
            format!(", dtype={}", self.dtype.ops().name())
        } else {
            String::new()
        };

        format!("array({}{})", data, suffix)
    }

    /// Format array as str: `[...]` without wrapper.
    pub fn format_str(&self) -> String {
        let threshold = 1000;
        let edgeitems = 3;
        let truncate = self.size() > threshold;

        self.format_data(" ", truncate, edgeitems, 1) // 1 = len("[")
    }

    /// Format the data portion with given separator.
    /// base_indent is extra indentation for multi-line arrays (e.g., 7 for "array([").
    fn format_data(
        &self,
        sep: &str,
        truncate: bool,
        edgeitems: usize,
        base_indent: usize,
    ) -> String {
        if self.size() == 0 {
            return "[]".to_string();
        }

        // Collect all element strings for width calculation
        let elements = self.collect_elements(truncate, edgeitems);

        // Calculate max width for alignment
        let max_width = elements.iter().map(|s| s.len()).max().unwrap_or(0);

        // Format recursively
        self.format_recursive(&[], sep, truncate, edgeitems, max_width, base_indent)
    }

    /// Get indices to display for a dimension, with truncation.
    /// Returns (indices, needs_ellipsis).
    fn display_indices(
        &self,
        dim_size: usize,
        truncate: bool,
        edgeitems: usize,
    ) -> (Vec<usize>, bool) {
        if truncate && dim_size > 2 * edgeitems {
            let indices = (0..edgeitems)
                .chain((dim_size - edgeitems)..dim_size)
                .collect();
            (indices, true)
        } else {
            ((0..dim_size).collect(), false)
        }
    }

    /// Collect element strings (for width calculation).
    fn collect_elements(&self, truncate: bool, edgeitems: usize) -> Vec<String> {
        let mut result = Vec::new();
        self.collect_elements_recursive(&mut result, &[], truncate, edgeitems);
        result
    }

    fn collect_elements_recursive(
        &self,
        result: &mut Vec<String>,
        prefix: &[usize],
        truncate: bool,
        edgeitems: usize,
    ) {
        let depth = prefix.len();
        let (indices, _) = self.display_indices(self.shape[depth], truncate, edgeitems);
        let ptr = self.data_ptr();

        for i in indices {
            let mut new_prefix = prefix.to_vec();
            new_prefix.push(i);

            if new_prefix.len() == self.ndim() {
                let offset = self.byte_offset_for(&new_prefix);
                result.push(unsafe { self.dtype.ops().format_element(ptr, offset) });
            } else {
                self.collect_elements_recursive(result, &new_prefix, truncate, edgeitems);
            }
        }
    }

    /// Recursive formatting with proper indentation.
    fn format_recursive(
        &self,
        prefix: &[usize],
        sep: &str,
        truncate: bool,
        edgeitems: usize,
        width: usize,
        base_indent: usize,
    ) -> String {
        let depth = prefix.len();

        if depth == self.ndim() {
            let ptr = self.data_ptr();
            let offset = self.byte_offset_for(prefix);
            let s = unsafe { self.dtype.ops().format_element(ptr, offset) };
            return format!("{:>w$}", s, w = width);
        }

        let dim_size = self.shape[depth];
        let is_last_dim = depth == self.ndim() - 1;
        let (indices, has_ellipsis) = self.display_indices(dim_size, truncate, edgeitems);

        let mut parts = Vec::new();
        for (idx_pos, i) in indices.into_iter().enumerate() {
            // Insert ellipsis between front and back sections
            if has_ellipsis && idx_pos == edgeitems {
                parts.push("...".to_string());
            }

            let mut new_prefix = prefix.to_vec();
            new_prefix.push(i);
            parts.push(self.format_recursive(
                &new_prefix,
                sep,
                truncate,
                edgeitems,
                width,
                base_indent,
            ));
        }

        if is_last_dim {
            format!("[{}]", parts.join(sep))
        } else {
            // Multi-line formatting for higher dimensions
            let indent = " ".repeat(base_indent + depth);
            let inner_sep = if depth == self.ndim() - 2 {
                format!("{}\n{}", sep.trim_end(), indent)
            } else {
                // Extra blank line between highest-level blocks
                format!("{}\n\n{}", sep.trim_end(), indent)
            };
            format!("[{}]", parts.join(&inner_sep))
        }
    }
}
