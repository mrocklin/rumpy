use bitflags::bitflags;

bitflags! {
    /// Array flags indicating memory layout and permissions.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct ArrayFlags: u32 {
        /// Data is in row-major (C) order.
        const C_CONTIGUOUS = 0x0001;
        /// Data is in column-major (Fortran) order.
        const F_CONTIGUOUS = 0x0002;
        /// Array is writeable.
        const WRITEABLE = 0x0400;
    }
}

impl ArrayFlags {
    /// Check if array is contiguous (either C or F order).
    pub fn is_contiguous(&self) -> bool {
        self.contains(Self::C_CONTIGUOUS) || self.contains(Self::F_CONTIGUOUS)
    }
}

impl Default for ArrayFlags {
    fn default() -> Self {
        ArrayFlags::C_CONTIGUOUS | ArrayFlags::WRITEABLE
    }
}
