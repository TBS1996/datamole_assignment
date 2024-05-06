# datamole_assignment


Library for statistical computation of a rolling window of i32-values inputted as raw bytes.

features:

- no_std compatible.
- Standard deviation and mean of values 
- Get a random sample from the SD and mean value.
- Supports both little and big endian representation of i32. 
- Buffer that keeps track of incomplete i32 values. 
- No heap-allocations.
- Compile-time assert that the window is non-zero.
