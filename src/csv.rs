#![allow(dead_code)]

use std::{
    fmt::Display,
    fs::File,
    io::{self, BufWriter, Write},
    path::Path,
};

pub(crate) fn write_csv<T: Display>(arr: &[T], path: impl AsRef<Path>) -> Result<(), io::Error> {
    let mut file = BufWriter::new(File::create(path)?);

    for elem in arr {
        writeln!(file, "{elem}")?;
    }

    Ok(())
}

pub(crate) fn write_csv_2<T: Display>(
    arr: &[T],
    arr2: &[T],
    path: impl AsRef<Path>,
) -> Result<(), io::Error> {
    let mut file = BufWriter::new(File::create(path)?);

    for (elem1, elem2) in arr.iter().zip(arr2) {
        writeln!(file, "{elem1}, {elem2}")?;
    }

    Ok(())
}
