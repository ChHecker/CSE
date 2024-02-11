pub fn selection_sort<T: Ord>(arr: &mut [T]) {
    for i in 0..arr.len() {
        let min = match arr.iter().enumerate().skip(i).min_by_key(|(_, t)| *t) {
            Some((min, _)) => min,
            None => return,
        };
        arr.swap(i, min);
    }
}

pub fn merge_sort<T: Clone + Ord>(arr: &[T]) -> Vec<T> {
    let len = arr.len();
    let mid = len / 2;

    let mut merged: Vec<T> = Vec::with_capacity(len);
    if len > 2 {
        let left = merge_sort(&arr[0..mid]).into_iter();
        let mut right = merge_sort(&arr[mid..len]).into_iter();

        // Merge
        for l in left {
            for r in right.by_ref() {
                if l >= r {
                    merged.push(l);
                    break;
                } else {
                    merged.push(r);
                }
            }
        }

        merged.reverse();
    } else if arr.len() == 2 {
        if arr[0] <= arr[1] {
            merged.push(arr[0].clone());
            merged.push(arr[1].clone());
        } else {
            merged.push(arr[1].clone());
            merged.push(arr[0].clone());
        }
    } else if arr.len() == 1 {
        merged.push(arr[0].clone())
    }

    merged
}

pub fn quick_sort<T: Clone + Ord>(arr: &mut [T]) {
    if arr.len() > 1 {
        let pivot = arr.len() - 1;
        let mut i = 0;
        let mut j = arr.len() - 2;
        while i < j {
            while arr[i] <= arr[pivot] && i < j {
                i += 1;
            }
            while arr[j] > arr[pivot] && i < j {
                j -= 1;
            }
            if arr[i] > arr[j] {
                arr.swap(i, j);
            }
        }
        if arr[i] > arr[pivot] {
            arr.swap(pivot, i);
        } else {
            i = pivot;
        }

        quick_sort(&mut arr[0..i]);
        quick_sort(&mut arr[i + 1..]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ARR: [u8; 8] = [12, 10, 24, 16, 36, 23, 15, 35];

    #[test]
    fn test_selection_sort() {
        let mut arr = ARR;
        selection_sort(&mut arr);

        for w in arr.windows(2) {
            assert!(w[0] < w[1]);
        }
    }

    #[test]
    fn test_merge_sort() {
        let arr = ARR;
        let sorted = merge_sort(&arr);

        for w in sorted.windows(2) {
            assert!(w[0] < w[1]);
        }
    }

    #[test]
    fn test_quick_sort() {
        let mut arr = ARR;
        quick_sort(&mut arr);

        for w in arr.windows(2) {
            assert!(w[0] < w[1]);
        }
    }
}
