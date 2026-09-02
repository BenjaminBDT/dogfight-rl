#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Align {
    Left,
    Right,
}

#[derive(Debug, Clone)]
pub struct Cell {
    pub text: String,
    pub align: Align,
}

impl Cell {
    pub fn left(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            align: Align::Left,
        }
    }

    pub fn right(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            align: Align::Right,
        }
    }
}

pub fn render_rows(rows: &[Vec<Cell>], column_gap: usize) -> String {
    render_rows_with_min_widths(rows, column_gap, &[])
}

pub fn render_rows_with_min_widths(
    rows: &[Vec<Cell>],
    column_gap: usize,
    min_widths: &[usize],
) -> String {
    let column_count = rows.iter().map(Vec::len).max().unwrap_or(0);
    if column_count == 0 {
        return String::new();
    }

    let mut widths = vec![0usize; column_count];
    for row in rows {
        for (index, cell) in row.iter().enumerate() {
            widths[index] = widths[index].max(cell.text.chars().count());
        }
    }
    for (index, width) in min_widths.iter().copied().enumerate() {
        if let Some(current) = widths.get_mut(index) {
            *current = (*current).max(width);
        }
    }

    let gap = " ".repeat(column_gap);
    rows.iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(index, cell)| match cell.align {
                    Align::Left => format!("{:<width$}", cell.text, width = widths[index]),
                    Align::Right => format!("{:>width$}", cell.text, width = widths[index]),
                })
                .collect::<Vec<_>>()
                .join(&gap)
                .trim_end()
                .to_string()
        })
        .collect::<Vec<_>>()
        .join("\n")
}
