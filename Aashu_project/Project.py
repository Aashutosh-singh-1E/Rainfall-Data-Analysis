from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CSV_FILE = Path(__file__).with_name("F_project.csv")
HEAVY_RAINFALL_MM = 200


def load_data(file_path: Path) -> pd.DataFrame:
    """Load rainfall data and skip rows with invalid months or rainfall values."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    raw_data = pd.read_csv(file_path)
    if raw_data.empty:
        raise ValueError(f"No data rows found in {file_path.name}.")

    raw_data.columns = [str(column).strip().lower() for column in raw_data.columns]
    rainfall_column = next(
        (name for name in ("rainfall_mm", "rainfall") if name in raw_data.columns),
        None,
    )
    month_column = next(
        (name for name in ("month", "date") if name in raw_data.columns),
        None,
    )
    if rainfall_column is None or month_column is None:
        raise ValueError(
            "The CSV must contain Month and Rainfall_mm columns "
            "(a date column can be used instead of Month)."
        )

    valid_rows = []
    skipped_rows = 0
    for row_number, (_, row) in enumerate(raw_data.iterrows(), start=1):
        rainfall_value = row[rainfall_column]
        rainfall = pd.to_numeric(rainfall_value, errors="coerce")
        if pd.isna(rainfall) or not np.isfinite(rainfall):
            print(
                f"Row {row_number} skipped: invalid Rainfall "
                f"{str(rainfall_value)!r}."
            )
            skipped_rows += 1
            continue
        if rainfall < 0:
            print(f"Row {row_number} skipped: Rainfall cannot be negative.")
            skipped_rows += 1
            continue

        month_value = row[month_column]
        if month_column == "date":
            parsed_date = pd.to_datetime(month_value, errors="coerce")
            if pd.isna(parsed_date):
                print(f"Row {row_number} skipped: invalid date {str(month_value)!r}.")
                skipped_rows += 1
                continue
            month = parsed_date.strftime("%b")
        elif pd.isna(month_value) or not str(month_value).strip():
            print(f"Row {row_number} skipped: Month is missing.")
            skipped_rows += 1
            continue
        else:
            month = str(month_value).strip()

        valid_rows.append({"Month": month, "Rainfall_mm": float(rainfall)})

    if not valid_rows:
        raise ValueError("No valid rainfall rows were found in the CSV.")

    data = pd.DataFrame(valid_rows)
    if skipped_rows:
        print(
            f"Valid rows: {len(data)} | "
            f"Total: {format_rainfall(data['Rainfall_mm'].sum())} mm"
        )
    return data


def format_rainfall(value: float) -> str:
    """Format whole rainfall amounts without unnecessary decimal places."""
    return str(int(value)) if float(value).is_integer() else f"{value:.2f}"


def rainfall_statistics(data: pd.DataFrame) -> dict[str, object]:
    """Calculate the required rainfall statistics."""
    rainfall = data["Rainfall_mm"]
    maximum_index = rainfall.idxmax()
    minimum_index = rainfall.idxmin()

    return {
        "total": float(rainfall.sum()),
        "mean": float(rainfall.mean()),
        "maximum_month": str(data.loc[maximum_index, "Month"]),
        "maximum": float(rainfall.loc[maximum_index]),
        "minimum_month": str(data.loc[minimum_index, "Month"]),
        "minimum": float(rainfall.loc[minimum_index]),
        "standard_deviation": float(np.std(rainfall, ddof=0)),
        "heavy_months": data.loc[
            rainfall > HEAVY_RAINFALL_MM, "Month"
        ].tolist(),
    }


def show_statistics(data: pd.DataFrame) -> None:
    """Print totals, averages, range, population deviation, and heavy months."""
    stats = rainfall_statistics(data)
    print(f"Total: {format_rainfall(stats['total'])} mm | Mean: {stats['mean']:.2f}")
    print(
        f"Max: {stats['maximum_month']} "
        f"({format_rainfall(stats['maximum'])}) | "
        f"Min: {stats['minimum_month']} "
        f"({format_rainfall(stats['minimum'])})"
    )
    print(f"Std Dev: {stats['standard_deviation']:.2f}")
    if stats["heavy_months"]:
        print(f"Heavy rainfall months: {', '.join(stats['heavy_months'])}")
    else:
        print("No month above 200 mm")


def compare_with_average(data: pd.DataFrame) -> None:
    """Print months above and below the average rainfall."""
    average = data["Rainfall_mm"].mean()
    above_average = data.loc[data["Rainfall_mm"] > average]
    below_average = data.loc[data["Rainfall_mm"] < average]

    print(f"Average rainfall: {average:.2f} mm")
    if above_average.empty:
        print("Above average: None")
    else:
        months = " | ".join(
            f"{row.Month} ({format_rainfall(row.Rainfall_mm)})"
            for row in above_average.itertuples(index=False)
        )
        print(f"Above average ({average:.2f}): {months}")

    if below_average.empty:
        print("Below average: None")
    else:
        months = " | ".join(
            f"{row.Month} ({format_rainfall(row.Rainfall_mm)})"
            for row in below_average.itertuples(index=False)
        )
        print(f"Below average: {months}")


def plot_rainfall(data: pd.DataFrame, include_average: bool = False) -> None:
    """Display labeled bar and line charts for monthly rainfall."""
    months = data["Month"].tolist()
    rainfall = data["Rainfall_mm"].tolist()
    average = float(np.mean(rainfall))

    figure, (bar_axis, line_axis) = plt.subplots(1, 2, figsize=(12, 5))
    bar_axis.bar(months, rainfall, color="steelblue")
    bar_axis.set_title("Monthly Rainfall")
    bar_axis.set_xlabel("Month")
    bar_axis.set_ylabel("Rainfall (mm)")
    bar_axis.tick_params(axis="x", rotation=45)

    line_axis.plot(months, rainfall, marker="o", color="darkorange")
    line_axis.set_title("Monthly Rainfall Trend")
    line_axis.set_xlabel("Month")
    line_axis.set_ylabel("Rainfall (mm)")
    line_axis.tick_params(axis="x", rotation=45)

    if include_average:
        bar_axis.axhline(
            average, color="red", linestyle="--", label=f"Average ({average:.2f} mm)"
        )
        bar_axis.legend()
        line_axis.axhline(
            average, color="red", linestyle="--", label=f"Average ({average:.2f} mm)"
        )
        line_axis.legend()

    figure.tight_layout()
    plt.show()
    if include_average:
        print("Bar chart 'Monthly Rainfall' displayed.")
    else:
        print("Bar and line charts displayed.")


def run_menu(data: pd.DataFrame) -> None:
    """Run the interactive project menu."""
    while True:
        print("\n===== Rainfall Data Analyzer =====")
        print("1. Show statistics")
        print("2. Plot rainfall charts")
        print("3. Compare with average and plot")
        print("4. Exit")
        choice = input("Choose an option (1-4): ").strip()

        if choice == "1":
            show_statistics(data)
        elif choice == "2":
            plot_rainfall(data)
        elif choice == "3":
            compare_with_average(data)
            plot_rainfall(data, include_average=True)
        elif choice == "4":
            print("Goodbye.")
            return
        else:
            print("Invalid option. Please choose 1, 2, 3, or 4.")


if __name__ == "__main__":
    try:
        rainfall_data = load_data(CSV_FILE)
        run_menu(rainfall_data)
    except (FileNotFoundError, pd.errors.ParserError, ValueError) as error:
        print(f"Error: {error}")
