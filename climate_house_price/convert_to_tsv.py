import os


CLIMATE_DATA_DIR = "NOAA Climate Data"

for fname in ["cdd", "hdd", "pcpn", "tmax", "tmin", "tavg"]:
    write_lines = [
        "\t".join(
            [
                "Code",
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ]
        )
    ]
    with open(os.path.join(CLIMATE_DATA_DIR, fname + ".txt")) as f:
        lines = f.readlines()
        for line in lines:
            write_lines.append("\t".join(line.split()))

    with open(os.path.join(CLIMATE_DATA_DIR, fname + ".tsv"), "w") as f:
        f.write("\n".join(write_lines))
