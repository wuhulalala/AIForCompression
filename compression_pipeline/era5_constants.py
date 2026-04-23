VNAMES = {
    "pressure": ["z", "q", "u", "v", "t", "r", "w"],
    "single": ["v10", "u10", "v100", "u100", "t2m", "tcc", "sp", "tp", "msl"],
}

PRESSURE_LEVELS = [
    1000.0, 975.0, 950.0, 925.0, 900.0, 875.0, 850.0, 825.0, 800.0,
    775.0, 750.0, 700.0, 650.0, 600.0, 550.0, 500.0, 450.0, 400.0,
    350.0, 300.0, 250.0, 225.0, 200.0, 175.0, 150.0, 125.0, 100.0,
    70.0, 50.0, 30.0, 20.0, 10.0, 7.0, 5.0, 3.0, 2.0, 1.0,
]

ERA5_CHANNELS = len(VNAMES["pressure"]) * len(PRESSURE_LEVELS) + len(VNAMES["single"])


def era5_channel_names() -> list[str]:
    names: list[str] = []
    for vname in VNAMES["pressure"]:
        names.extend(f"{vname}_{level:g}" for level in PRESSURE_LEVELS)
    names.extend(VNAMES["single"])
    return names

