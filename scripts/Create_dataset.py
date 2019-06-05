import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_dataset(dataset, folder):
    plt.plot(dataset.loc[:, ["gFx", "gFy", "gFz"]])
    plt.legend(("gFx", "gFy", "gFz"))
    plt.ylabel("g-force")
    plt.xlabel("Time (s)")
    plt.savefig(folder + "g_force.png")
    plt.close()

    plt.plot(dataset.loc[:, ["ax", "ay", "az"]])
    plt.legend(("ax", "ay", "az"))
    plt.ylabel("Acceleration (m/s" + r'$^2$' + ")")
    plt.xlabel("Time (s)")
    plt.savefig(folder + "lin_acc.png")
    plt.close()

    plt.plot(dataset.loc[:, ["wx", "wy", "wz"]])
    plt.legend(("wx", "wy", "wz"))
    plt.ylabel("Rotation (rad/s)")
    plt.xlabel("Time (s)")
    plt.savefig(folder + "gyro.png")
    plt.close()

    plt.plot(dataset.loc[:, "p"])
    plt.ylabel("Pressure (hPa)")
    plt.xlabel("Time (s)")
    plt.savefig(folder + "pressure.png")
    plt.close()

    plt.plot(dataset.loc[:, ["Bx", "By", "Bz"]])
    plt.legend(("Bx", "By", "Bz"))
    plt.ylabel("Magnetic force (" + r'$\mu$' + 'T)')
    plt.xlabel("Time (s)")
    plt.savefig(folder + "magnetic.png")
    plt.close()

    plt.plot(dataset.loc[:, ["Azimuth", "Pitch", "Roll"]])
    plt.legend(("Azimuth", "Pitch", "Roll"))
    plt.ylabel("Inclination (degrees)")
    plt.xlabel("Time (s)")
    plt.savefig(folder + "inclination.png")
    plt.close()

    plt.plot(dataset.loc[:, "Gain"])
    plt.ylabel("Sound intensity (dB)")
    plt.xlabel("Time (s)")
    plt.savefig(folder + "sound.png")
    plt.close()


def make_empty_dataset(min_t, max_t, cols, delta_t):
    timestamps = np.arange(0, max_t - min_t, delta_t)
    empty_dataset = pd.DataFrame(index=timestamps, columns=cols)

    return empty_dataset


def create_dataset(df_raw, delta_t):
    min_t = min(df_raw.time)
    max_t = max(df_raw.time)
    cols = df_raw.drop(["time"], axis=1).columns

    empty_dataset = make_empty_dataset(min_t, max_t, cols, delta_t)

    for i in range(0, len(empty_dataset.index)):
        relevant_rows = df_raw[
            (df_raw["time"] - min_t >= i * delta_t) &
            (df_raw["time"] - min_t < (i + 1) * delta_t)
            ]

        for col in empty_dataset.columns:
            if len(relevant_rows) > 0:
                empty_dataset.loc[empty_dataset.index[i], col] = np.average(relevant_rows[col])
            else:
                raise ValueError("No relevant rows.")

    return empty_dataset


def preprocess(data):
    df_raw = data.drop(["Unnamed: 18"], axis=1)
    df_raw = df_raw[df_raw.Gain != "-∞"]
    df_raw = df_raw.astype("float64")

    return df_raw


def main():
    # Variables
    delta_t_list = [0.25, 6]
    folders = []
    for i in range(len(delta_t_list)):
        folders.append("../output/" + str(int(delta_t_list[i]*1000)) + "ms_")

    data = pd.read_csv("../data/first_data")

    print(folders)

    df_raw = preprocess(data)

    for delta_t, folder in zip(delta_t_list, folders):
        dataset = create_dataset(df_raw, delta_t)

        plot_dataset(dataset, folder)


if __name__ == '__main__':
    main()
