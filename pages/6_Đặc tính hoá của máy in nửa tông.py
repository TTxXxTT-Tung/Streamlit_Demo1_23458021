from pathlib import Path

import colour
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import streamlit as st

st.set_page_config(layout="wide")
st.title("Đặc tính hoá của máy in nửa tông")

# Luôn đọc dữ liệu cùng project, KHÔNG phụ thuộc thư mục mở CMD.
DATA_FILE = Path(__file__).resolve().parent.parent / "halftone.mat"


def load_halftone_data(mat_path):
    """Đọc phổ Y/M: 1 mẫu giấy trắng + 120 mẫu in, mỗi mẫu 31 bước sóng."""
    if not mat_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file phổ: {mat_path}")

    data = scipy.io.loadmat(str(mat_path))
    if "W" not in data or "P" not in data:
        raise ValueError("File halftone.mat phải chứa hai biến W và P.")

    W = np.asarray(data["W"], dtype=np.float64)
    P_measured = np.asarray(data["P"], dtype=np.float64)

    if W.shape != (1, 31) or P_measured.shape != (120, 31):
        raise ValueError(
            "Kích thước dữ liệu phổ không đúng: "
            f"W = {W.shape} (cần 1 × 31), "
            f"P = {P_measured.shape} (cần 120 × 31). "
            "Hãy kiểm tra file halftone.mat của project."
        )

    P = np.vstack((W, P_measured))
    if not np.all(np.isfinite(P)):
        raise ValueError("Dữ liệu phổ có giá trị trống (NaN) hoặc vô hạn.")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("Phổ phản xạ phải nằm trong khoảng 0 đến 1.")

    # Quy ước tập dữ liệu gốc: 11 mức Yellow thay đổi nhanh nhất,
    # 11 mức Magenta thay đổi chậm nhất; mẫu giấy trắng đặt ở hàng 0.
    if "concs" in data:
        concentrations = np.asarray(data["concs"], dtype=np.float64)
        if concentrations.shape != (120, 2):
            raise ValueError(
                f"Biến concs có kích thước {concentrations.shape}, "
                "cần (120, 2)."
            )
        expected = np.array(
            [(m / 10, y / 10) for m in range(11) for y in range(11)
             if not (m == 0 and y == 0)], dtype=np.float64
        )
        if not np.allclose(concentrations, expected, rtol=0, atol=1e-7):
            raise ValueError(
                "Thứ tự nồng độ trong concs khác cấu trúc 11 × 11 "
                "mà chương trình sử dụng; cần kiểm tra file dữ liệu."
            )

    yellow = P[:11, :]
    magenta = P[::11, :]
    return P, yellow, magenta


def ConvertToTristimulusValues(reflection, cmfs, illuminant):
    """Chuyển phổ phản xạ 400–700 nm thành RGB để hiển thị (0–255)."""
    wavelengths = np.arange(400, 701, 10)
    sd = colour.SpectralDistribution(dict(zip(wavelengths, reflection)))
    XYZ = colour.sd_to_XYZ(sd, cmfs, illuminant)
    RGB = colour.XYZ_to_sRGB(XYZ / 100) * 255
    # Không ép uint8 trực tiếp vì giá trị ngoài gamut có thể bị tràn số.
    return np.rint(np.clip(RGB, 0, 255)).astype(np.uint8)


def gettrc(dig, R, W, Solid, n):
    """Đường cong tái tạo tông theo công thức Yule–Nielsen gốc (11.5)."""
    if n <= 0:
        raise ValueError("Hệ số n phải lớn hơn 0.")
    R_n = np.asarray(R, dtype=np.float64) ** (1 / n)
    W_n = np.asarray(W, dtype=np.float64) ** (1 / n)
    Solid_n = np.asarray(Solid, dtype=np.float64) ** (1 / n)
    delta = Solid_n - W_n
    denominator = np.sum(delta * delta)
    if np.isclose(denominator, 0):
        raise ValueError("Phổ màu 100% và phổ giấy trắng trùng nhau; không thể tính TRC.")
    c = np.array([
        1 - np.sum((Solid_n - row) * delta) / denominator
        for row in R_n
    ])
    p = np.polyfit(dig, c, 3)
    return c, p


try:
    P, yellow, magenta = load_halftone_data(DATA_FILE)
except (FileNotFoundError, ValueError, OSError) as exc:
    st.error(f"Không thể nạp dữ liệu đặc tính hoá: {exc}")
    st.info("Hãy đặt file halftone.mat hợp lệ cùng cấp với 'Khoa học Màu sắc.py'.")
    st.stop()

chon_item = st.sidebar.radio(
    "Chọn các mục sau",
    (
        "Ảnh trộn màu vàng và tím",
        "Phổ màu vàng và tím trên nền trắng",
        "Đường cong tái tạo tông màu vàng và tím",
        "Trộn hai màu vàng và tím",
    ),
)

if chon_item == "Ảnh trộn màu vàng và tím":
    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    illuminant = colour.SDS_ILLUMINANTS["D65"]
    SIZE = 40
    image = np.zeros((SIZE * 11, SIZE * 11, 3), dtype=np.uint8)
    k = 0
    for x in range(11):
        for y in range(11):
            reflection = P[k, :]
            RGB = ConvertToTristimulusValues(reflection, cmfs, illuminant)
            image[x * SIZE:(x + 1) * SIZE, y * SIZE:(y + 1) * SIZE, :] = RGB
            k += 1

    image = image[::-1, :, :]
    _, col, _ = st.columns([1, 3, 1])
    with col:
        fig, ax = plt.subplots()
        ax.imshow(image)
        ticks = list(range(20, 440, 40))
        levels = [round(i / 10, 1) for i in range(11)]
        ax.set_xticks(ticks, levels)
        ax.set_yticks(ticks, list(reversed(levels)))
        ax.set_xlabel("Yellow")
        ax.set_ylabel("Magenta")
        ax.set_title("Mix Yellow and Magenta Colors")
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

elif chon_item == "Phổ màu vàng và tím trên nền trắng":
    wavelengths = np.arange(400, 701, 10)
    col1, col2 = st.columns(2)
    for col, spectra, color, label in (
        (col1, yellow, "#FF8000", "vàng"),
        (col2, magenta, "#FF00FF", "tím"),
    ):
        with col:
            fig, ax = plt.subplots(figsize=(5, 5))
            for row in spectra:
                ax.plot(wavelengths, row, color=color)
            ax.set_yticks(np.arange(0, 1.01, 0.1))
            ax.set_xlim(400, 700)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Bước sóng (nm)")
            ax.set_ylabel("Phổ phản xạ")
            ax.set_title(f"Phổ màu {label} trên nền trắng với a = 0:0.1:1")
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)

elif chon_item == "Đường cong tái tạo tông màu vàng và tím":
    n = 20
    a_target = np.linspace(0, 1, 11)
    col1, col2 = st.columns(2)
    for col, spectra, color, label in (
        (col1, yellow, "#FF8000", "vàng"),
        (col2, magenta, "#FF00FF", "tím"),
    ):
        with col:
            a_actual, p = gettrc(
                a_target, spectra, spectra[0, :], spectra[10, :], n
            )
            x_ve = np.linspace(0, 1, 101)
            y_ve = np.polyval(p, x_ve)
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot(a_target, a_actual, "o", color=color, label="Số liệu tính")
            ax.plot(x_ve, y_ve, color="black", label="Đa thức bậc 3")
            ax.set_xlabel("Giá trị mục tiêu của a")
            ax.set_ylabel("Giá trị thực tế của a")
            ax.set_title(f"Đường cong tái tạo tông màu {label}")
            ax.legend()
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)

elif chon_item == "Trộn hai màu vàng và tím":
    _, col, _ = st.columns([1, 3, 1])
    with col:
        a_mag_target = st.number_input(
            "Nhập mục tiêu a của màu tím [0:0.1:1]",
            min_value=0.0, max_value=1.0, value=0.0, step=0.1, format="%.1f",
        )
        a_yel_target = st.number_input(
            "Nhập mục tiêu a của màu vàng [0:0.1:1]",
            min_value=0.0, max_value=1.0, value=0.0, step=0.1, format="%.1f",
        )
        if st.button("Dự báo"):
            i_mag = round(a_mag_target * 10)
            i_yel = round(a_yel_target * 10)
            if (abs(a_mag_target * 10 - i_mag) > 1e-6
                    or abs(a_yel_target * 10 - i_yel) > 1e-6):
                st.warning("Chỉ nhập các mức 0.0, 0.1, 0.2, …, 1.0.")
            else:
                i = i_mag * 11 + i_yel
                measured = P[i, :]
                n = 20
                a_target = np.linspace(0, 1, 11)
                _, p_mag = gettrc(a_target, magenta, magenta[0, :], magenta[10, :], n)
                _, p_yel = gettrc(a_target, yellow, yellow[0, :], yellow[10, :], n)

                # Đa thức bậc 3 có thể nhích ra ngoài 0..1 ở hai đầu;
                # diện tích hạt trame thực tế luôn nằm trong 0..1.
                a_mag_actual = float(np.clip(np.polyval(p_mag, a_mag_target), 0, 1))
                a_yel_actual = float(np.clip(np.polyval(p_yel, a_yel_target), 0, 1))
                Am = a_mag_actual * (1 - a_yel_actual)
                Ay = a_yel_actual * (1 - a_mag_actual)
                Aw = (1 - a_mag_actual) * (1 - a_yel_actual)
                Ao = a_mag_actual * a_yel_actual

                overlap = P[120, :]
                WW = P[0, :]
                pred = (
                    Am * magenta[10, :] ** (1 / n)
                    + Ay * yellow[10, :] ** (1 / n)
                    + Ao * overlap ** (1 / n)
                    + Aw * WW ** (1 / n)
                ) ** n

                wavelengths = np.arange(400, 701, 10)
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.plot(wavelengths, measured, "ro", label="Phổ đo thực tế")
                ax.plot(wavelengths, pred, "b", label="Phổ dự báo")
                ax.set_ylim(0, 1)
                ax.legend()
                ax.set_xlabel("Bước sóng (nm)")
                ax.set_ylabel("Phổ phản xạ")
                ax.set_title(
                    "Phổ phản xạ khi trộn hai màu tím và vàng\n"
                    f"với tỷ lệ hạt trame a_mag = {a_mag_target:.1f} "
                    f"và a_yel = {a_yel_target:.1f}"
                )
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)
