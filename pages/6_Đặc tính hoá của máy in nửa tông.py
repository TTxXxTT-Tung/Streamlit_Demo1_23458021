import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from utils.halftone import load_halftone, get_trc, evaluate_pair, evaluate_all
from utils.color_math import xyz100_to_srgb_preview
from utils.spectral import spectrum_array_to_xyz
from utils.ui import DATA_DIR, project_header, theory_box, footer_note

st.set_page_config(page_title="6 - Halftone", page_icon="6️⃣", layout="wide")
project_header("6. Đặc tính hóa máy in nửa tông", "Mô hình Magenta–Yellow từ dữ liệu halftone.mat")

theory_box([
    "Dữ liệu gồm phổ nền giấy W, 120 phổ trộn Magenta–Yellow và nồng độ/diện tích mục tiêu tương ứng.",
    "TRC dùng để ánh xạ diện tích tram mục tiêu sang diện tích hiệu dụng. Project mẫu khớp TRC bằng đa thức bậc 3.",
    r"Demichel hai mực: $A_M=a_M(1-a_Y)$, $A_Y=a_Y(1-a_M)$, $A_O=a_Ma_Y$, $A_W=(1-a_M)(1-a_Y)$.",
    r"Phổ dự báo dùng dạng Yule–Nielsen/Neugebauer: $R=[\sum A_i R_i^{1/n}]^n$. Trong project mẫu, $n=20$ là tham số của bộ dữ liệu, không phải hằng số phổ quát.",
])

path = DATA_DIR / "halftone.mat"
data = load_halftone(path)
mode = st.sidebar.radio("Nội dung", [
    "Ma trận màu M–Y",
    "Phổ Yellow và Magenta",
    "Đường cong tái tạo tông (TRC)",
    "Dự báo trộn M–Y",
    "Đánh giá toàn bộ mô hình",
])

if mode == "Ma trận màu M–Y":
    size = 36
    image = np.zeros((size * 11, size * 11, 3), dtype=np.uint8)
    levels = [round(i / 10, 1) for i in range(11)]
    for i_m, m in enumerate(levels):
        for i_y, y in enumerate(levels):
            spec = data.spectra[(m, y)]
            xyz = spectrum_array_to_xyz(spec, "D65")
            rgb, _ = xyz100_to_srgb_preview(xyz)
            rgb255 = np.round(rgb * 255).astype(np.uint8)
            image[i_m*size:(i_m+1)*size, i_y*size:(i_y+1)*size] = rgb255
    image = image[::-1]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    ticks = list(range(size//2, size*11, size))
    ax.set_xticks(ticks, [f"{v:.1f}" for v in levels])
    ax.set_yticks(ticks, [f"{v:.1f}" for v in levels[::-1]])
    ax.set_xlabel("Yellow target area")
    ax.set_ylabel("Magenta target area")
    ax.set_title("Mô phỏng màu hiển thị của lưới Magenta–Yellow")
    st.pyplot(fig, clear_figure=True)
    st.caption("Màu trên màn hình là mô phỏng sRGB của phổ đo; không phải bản thân phổ phản xạ.")

elif mode == "Phổ Yellow và Magenta":
    wave = np.arange(400, 701, 10)
    c1, c2 = st.columns(2)
    for col, ramp, title in [(c1, data.yellow_ramp, "Yellow"), (c2, data.magenta_ramp, "Magenta")]:
        with col:
            fig, ax = plt.subplots()
            for i in range(11):
                ax.plot(wave, ramp[i], label=f"a={i/10:.1f}")
            ax.set_xlim(400, 700)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Bước sóng (nm)")
            ax.set_ylabel("Phản xạ R(λ)")
            ax.set_title(f"Phổ {title} theo mức tram")
            ax.grid(True, alpha=0.2)
            st.pyplot(fig, clear_figure=True)

elif mode == "Đường cong tái tạo tông (TRC)":
    n = st.sidebar.number_input("Hệ số Yule–Nielsen n", min_value=0.5, max_value=50.0, value=20.0, step=0.5)
    target = np.linspace(0, 1, 11)
    c1, c2 = st.columns(2)
    for col, ramp, title in [(c1, data.yellow_ramp, "Yellow"), (c2, data.magenta_ramp, "Magenta")]:
        actual, coeff = get_trc(target, ramp, data.white, ramp[-1], float(n))
        x = np.linspace(0, 1, 101)
        y = np.clip(np.polyval(coeff, x), 0, 1)
        with col:
            fig, ax = plt.subplots()
            ax.plot(target, actual, "o", label="Điểm suy ra từ phổ")
            ax.plot(x, y, "-", label="Đa thức bậc 3")
            ax.plot([0, 1], [0, 1], "--", label="Lý tưởng")
            ax.set_xlabel("Diện tích mục tiêu")
            ax.set_ylabel("Diện tích hiệu dụng")
            ax.set_title(f"TRC {title}")
            ax.grid(True, alpha=0.2)
            ax.legend()
            st.pyplot(fig, clear_figure=True)
            df = pd.DataFrame({"target": target, "actual": actual, "TVI": actual-target})
            st.dataframe(df.round(4), use_container_width=True)

elif mode == "Dự báo trộn M–Y":
    levels = [round(i / 10, 1) for i in range(11)]
    m = st.sidebar.select_slider("Magenta target", options=levels, value=0.5)
    y = st.sidebar.select_slider("Yellow target", options=levels, value=0.5)
    n = st.sidebar.number_input("Hệ số Yule–Nielsen n", min_value=0.5, max_value=50.0, value=20.0, step=0.5)
    ill = st.sidebar.selectbox("Nguồn sáng để tính ΔE00", ["D65", "D50"], index=0)
    ev = evaluate_pair(data, m, y, n=float(n), illuminant=ill)

    a, b, c, d = st.columns(4)
    a.metric("aM hiệu dụng", f"{ev['a_m_actual']:.4f}")
    b.metric("aY hiệu dụng", f"{ev['a_y_actual']:.4f}")
    c.metric("RMSE phổ", f"{ev['rmse']:.5f}")
    d.metric("ΔE00", f"{ev['delta_e00']:.3f}")

    wave = np.arange(400, 701, 10)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(wave, ev["measured"], "o", label="Phổ đo")
    ax.plot(wave, ev["predicted"], "-", label="Phổ dự báo")
    ax.set_xlim(400, 700)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Bước sóng (nm)")
    ax.set_ylabel("Phản xạ")
    ax.set_title(f"M={m:.1f}, Y={y:.1f}, n={n:g}")
    ax.grid(True, alpha=0.2)
    ax.legend()
    st.pyplot(fig, clear_figure=True)

    areas = pd.DataFrame({"Vùng": list(ev["areas"].keys()), "Diện tích": list(ev["areas"].values())})
    st.dataframe(areas, use_container_width=True)
    st.caption(f"Tổng diện tích Demichel = {areas['Diện tích'].sum():.6f} (phải xấp xỉ 1).")

elif mode == "Đánh giá toàn bộ mô hình":
    n = st.sidebar.number_input("Hệ số Yule–Nielsen n", min_value=0.5, max_value=50.0, value=20.0, step=0.5)
    ill = st.sidebar.selectbox("Nguồn sáng", ["D65", "D50"], index=0)
    if st.button("Chạy đánh giá 120 ô", type="primary"):
        rows = evaluate_all(data, n=float(n), illuminant=ill)
        df = pd.DataFrame(rows, columns=["Magenta", "Yellow", "RMSE_spectral", "DeltaE00"])
        st.session_state["halftone_eval"] = df
    if "halftone_eval" in st.session_state:
        df = st.session_state["halftone_eval"]
        c1, c2, c3 = st.columns(3)
        c1.metric("ΔE00 trung bình", f"{df['DeltaE00'].mean():.3f}")
        c2.metric("ΔE00 lớn nhất", f"{df['DeltaE00'].max():.3f}")
        c3.metric("RMSE phổ trung bình", f"{df['RMSE_spectral'].mean():.5f}")
        st.dataframe(df.sort_values("DeltaE00", ascending=False), use_container_width=True, height=420)
        fig, ax = plt.subplots()
        sc = ax.scatter(df["Yellow"], df["Magenta"], c=df["DeltaE00"], s=80)
        fig.colorbar(sc, ax=ax, label="ΔE00")
        ax.set_xlabel("Yellow")
        ax.set_ylabel("Magenta")
        ax.set_title("Bản đồ sai lệch mô hình theo lưới M–Y")
        st.pyplot(fig, clear_figure=True)

footer_note()
