import scikit_build_example as sb
import os


freq = 5.0
t_start = 0.0
t_end = 1.0
num_samples = 1000

sine = sb.generate_sin(freq, t_start, t_end, num_samples)
cosine = sb.generate_cos(freq, t_start, t_end, num_samples)
square = sb.generate_square(freq, t_start, t_end, num_samples)
sawtooth = sb.generate_sawtooth(freq, t_start, t_end, num_samples)


dt = (t_end - t_start) / (num_samples - 1)
time = [t_start + i * dt for i in range(num_samples)]


kernel_1d = [1/3, 1/3, 1/3]
filtered_1d = sb.filter_1d(sine, kernel_1d)


image_2d = [[(i + j) % 255 for j in range(100)] for i in range(100)]
kernel_2d = [
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9]
]
filtered_2d = sb.filter_2d(image_2d, kernel_2d)


dft_y = sb.dft(sine)
idft_y = sb.idft(dft_y)


def save_signal(filename, x, y):
    with open(filename, "w") as f:
        for xi, yi in zip(x, y):
            f.write(f"{xi} {yi}\n")

save_signal("sin.dat", time, sine)
save_signal("cosine.dat", time, cosine)
save_signal("square.dat", time, square)
save_signal("sawtooth.dat", time, sawtooth)
save_signal("filtered_1d.dat", time, filtered_1d)
save_signal("idft.dat", time, idft_y)
with open("dft.dat", "w") as f:
    for i, val in enumerate(dft_y):
        f.write(f"{i} {abs(val)}\n")


with open("filtered_2d.dat", "w") as f:
    for row in filtered_2d:
        f.write(" ".join(str(x) for x in row) + "\n")


with open("plot_all.gp", "w") as f:
    f.write("""
set terminal wxt size 1600,800 enhanced
set grid
set multiplot layout 2,4 title "Sygnaly i Filtracje"
unset key

set title "Sinus"
set ylabel "Amplitude"
plot 'sin.dat' with lines notitle

set title "Cosinus"
set ylabel "Amplitude"
plot 'cosine.dat' with lines notitle

set title "Prostokąt"
set ylabel "Amplitude"
plot 'square.dat' with lines notitle

set title "Piła"
set ylabel "Amplitude"
plot 'sawtooth.dat' with lines notitle

set title "DFT"
set ylabel "Amplitude"
plot 'dft.dat' with lines notitle

set title "IDFT"
set ylabel "Amplitude"
plot 'idft.dat' with lines notitle
            
set title "Filtracja 1D"
set ylabel "Amplitude"
plot 'filtered_1d.dat' with lines notitle

set title "Filtracja 2D"
set ylabel "Amplitude"
plot 'filtered_2d.dat' matrix with image notitle

unset multiplot
pause -1
""")


os.system("start gnuplot -persist plot_all.gp")


