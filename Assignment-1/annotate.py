import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread("cuboid.jpg")
points = []

fig, ax = plt.subplots()
ax.imshow(img)
ax.set_title("Click 6 corners, then close the window")

def on_click(event):
    if event.xdata is None or event.ydata is None:
        return
    x, y = int(round(event.xdata)), int(round(event.ydata))
    points.append((x, y))
    print(f"Point {len(points)}: ({x}, {y})")
    ax.plot(x, y, "ro", markersize=6)
    ax.annotate(str(len(points)), (x + 10, y - 10), color="red", fontsize=12)
    fig.canvas.draw()

fig.canvas.mpl_connect("button_press_event", on_click)
plt.show()

with open("cuboid_points.txt", "w") as f:
    for x, y in points:
        f.write(f"{x} {y}\n")

print(f"Saved {len(points)} points to cuboid_points.txt")
